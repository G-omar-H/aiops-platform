package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-contrib/pprof"
	"github.com/gin-gonic/gin"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	swaggerFiles "github.com/swaggo/files"
	ginSwagger "github.com/swaggo/gin-swagger"

	"aiops-platform/services/api-gateway/internal/config"
	"aiops-platform/services/api-gateway/internal/handlers"
	"aiops-platform/services/api-gateway/internal/middleware"
	"aiops-platform/services/api-gateway/internal/services"
	"aiops-platform/services/api-gateway/internal/utils"
)

// @title AIOps Platform API Gateway
// @version 1.0
// @description Self-Healing Enterprise Application Monitoring Platform API Gateway
// @termsOfService http://swagger.io/terms/

// @contact.name API Support
// @contact.url http://www.aiops-platform.com/support
// @contact.email support@aiops-platform.com

// @license.name MIT
// @license.url https://opensource.org/licenses/MIT

// @host localhost:8080
// @BasePath /api/v1

// @securityDefinitions.apikey BearerAuth
// @in header
// @name Authorization
// @description Type "Bearer" followed by a space and JWT token.

func main() {
	// Initialize configuration
	cfg, err := config.Load()
	if err != nil {
		log.Fatal("Failed to load configuration:", err)
	}

	// Initialize logger
	logger := utils.NewLogger(cfg.LogLevel)
	logger.Info("Starting AIOps Platform API Gateway", "version", "1.0.0", "environment", cfg.Environment)

	// Initialize services
	serviceManager := services.NewManager(cfg, logger)
	if err := serviceManager.Initialize(); err != nil {
		logger.Error("Failed to initialize services", "error", err)
		os.Exit(1)
	}
	defer serviceManager.Shutdown()

	// Initialize handlers
	handlerManager := handlers.NewManager(serviceManager, logger)

	// Setup router
	router := setupRouter(cfg, handlerManager, logger)

	// Create HTTP server
	server := &http.Server{
		Addr:         fmt.Sprintf(":%s", cfg.Port),
		Handler:      router,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	// Start server in a goroutine
	go func() {
		logger.Info("Starting server", "port", cfg.Port, "environment", cfg.Environment)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Error("Failed to start server", "error", err)
			os.Exit(1)
		}
	}()

	// Wait for interrupt signal to gracefully shutdown the server
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	logger.Info("Shutting down server...")

	// Give outstanding requests 30 seconds to complete
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := server.Shutdown(ctx); err != nil {
		logger.Error("Server forced to shutdown", "error", err)
	}

	logger.Info("Server exited")
}

func setupRouter(cfg *config.Config, handlers *handlers.Manager, logger utils.Logger) *gin.Engine {
	// Set gin mode
	if cfg.Environment == "production" {
		gin.SetMode(gin.ReleaseMode)
	}

	router := gin.New()

	// Middleware
	router.Use(middleware.RequestID())
	router.Use(middleware.Logger(logger))
	router.Use(middleware.Recovery(logger))
	router.Use(middleware.SecurityHeaders())
	router.Use(middleware.RateLimit(cfg.RateLimit))

	// CORS configuration
	corsConfig := cors.DefaultConfig()
	corsConfig.AllowOrigins = cfg.CORS.AllowedOrigins
	corsConfig.AllowMethods = cfg.CORS.AllowedMethods
	corsConfig.AllowHeaders = cfg.CORS.AllowedHeaders
	corsConfig.AllowCredentials = true
	router.Use(cors.New(corsConfig))

	// Health check endpoint (no auth required)
	router.GET("/health", handlers.Health.Check)
	router.GET("/health/ready", handlers.Health.Ready)
	router.GET("/health/live", handlers.Health.Live)

	// Metrics endpoint for Prometheus
	router.GET("/metrics", gin.WrapH(promhttp.Handler()))

	// Profiling endpoints (only in development)
	if cfg.Environment == "development" {
		pprof.Register(router)
	}

	// API documentation
	if cfg.Swagger.Enabled {
		router.GET("/swagger/*any", ginSwagger.WrapHandler(swaggerFiles.Handler))
	}

	// API v1 routes
	v1 := router.Group("/api/v1")
	{
		// Authentication routes (no auth required)
		auth := v1.Group("/auth")
		{
			auth.POST("/login", handlers.Auth.Login)
			auth.POST("/logout", handlers.Auth.Logout)
			auth.POST("/refresh", handlers.Auth.RefreshToken)
			auth.GET("/callback", handlers.Auth.OAuthCallback)
		}

		// Protected routes
		protected := v1.Group("/")
		protected.Use(middleware.JWTAuth(cfg.JWT.Secret))
		{
			// User management
			users := protected.Group("/users")
			{
				users.GET("", handlers.Users.List)
				users.POST("", handlers.Users.Create)
				users.GET("/:id", handlers.Users.Get)
				users.PUT("/:id", handlers.Users.Update)
				users.DELETE("/:id", handlers.Users.Delete)
				users.GET("/profile", handlers.Users.Profile)
				users.PUT("/profile", handlers.Users.UpdateProfile)
			}

			// Monitoring routes
			monitoring := protected.Group("/monitoring")
			{
				monitoring.GET("/metrics", handlers.Monitoring.GetMetrics)
				monitoring.POST("/metrics", handlers.Monitoring.IngestMetrics)
				monitoring.GET("/alerts", handlers.Monitoring.GetAlerts)
				monitoring.POST("/alerts", handlers.Monitoring.CreateAlert)
				monitoring.PUT("/alerts/:id", handlers.Monitoring.UpdateAlert)
				monitoring.DELETE("/alerts/:id", handlers.Monitoring.DeleteAlert)
				monitoring.GET("/dashboards", handlers.Monitoring.GetDashboards)
				monitoring.POST("/dashboards", handlers.Monitoring.CreateDashboard)
			}

			// Prediction routes (AI/ML)
			predictions := protected.Group("/predictions")
			{
				predictions.GET("", handlers.Predictions.List)
				predictions.POST("/analyze", handlers.Predictions.Analyze)
				predictions.GET("/models", handlers.Predictions.GetModels)
				predictions.POST("/models/train", handlers.Predictions.TrainModel)
				predictions.GET("/models/:id/performance", handlers.Predictions.GetModelPerformance)
				predictions.POST("/forecast", handlers.Predictions.Forecast)
			}

			// Incident management routes
			incidents := protected.Group("/incidents")
			{
				incidents.GET("", handlers.Incidents.List)
				incidents.POST("", handlers.Incidents.Create)
				incidents.GET("/:id", handlers.Incidents.Get)
				incidents.PUT("/:id", handlers.Incidents.Update)
				incidents.DELETE("/:id", handlers.Incidents.Delete)
				incidents.POST("/:id/assign", handlers.Incidents.Assign)
				incidents.POST("/:id/resolve", handlers.Incidents.Resolve)
				incidents.GET("/:id/timeline", handlers.Incidents.GetTimeline)
				incidents.POST("/:id/comments", handlers.Incidents.AddComment)
			}

			// Remediation routes
			remediation := protected.Group("/remediation")
			{
				remediation.GET("/actions", handlers.Remediation.GetActions)
				remediation.POST("/actions", handlers.Remediation.CreateAction)
				remediation.GET("/actions/:id", handlers.Remediation.GetAction)
				remediation.PUT("/actions/:id", handlers.Remediation.UpdateAction)
				remediation.POST("/actions/:id/execute", handlers.Remediation.ExecuteAction)
				remediation.GET("/actions/:id/status", handlers.Remediation.GetActionStatus)
				remediation.GET("/playbooks", handlers.Remediation.GetPlaybooks)
				remediation.POST("/playbooks", handlers.Remediation.CreatePlaybook)
			}

			// Compliance routes
			compliance := protected.Group("/compliance")
			{
				compliance.GET("/frameworks", handlers.Compliance.GetFrameworks)
				compliance.GET("/controls", handlers.Compliance.GetControls)
				compliance.POST("/assessments", handlers.Compliance.CreateAssessment)
				compliance.GET("/assessments", handlers.Compliance.GetAssessments)
				compliance.GET("/assessments/:id", handlers.Compliance.GetAssessment)
				compliance.GET("/reports", handlers.Compliance.GetReports)
				compliance.POST("/reports/generate", handlers.Compliance.GenerateReport)
			}

			// Configuration routes
			config := protected.Group("/config")
			{
				config.GET("/settings", handlers.Config.GetSettings)
				config.PUT("/settings", handlers.Config.UpdateSettings)
				config.GET("/integrations", handlers.Config.GetIntegrations)
				config.POST("/integrations", handlers.Config.CreateIntegration)
				config.PUT("/integrations/:id", handlers.Config.UpdateIntegration)
				config.DELETE("/integrations/:id", handlers.Config.DeleteIntegration)
				config.POST("/integrations/:id/test", handlers.Config.TestIntegration)
			}

			// Analytics routes
			analytics := protected.Group("/analytics")
			{
				analytics.GET("/summary", handlers.Analytics.GetSummary)
				analytics.GET("/trends", handlers.Analytics.GetTrends)
				analytics.GET("/performance", handlers.Analytics.GetPerformance)
				analytics.GET("/cost-analysis", handlers.Analytics.GetCostAnalysis)
				analytics.GET("/reports", handlers.Analytics.GetReports)
				analytics.POST("/reports/custom", handlers.Analytics.CreateCustomReport)
			}

			// Infrastructure routes
			infrastructure := protected.Group("/infrastructure")
			{
				infrastructure.GET("/resources", handlers.Infrastructure.GetResources)
				infrastructure.GET("/topology", handlers.Infrastructure.GetTopology)
				infrastructure.GET("/health", handlers.Infrastructure.GetHealth)
				infrastructure.POST("/scan", handlers.Infrastructure.ScanInfrastructure)
				infrastructure.GET("/dependencies", handlers.Infrastructure.GetDependencies)
			}

			// AI/ML routes
			ai := protected.Group("/ai")
			{
				ai.POST("/chat", handlers.AI.Chat)
				ai.POST("/analyze-logs", handlers.AI.AnalyzeLogs)
				ai.POST("/generate-solution", handlers.AI.GenerateSolution)
				ai.POST("/explain-error", handlers.AI.ExplainError)
				ai.POST("/optimize-query", handlers.AI.OptimizeQuery)
				ai.GET("/insights", handlers.AI.GetInsights)
			}
		}

		// WebSocket routes (with auth)
		ws := v1.Group("/ws")
		ws.Use(middleware.WSAuth(cfg.JWT.Secret))
		{
			ws.GET("/events", handlers.WebSocket.Events)
			ws.GET("/metrics", handlers.WebSocket.Metrics)
			ws.GET("/logs", handlers.WebSocket.Logs)
		}
	}

	// Proxy routes for other services
	proxy := router.Group("/proxy")
	proxy.Use(middleware.JWTAuth(cfg.JWT.Secret))
	{
		// Grafana proxy
		proxy.Any("/grafana/*path", handlers.Proxy.Grafana)
		// Prometheus proxy
		proxy.Any("/prometheus/*path", handlers.Proxy.Prometheus)
		// Kibana proxy
		proxy.Any("/kibana/*path", handlers.Proxy.Kibana)
		// MLflow proxy
		proxy.Any("/mlflow/*path", handlers.Proxy.MLflow)
	}

	// 404 handler
	router.NoRoute(func(c *gin.Context) {
		c.JSON(http.StatusNotFound, gin.H{
			"error":   "Not Found",
			"message": "The requested resource was not found",
			"path":    c.Request.URL.Path,
		})
	})

	return router
} 