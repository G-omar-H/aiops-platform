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

	"github.com/gin-gonic/gin"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	swaggerFiles "github.com/swaggo/files"
	ginSwagger "github.com/swaggo/gin-swagger"
	"go.uber.org/zap"
	"gorm.io/gorm"

	"auto-remediation/config"
	"auto-remediation/internal/handlers"
	"auto-remediation/internal/middleware"
	"auto-remediation/internal/models"
	"auto-remediation/internal/services"
	"auto-remediation/internal/storage"
	"auto-remediation/internal/workflow"
)

// @title AIOps Auto-Remediation Service API
// @version 1.0
// @description Enterprise auto-remediation service for autonomous incident response
// @termsOfService https://aiops-platform.com/terms
// @contact.name AIOps Platform Support
// @contact.url https://aiops-platform.com/support
// @contact.email support@aiops-platform.com
// @license.name MIT
// @license.url https://opensource.org/licenses/MIT
// @host localhost:8080
// @BasePath /api/v1
// @securityDefinitions.apikey ApiKeyAuth
// @in header
// @name Authorization

var (
	// Prometheus metrics
	remediationsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "auto_remediation_total",
			Help: "Total number of remediation attempts",
		},
		[]string{"action_type", "status", "system_id"},
	)

	remediationDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "auto_remediation_duration_seconds",
			Help:    "Duration of remediation actions",
			Buckets: prometheus.ExponentialBuckets(0.1, 2, 10),
		},
		[]string{"action_type", "system_id"},
	)

	activeRemediations = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Name: "auto_remediation_active",
			Help: "Number of currently active remediations",
		},
	)

	rollbacksTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "auto_remediation_rollbacks_total",
			Help: "Total number of rollbacks performed",
		},
		[]string{"action_type", "reason"},
	)

	safetyChecksTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "auto_remediation_safety_checks_total",
			Help: "Total number of safety checks performed",
		},
		[]string{"check_type", "result"},
	)
)

func init() {
	// Register Prometheus metrics
	prometheus.MustRegister(remediationsTotal)
	prometheus.MustRegister(remediationDuration)
	prometheus.MustRegister(activeRemediations)
	prometheus.MustRegister(rollbacksTotal)
	prometheus.MustRegister(safetyChecksTotal)
}

func main() {
	// Load configuration
	cfg, err := config.Load()
	if err != nil {
		log.Fatalf("Failed to load configuration: %v", err)
	}

	// Setup logger
	logger, err := setupLogger(cfg.Environment)
	if err != nil {
		log.Fatalf("Failed to setup logger: %v", err)
	}
	defer logger.Sync()

	logger.Info("🔧 Starting Auto-Remediation Service",
		zap.String("version", "1.0.0"),
		zap.String("environment", cfg.Environment),
		zap.Int("port", cfg.Server.Port),
	)

	// Initialize database
	db, err := storage.Initialize(cfg.Database.URL)
	if err != nil {
		logger.Fatal("Failed to initialize database", zap.Error(err))
	}

	// Auto-migrate database schemas
	if err := db.AutoMigrate(
		&models.RemediationAction{},
		&models.WorkflowExecution{},
		&models.SafetyCheck{},
		&models.RollbackPlan{},
		&models.ActionTemplate{},
		&models.SystemIntegration{},
	); err != nil {
		logger.Fatal("Failed to migrate database", zap.Error(err))
	}

	// Initialize services
	workflowEngine := workflow.NewEngine(cfg, logger, db)
	remediationService := services.NewRemediationService(cfg, logger, db, workflowEngine)
	safetyService := services.NewSafetyService(cfg, logger, db)
	integrationService := services.NewIntegrationService(cfg, logger)
	auditService := services.NewAuditService(cfg, logger, db)

	// Initialize all services
	if err := initializeServices(
		remediationService,
		safetyService,
		integrationService,
		auditService,
		workflowEngine,
	); err != nil {
		logger.Fatal("Failed to initialize services", zap.Error(err))
	}

	// Setup HTTP server
	router := setupRouter(cfg, logger, remediationService, safetyService, integrationService, auditService)

	server := &http.Server{
		Addr:           fmt.Sprintf(":%d", cfg.Server.Port),
		Handler:        router,
		ReadTimeout:    time.Duration(cfg.Server.ReadTimeout) * time.Second,
		WriteTimeout:   time.Duration(cfg.Server.WriteTimeout) * time.Second,
		IdleTimeout:    time.Duration(cfg.Server.IdleTimeout) * time.Second,
		MaxHeaderBytes: cfg.Server.MaxHeaderBytes,
	}

	// Start server in goroutine
	go func() {
		logger.Info("🚀 Auto-Remediation Service started",
			zap.String("address", server.Addr),
			zap.String("docs", fmt.Sprintf("http://localhost:%d/docs", cfg.Server.Port)),
		)

		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Fatal("Failed to start server", zap.Error(err))
		}
	}()

	// Start background services
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go startBackgroundServices(ctx, logger, remediationService, safetyService)

	// Wait for interrupt signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	logger.Info("🛑 Shutting down Auto-Remediation Service...")

	// Graceful shutdown
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer shutdownCancel()

	if err := server.Shutdown(shutdownCtx); err != nil {
		logger.Error("Server forced to shutdown", zap.Error(err))
	}

	// Cancel background services
	cancel()

	logger.Info("✅ Auto-Remediation Service shutdown completed")
}

func setupLogger(environment string) (*zap.Logger, error) {
	if environment == "production" {
		return zap.NewProduction()
	}
	return zap.NewDevelopment()
}

func initializeServices(services ...interface{ Initialize() error }) error {
	for _, service := range services {
		if err := service.Initialize(); err != nil {
			return err
		}
	}
	return nil
}

func setupRouter(
	cfg *config.Config,
	logger *zap.Logger,
	remediationService *services.RemediationService,
	safetyService *services.SafetyService,
	integrationService *services.IntegrationService,
	auditService *services.AuditService,
) *gin.Engine {
	// Set Gin mode
	if cfg.Environment == "production" {
		gin.SetMode(gin.ReleaseMode)
	}

	router := gin.New()

	// Global middleware
	router.Use(middleware.Logger(logger))
	router.Use(middleware.Recovery(logger))
	router.Use(middleware.CORS())
	router.Use(middleware.Security())
	router.Use(middleware.Metrics())

	// Prometheus metrics endpoint
	router.GET("/metrics", gin.WrapH(promhttp.Handler()))

	// Health check endpoints
	router.GET("/health", handlers.HealthCheck(cfg, logger))
	router.GET("/health/live", handlers.LivenessCheck())
	router.GET("/health/ready", handlers.ReadinessCheck(remediationService, safetyService))

	// Swagger documentation
	if cfg.Environment != "production" {
		router.GET("/docs/*any", ginSwagger.WrapHandler(swaggerFiles.Handler))
	}

	// API routes with authentication
	api := router.Group("/api/v1")
	api.Use(middleware.Authentication(cfg.Security.JWTSecret))
	api.Use(middleware.RateLimit(cfg.RateLimit.RequestsPerMinute))

	// Remediation endpoints
	remediationHandler := handlers.NewRemediationHandler(remediationService, safetyService, auditService, logger)
	remediation := api.Group("/remediation")
	{
		remediation.POST("/execute", remediationHandler.ExecuteRemediation)
		remediation.POST("/batch", remediationHandler.BatchExecuteRemediation)
		remediation.GET("/status/:id", remediationHandler.GetRemediationStatus)
		remediation.POST("/rollback/:id", remediationHandler.RollbackRemediation)
		remediation.GET("/history", remediationHandler.GetRemediationHistory)
		remediation.POST("/approve/:id", remediationHandler.ApproveRemediation)
		remediation.POST("/cancel/:id", remediationHandler.CancelRemediation)
	}

	// Workflow endpoints
	workflowHandler := handlers.NewWorkflowHandler(remediationService, logger)
	workflow := api.Group("/workflow")
	{
		workflow.POST("/create", workflowHandler.CreateWorkflow)
		workflow.GET("/templates", workflowHandler.ListWorkflowTemplates)
		workflow.GET("/:id", workflowHandler.GetWorkflow)
		workflow.PUT("/:id", workflowHandler.UpdateWorkflow)
		workflow.DELETE("/:id", workflowHandler.DeleteWorkflow)
		workflow.POST("/:id/execute", workflowHandler.ExecuteWorkflow)
	}

	// Safety endpoints
	safetyHandler := handlers.NewSafetyHandler(safetyService, logger)
	safety := api.Group("/safety")
	{
		safety.GET("/checks", safetyHandler.ListSafetyChecks)
		safety.POST("/validate", safetyHandler.ValidateAction)
		safety.GET("/policies", safetyHandler.GetSafetyPolicies)
		safety.PUT("/policies", safetyHandler.UpdateSafetyPolicies)
		safety.POST("/emergency-stop", safetyHandler.EmergencyStop)
	}

	// Integration endpoints
	integrationHandler := handlers.NewIntegrationHandler(integrationService, logger)
	integration := api.Group("/integrations")
	{
		integration.GET("/", integrationHandler.ListIntegrations)
		integration.POST("/", integrationHandler.CreateIntegration)
		integration.GET("/:id", integrationHandler.GetIntegration)
		integration.PUT("/:id", integrationHandler.UpdateIntegration)
		integration.DELETE("/:id", integrationHandler.DeleteIntegration)
		integration.POST("/:id/test", integrationHandler.TestIntegration)
	}

	// Audit endpoints
	auditHandler := handlers.NewAuditHandler(auditService, logger)
	audit := api.Group("/audit")
	{
		audit.GET("/logs", auditHandler.GetAuditLogs)
		audit.GET("/reports", auditHandler.GenerateReport)
		audit.GET("/compliance", auditHandler.GetComplianceStatus)
	}

	// Admin endpoints (require admin role)
	admin := api.Group("/admin")
	admin.Use(middleware.RequireRole("admin"))
	{
		admin.GET("/stats", handlers.GetSystemStats(remediationService, safetyService))
		admin.POST("/maintenance", handlers.EnableMaintenanceMode)
		admin.DELETE("/maintenance", handlers.DisableMaintenanceMode)
		admin.POST("/cache/clear", handlers.ClearCache)
	}

	return router
}

func startBackgroundServices(
	ctx context.Context,
	logger *zap.Logger,
	remediationService *services.RemediationService,
	safetyService *services.SafetyService,
) {
	logger.Info("🔄 Starting background services...")

	// Start monitoring goroutines
	go remediationService.StartPeriodicCleanup(ctx)
	go safetyService.StartSafetyMonitoring(ctx)
	go startMetricsCollection(ctx, logger, remediationService)
	go startHealthMonitoring(ctx, logger)

	logger.Info("✅ Background services started")
}

func startMetricsCollection(ctx context.Context, logger *zap.Logger, remediationService *services.RemediationService) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			// Update active remediations metric
			count, err := remediationService.GetActiveRemediationCount()
			if err != nil {
				logger.Error("Failed to get active remediation count", zap.Error(err))
				continue
			}
			activeRemediations.Set(float64(count))
		}
	}
}

func startHealthMonitoring(ctx context.Context, logger *zap.Logger) {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			// Perform health checks and log status
			logger.Debug("Health monitoring tick")
		}
	}
} 