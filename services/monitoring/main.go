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
	"go.uber.org/zap"

	"monitoring-service/internal/config"
	"monitoring-service/internal/handlers"
	"monitoring-service/internal/middleware"
	"monitoring-service/internal/services"
	"monitoring-service/internal/storage"
	"monitoring-service/pkg/logger"
	"monitoring-service/pkg/metrics"
)

// Application version and build information
var (
	Version   = "1.0.0"
	BuildTime = "unknown"
	GitCommit = "unknown"
)

// Application holds the main application context
type Application struct {
	Config         *config.Config
	Logger         *zap.Logger
	Router         *gin.Engine
	MetricsService *services.MetricsService
	AlertService   *services.AlertService
	Storage        storage.Storage
	Server         *http.Server
}

func main() {
	// Initialize application
	app, err := initializeApplication()
	if err != nil {
		log.Fatalf("Failed to initialize application: %v", err)
	}
	defer app.cleanup()

	// Start application
	if err := app.start(); err != nil {
		app.Logger.Fatal("Failed to start application", zap.Error(err))
	}
}

// initializeApplication sets up all application components
func initializeApplication() (*Application, error) {
	// Load configuration
	cfg, err := config.Load()
	if err != nil {
		return nil, fmt.Errorf("failed to load config: %w", err)
	}

	// Initialize logger
	logger, err := logger.NewLogger(cfg.Log.Level, cfg.Log.Format)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize logger: %w", err)
	}

	logger.Info("🚀 Starting AIOps Monitoring Service",
		zap.String("version", Version),
		zap.String("build_time", BuildTime),
		zap.String("git_commit", GitCommit),
		zap.String("environment", cfg.Environment),
	)

	// Initialize storage
	storage, err := storage.NewStorage(cfg, logger)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize storage: %w", err)
	}

	// Initialize metrics registry
	metricsRegistry := metrics.NewRegistry()

	// Initialize services
	metricsService := services.NewMetricsService(cfg, logger, storage, metricsRegistry)
	alertService := services.NewAlertService(cfg, logger, storage)

	// Initialize HTTP router
	router := setupRouter(cfg, logger, metricsService, alertService)

	// Create HTTP server
	server := &http.Server{
		Addr:           fmt.Sprintf(":%d", cfg.Server.Port),
		Handler:        router,
		ReadTimeout:    time.Duration(cfg.Server.ReadTimeout) * time.Second,
		WriteTimeout:   time.Duration(cfg.Server.WriteTimeout) * time.Second,
		IdleTimeout:    time.Duration(cfg.Server.IdleTimeout) * time.Second,
		MaxHeaderBytes: cfg.Server.MaxHeaderBytes,
	}

	app := &Application{
		Config:         cfg,
		Logger:         logger,
		Router:         router,
		MetricsService: metricsService,
		AlertService:   alertService,
		Storage:        storage,
		Server:         server,
	}

	return app, nil
}

// setupRouter configures the HTTP router with all routes and middleware
func setupRouter(cfg *config.Config, logger *zap.Logger, metricsService *services.MetricsService, alertService *services.AlertService) *gin.Engine {
	// Set Gin mode
	if cfg.Environment == "production" {
		gin.SetMode(gin.ReleaseMode)
	}

	router := gin.New()

	// Add middleware
	router.Use(middleware.Logger(logger))
	router.Use(middleware.Recovery(logger))
	router.Use(middleware.CORS(cfg.Security.CORSOrigins))
	router.Use(middleware.Security())
	router.Use(middleware.RateLimiter(cfg.Security.RateLimit))

	// Prometheus metrics endpoint
	router.GET("/metrics", gin.WrapH(promhttp.Handler()))

	// Health check endpoints
	router.GET("/health", handlers.HealthCheck(cfg, logger))
	router.GET("/health/live", handlers.LivenessCheck(logger))
	router.GET("/health/ready", handlers.ReadinessCheck(logger, metricsService))

	// API routes
	v1 := router.Group("/api/v1")
	{
		// Authentication middleware for API routes
		v1.Use(middleware.Authentication(cfg.Security.JWTSecret))

		// Metrics routes
		metricsGroup := v1.Group("/metrics")
		{
			metricsGroup.POST("/collect", handlers.CollectMetrics(metricsService))
			metricsGroup.GET("/systems/:systemId", handlers.GetSystemMetrics(metricsService))
			metricsGroup.GET("/systems/:systemId/historical", handlers.GetHistoricalMetrics(metricsService))
			metricsGroup.GET("/aggregated", handlers.GetAggregatedMetrics(metricsService))
			metricsGroup.POST("/query", handlers.QueryMetrics(metricsService))
		}

		// Alerts routes
		alertsGroup := v1.Group("/alerts")
		{
			alertsGroup.GET("/", handlers.GetAlerts(alertService))
			alertsGroup.POST("/", handlers.CreateAlert(alertService))
			alertsGroup.PUT("/:alertId", handlers.UpdateAlert(alertService))
			alertsGroup.DELETE("/:alertId", handlers.DeleteAlert(alertService))
			alertsGroup.POST("/:alertId/acknowledge", handlers.AcknowledgeAlert(alertService))
		}

		// Systems routes
		systemsGroup := v1.Group("/systems")
		{
			systemsGroup.GET("/", handlers.ListSystems(metricsService))
			systemsGroup.GET("/:systemId", handlers.GetSystemDetails(metricsService))
			systemsGroup.PUT("/:systemId", handlers.UpdateSystem(metricsService))
			systemsGroup.DELETE("/:systemId", handlers.DeleteSystem(metricsService))
		}

		// Dashboards routes
		dashboardsGroup := v1.Group("/dashboards")
		{
			dashboardsGroup.GET("/", handlers.ListDashboards(metricsService))
			dashboardsGroup.POST("/", handlers.CreateDashboard(metricsService))
			dashboardsGroup.GET("/:dashboardId", handlers.GetDashboard(metricsService))
			dashboardsGroup.PUT("/:dashboardId", handlers.UpdateDashboard(metricsService))
			dashboardsGroup.DELETE("/:dashboardId", handlers.DeleteDashboard(metricsService))
		}
	}

	return router
}

// start begins the application lifecycle
func (a *Application) start() error {
	// Start background services
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Start metrics collection
	go func() {
		if err := a.MetricsService.StartCollection(ctx); err != nil {
			a.Logger.Error("Failed to start metrics collection", zap.Error(err))
		}
	}()

	// Start alert processing
	go func() {
		if err := a.AlertService.StartProcessing(ctx); err != nil {
			a.Logger.Error("Failed to start alert processing", zap.Error(err))
		}
	}()

	// Start HTTP server
	go func() {
		a.Logger.Info("🌐 HTTP server starting",
			zap.String("address", a.Server.Addr),
			zap.String("environment", a.Config.Environment),
		)

		if err := a.Server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			a.Logger.Fatal("Failed to start HTTP server", zap.Error(err))
		}
	}()

	// Wait for shutdown signal
	return a.waitForShutdown(ctx, cancel)
}

// waitForShutdown waits for termination signals and handles graceful shutdown
func (a *Application) waitForShutdown(ctx context.Context, cancel context.CancelFunc) error {
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

	sig := <-quit
	a.Logger.Info("🛑 Shutdown signal received", zap.String("signal", sig.String()))

	// Cancel context to stop background services
	cancel()

	// Shutdown HTTP server with timeout
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer shutdownCancel()

	a.Logger.Info("📝 Shutting down HTTP server...")
	if err := a.Server.Shutdown(shutdownCtx); err != nil {
		a.Logger.Error("Failed to shutdown HTTP server gracefully", zap.Error(err))
		return err
	}

	a.Logger.Info("✅ Application shutdown completed successfully")
	return nil
}

// cleanup performs application cleanup
func (a *Application) cleanup() {
	if a.Logger != nil {
		a.Logger.Info("🧹 Cleaning up application resources...")
		
		// Close storage connections
		if a.Storage != nil {
			if err := a.Storage.Close(); err != nil {
				a.Logger.Error("Failed to close storage", zap.Error(err))
			}
		}

		// Sync logger
		_ = a.Logger.Sync()
	}
} 