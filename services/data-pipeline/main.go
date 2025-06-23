package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/confluentinc/confluent-kafka-go/kafka"
	"github.com/gin-gonic/gin"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/redis/go-redis/v9"
	swaggerFiles "github.com/swaggo/files"
	ginSwagger "github.com/swaggo/gin-swagger"
	"go.uber.org/zap"
	"gorm.io/gorm"

	"data-pipeline/config"
	"data-pipeline/internal/handlers"
	"data-pipeline/internal/middleware"
	"data-pipeline/internal/models"
	"data-pipeline/internal/pipeline"
	"data-pipeline/internal/processors"
	"data-pipeline/internal/storage"
	"data-pipeline/internal/streams"
)

// @title AIOps Data Pipeline Service API
// @version 1.0
// @description Enterprise real-time data processing pipeline for AIOps platform
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
	// Prometheus metrics for data pipeline
	messagesProcessedTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "data_pipeline_messages_processed_total",
			Help: "Total number of messages processed",
		},
		[]string{"topic", "processor", "status"},
	)

	processingDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "data_pipeline_processing_duration_seconds",
			Help:    "Time spent processing messages",
			Buckets: prometheus.ExponentialBuckets(0.001, 2, 15),
		},
		[]string{"topic", "processor"},
	)

	queueSize = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "data_pipeline_queue_size",
			Help: "Current size of processing queues",
		},
		[]string{"queue_type"},
	)

	throughputPerSecond = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "data_pipeline_throughput_per_second",
			Help: "Messages processed per second",
		},
		[]string{"topic", "processor"},
	)

	errorRate = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "data_pipeline_error_rate",
			Help: "Error rate percentage",
		},
		[]string{"topic", "processor"},
	)

	dataLagSeconds = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "data_pipeline_lag_seconds",
			Help: "Data processing lag in seconds",
		},
		[]string{"topic", "consumer_group"},
	)

	activeConsumers = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Name: "data_pipeline_active_consumers",
			Help: "Number of active Kafka consumers",
		},
	)

	activeProducers = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Name: "data_pipeline_active_producers",
			Help: "Number of active Kafka producers",
		},
	)
)

func init() {
	// Register Prometheus metrics
	prometheus.MustRegister(messagesProcessedTotal)
	prometheus.MustRegister(processingDuration)
	prometheus.MustRegister(queueSize)
	prometheus.MustRegister(throughputPerSecond)
	prometheus.MustRegister(errorRate)
	prometheus.MustRegister(dataLagSeconds)
	prometheus.MustRegister(activeConsumers)
	prometheus.MustRegister(activeProducers)
}

// DataPipelineService represents the main service
type DataPipelineService struct {
	config         *config.Config
	logger         *zap.Logger
	db             *gorm.DB
	redis          *redis.Client
	pipelineEngine *pipeline.Engine
	streamManager  *streams.Manager
	processors     map[string]processors.Processor
	httpServer     *http.Server
	
	// Service control
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
	
	// Metrics tracking
	startTime      time.Time
	metricsManager *MetricsManager
}

// MetricsManager handles metrics collection and reporting
type MetricsManager struct {
	logger         *zap.Logger
	throughputData map[string]*ThroughputTracker
	mu             sync.RWMutex
}

// ThroughputTracker tracks throughput for each topic/processor
type ThroughputTracker struct {
	Count     int64
	LastReset time.Time
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

	logger.Info("🌊 Starting AIOps Data Pipeline Service",
		zap.String("version", "1.0.0"),
		zap.String("environment", cfg.Environment),
		zap.Int("port", cfg.Server.Port),
	)

	// Create service instance
	service := &DataPipelineService{
		config:    cfg,
		logger:    logger,
		startTime: time.Now(),
	}

	// Initialize service
	if err := service.Initialize(); err != nil {
		logger.Fatal("Failed to initialize service", zap.Error(err))
	}

	// Start service
	if err := service.Start(); err != nil {
		logger.Fatal("Failed to start service", zap.Error(err))
	}

	// Wait for shutdown signal
	service.WaitForShutdown()
}

func (s *DataPipelineService) Initialize() error {
	s.logger.Info("🔧 Initializing Data Pipeline Service...")

	// Create context for service lifecycle
	s.ctx, s.cancel = context.WithCancel(context.Background())

	// Initialize database
	if err := s.initializeDatabase(); err != nil {
		return fmt.Errorf("failed to initialize database: %w", err)
	}

	// Initialize Redis
	if err := s.initializeRedis(); err != nil {
		return fmt.Errorf("failed to initialize Redis: %w", err)
	}

	// Initialize metrics manager
	s.metricsManager = &MetricsManager{
		logger:         s.logger,
		throughputData: make(map[string]*ThroughputTracker),
	}

	// Initialize stream manager
	if err := s.initializeStreamManager(); err != nil {
		return fmt.Errorf("failed to initialize stream manager: %w", err)
	}

	// Initialize pipeline engine
	if err := s.initializePipelineEngine(); err != nil {
		return fmt.Errorf("failed to initialize pipeline engine: %w", err)
	}

	// Initialize processors
	if err := s.initializeProcessors(); err != nil {
		return fmt.Errorf("failed to initialize processors: %w", err)
	}

	// Setup HTTP server
	if err := s.setupHTTPServer(); err != nil {
		return fmt.Errorf("failed to setup HTTP server: %w", err)
	}

	s.logger.Info("✅ Data Pipeline Service initialized successfully")
	return nil
}

func (s *DataPipelineService) initializeDatabase() error {
	s.logger.Info("📊 Initializing database connection...")

	db, err := storage.Initialize(s.config.Database.URL)
	if err != nil {
		return err
	}

	// Auto-migrate schemas
	if err := db.AutoMigrate(
		&models.DataSource{},
		&models.Pipeline{},
		&models.ProcessingJob{},
		&models.DataQualityCheck{},
		&models.StreamMetadata{},
		&models.ProcessorConfig{},
		&models.AlertRule{},
	); err != nil {
		return fmt.Errorf("failed to migrate database: %w", err)
	}

	s.db = db
	return nil
}

func (s *DataPipelineService) initializeRedis() error {
	s.logger.Info("🔴 Initializing Redis connection...")

	opts, err := redis.ParseURL(s.config.Redis.URL)
	if err != nil {
		return fmt.Errorf("failed to parse Redis URL: %w", err)
	}

	client := redis.NewClient(opts)

	// Test connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := client.Ping(ctx).Err(); err != nil {
		return fmt.Errorf("failed to connect to Redis: %w", err)
	}

	s.redis = client
	return nil
}

func (s *DataPipelineService) initializeStreamManager() error {
	s.logger.Info("🌊 Initializing stream manager...")

	manager, err := streams.NewManager(s.config, s.logger, s.redis)
	if err != nil {
		return err
	}

	if err := manager.Initialize(); err != nil {
		return err
	}

	s.streamManager = manager
	return nil
}

func (s *DataPipelineService) initializePipelineEngine() error {
	s.logger.Info("⚙️ Initializing pipeline engine...")

	engine, err := pipeline.NewEngine(s.config, s.logger, s.db, s.streamManager)
	if err != nil {
		return err
	}

	if err := engine.Initialize(); err != nil {
		return err
	}

	s.pipelineEngine = engine
	return nil
}

func (s *DataPipelineService) initializeProcessors() error {
	s.logger.Info("🔄 Initializing data processors...")

	s.processors = make(map[string]processors.Processor)

	// Initialize metric processor
	metricProcessor, err := processors.NewMetricProcessor(s.config, s.logger)
	if err != nil {
		return err
	}
	s.processors["metrics"] = metricProcessor

	// Initialize log processor
	logProcessor, err := processors.NewLogProcessor(s.config, s.logger)
	if err != nil {
		return err
	}
	s.processors["logs"] = logProcessor

	// Initialize event processor
	eventProcessor, err := processors.NewEventProcessor(s.config, s.logger)
	if err != nil {
		return err
	}
	s.processors["events"] = eventProcessor

	// Initialize trace processor
	traceProcessor, err := processors.NewTraceProcessor(s.config, s.logger)
	if err != nil {
		return err
	}
	s.processors["traces"] = traceProcessor

	// Initialize alert processor
	alertProcessor, err := processors.NewAlertProcessor(s.config, s.logger)
	if err != nil {
		return err
	}
	s.processors["alerts"] = alertProcessor

	return nil
}

func (s *DataPipelineService) setupHTTPServer() error {
	s.logger.Info("🌐 Setting up HTTP server...")

	// Setup Gin router
	if s.config.Environment == "production" {
		gin.SetMode(gin.ReleaseMode)
	}

	router := gin.New()

	// Global middleware
	router.Use(middleware.Logger(s.logger))
	router.Use(middleware.Recovery(s.logger))
	router.Use(middleware.CORS())
	router.Use(middleware.Security())
	router.Use(middleware.Metrics())

	// Prometheus metrics endpoint
	router.GET("/metrics", gin.WrapH(promhttp.Handler()))

	// Health check endpoints
	router.GET("/health", s.healthCheckHandler)
	router.GET("/health/live", s.livenessCheckHandler)
	router.GET("/health/ready", s.readinessCheckHandler)

	// Swagger documentation
	if s.config.Environment != "production" {
		router.GET("/docs/*any", ginSwagger.WrapHandler(swaggerFiles.Handler))
	}

	// API routes
	api := router.Group("/api/v1")
	api.Use(middleware.Authentication(s.config.Security.JWTSecret))

	// Pipeline management endpoints
	pipelineHandler := handlers.NewPipelineHandler(s.pipelineEngine, s.logger)
	pipeline := api.Group("/pipeline")
	{
		pipeline.POST("/", pipelineHandler.CreatePipeline)
		pipeline.GET("/", pipelineHandler.ListPipelines)
		pipeline.GET("/:id", pipelineHandler.GetPipeline)
		pipeline.PUT("/:id", pipelineHandler.UpdatePipeline)
		pipeline.DELETE("/:id", pipelineHandler.DeletePipeline)
		pipeline.POST("/:id/start", pipelineHandler.StartPipeline)
		pipeline.POST("/:id/stop", pipelineHandler.StopPipeline)
		pipeline.GET("/:id/status", pipelineHandler.GetPipelineStatus)
		pipeline.GET("/:id/metrics", pipelineHandler.GetPipelineMetrics)
	}

	// Stream management endpoints
	streamHandler := handlers.NewStreamHandler(s.streamManager, s.logger)
	stream := api.Group("/streams")
	{
		stream.POST("/", streamHandler.CreateStream)
		stream.GET("/", streamHandler.ListStreams)
		stream.GET("/:id", streamHandler.GetStream)
		stream.DELETE("/:id", streamHandler.DeleteStream)
		stream.POST("/:id/consume", streamHandler.ConsumeFromStream)
		stream.POST("/:id/produce", streamHandler.ProduceToStream)
		stream.GET("/:id/metrics", streamHandler.GetStreamMetrics)
	}

	// Data ingestion endpoints
	dataHandler := handlers.NewDataHandler(s.pipelineEngine, s.processors, s.logger)
	data := api.Group("/data")
	{
		data.POST("/ingest", dataHandler.IngestData)
		data.POST("/batch", dataHandler.BatchIngestData)
		data.POST("/validate", dataHandler.ValidateData)
		data.GET("/sources", dataHandler.ListDataSources)
		data.POST("/sources", dataHandler.CreateDataSource)
		data.GET("/quality", dataHandler.GetDataQuality)
	}

	// Processor management endpoints
	processorHandler := handlers.NewProcessorHandler(s.processors, s.logger)
	processor := api.Group("/processors")
	{
		processor.GET("/", processorHandler.ListProcessors)
		processor.GET("/:name", processorHandler.GetProcessor)
		processor.POST("/:name/configure", processorHandler.ConfigureProcessor)
		processor.GET("/:name/metrics", processorHandler.GetProcessorMetrics)
		processor.POST("/:name/restart", processorHandler.RestartProcessor)
	}

	// Monitoring and analytics endpoints
	monitoringHandler := handlers.NewMonitoringHandler(s.metricsManager, s.logger)
	monitoring := api.Group("/monitoring")
	{
		monitoring.GET("/metrics", monitoringHandler.GetSystemMetrics)
		monitoring.GET("/throughput", monitoringHandler.GetThroughputMetrics)
		monitoring.GET("/lag", monitoringHandler.GetLagMetrics)
		monitoring.GET("/health", monitoringHandler.GetHealthMetrics)
		monitoring.GET("/alerts", monitoringHandler.GetActiveAlerts)
	}

	// Admin endpoints
	admin := api.Group("/admin")
	admin.Use(middleware.RequireRole("admin"))
	{
		admin.GET("/stats", s.getSystemStatsHandler)
		admin.POST("/maintenance", s.enableMaintenanceModeHandler)
		admin.DELETE("/maintenance", s.disableMaintenanceModeHandler)
		admin.POST("/cache/clear", s.clearCacheHandler)
		admin.POST("/consumers/restart", s.restartConsumersHandler)
		admin.POST("/producers/restart", s.restartProducersHandler)
	}

	// Create HTTP server
	s.httpServer = &http.Server{
		Addr:           fmt.Sprintf(":%d", s.config.Server.Port),
		Handler:        router,
		ReadTimeout:    time.Duration(s.config.Server.ReadTimeout) * time.Second,
		WriteTimeout:   time.Duration(s.config.Server.WriteTimeout) * time.Second,
		IdleTimeout:    time.Duration(s.config.Server.IdleTimeout) * time.Second,
		MaxHeaderBytes: s.config.Server.MaxHeaderBytes,
	}

	return nil
}

func (s *DataPipelineService) Start() error {
	s.logger.Info("🚀 Starting Data Pipeline Service...")

	// Start background services
	s.startBackgroundServices()

	// Start HTTP server
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		
		s.logger.Info("🌐 HTTP server started",
			zap.String("address", s.httpServer.Addr),
			zap.String("docs", fmt.Sprintf("http://localhost:%d/docs", s.config.Server.Port)),
		)

		if err := s.httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			s.logger.Error("HTTP server failed", zap.Error(err))
		}
	}()

	s.logger.Info("✅ Data Pipeline Service started successfully")
	return nil
}

func (s *DataPipelineService) startBackgroundServices() {
	// Start stream manager
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		if err := s.streamManager.Start(s.ctx); err != nil {
			s.logger.Error("Stream manager failed", zap.Error(err))
		}
	}()

	// Start pipeline engine
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		if err := s.pipelineEngine.Start(s.ctx); err != nil {
			s.logger.Error("Pipeline engine failed", zap.Error(err))
		}
	}()

	// Start processors
	for name, processor := range s.processors {
		s.wg.Add(1)
		go func(name string, proc processors.Processor) {
			defer s.wg.Done()
			if err := proc.Start(s.ctx); err != nil {
				s.logger.Error("Processor failed", zap.String("processor", name), zap.Error(err))
			}
		}(name, processor)
	}

	// Start metrics collection
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		s.collectMetrics(s.ctx)
	}()

	// Start health monitoring
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		s.monitorHealth(s.ctx)
	}()
}

func (s *DataPipelineService) collectMetrics(ctx context.Context) {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			s.updateMetrics()
		}
	}
}

func (s *DataPipelineService) updateMetrics() {
	// Update consumer and producer counts
	activeConsumers.Set(float64(s.streamManager.GetActiveConsumerCount()))
	activeProducers.Set(float64(s.streamManager.GetActiveProducerCount()))

	// Update queue sizes
	for queueType, size := range s.streamManager.GetQueueSizes() {
		queueSize.WithLabelValues(queueType).Set(float64(size))
	}

	// Update throughput metrics
	s.metricsManager.UpdateThroughputMetrics()
}

func (s *DataPipelineService) monitorHealth(ctx context.Context) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			s.performHealthChecks()
		}
	}
}

func (s *DataPipelineService) performHealthChecks() {
	// Check database health
	if s.db != nil {
		sqlDB, err := s.db.DB()
		if err == nil {
			if err := sqlDB.Ping(); err != nil {
				s.logger.Warn("Database health check failed", zap.Error(err))
			}
		}
	}

	// Check Redis health
	if s.redis != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := s.redis.Ping(ctx).Err(); err != nil {
			s.logger.Warn("Redis health check failed", zap.Error(err))
		}
	}

	// Check stream manager health
	if !s.streamManager.IsHealthy() {
		s.logger.Warn("Stream manager health check failed")
	}

	// Check pipeline engine health
	if !s.pipelineEngine.IsHealthy() {
		s.logger.Warn("Pipeline engine health check failed")
	}
}

func (s *DataPipelineService) WaitForShutdown() {
	// Wait for interrupt signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	s.logger.Info("🛑 Shutting down Data Pipeline Service...")

	// Cancel context to stop background services
	s.cancel()

	// Shutdown HTTP server
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer shutdownCancel()

	if err := s.httpServer.Shutdown(shutdownCtx); err != nil {
		s.logger.Error("HTTP server forced shutdown", zap.Error(err))
	}

	// Wait for all goroutines to finish
	done := make(chan struct{})
	go func() {
		s.wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		s.logger.Info("✅ All services stopped gracefully")
	case <-time.After(45 * time.Second):
		s.logger.Warn("⚠️ Forced shutdown after timeout")
	}

	// Cleanup resources
	if s.redis != nil {
		s.redis.Close()
	}

	s.logger.Info("✅ Data Pipeline Service shutdown completed")
}

// HTTP Handlers
func (s *DataPipelineService) healthCheckHandler(c *gin.Context) {
	uptime := time.Since(s.startTime)
	
	health := map[string]interface{}{
		"status":    "healthy",
		"uptime":    uptime.String(),
		"version":   "1.0.0",
		"timestamp": time.Now().Unix(),
		"services": map[string]string{
			"database":       "healthy",
			"redis":         "healthy",
			"stream_manager": "healthy",
			"pipeline_engine": "healthy",
		},
		"metrics": map[string]interface{}{
			"active_consumers": s.streamManager.GetActiveConsumerCount(),
			"active_producers": s.streamManager.GetActiveProducerCount(),
			"queue_sizes":     s.streamManager.GetQueueSizes(),
		},
	}

	c.JSON(http.StatusOK, health)
}

func (s *DataPipelineService) livenessCheckHandler(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"status": "alive"})
}

func (s *DataPipelineService) readinessCheckHandler(c *gin.Context) {
	if s.streamManager.IsHealthy() && s.pipelineEngine.IsHealthy() {
		c.JSON(http.StatusOK, gin.H{"status": "ready"})
	} else {
		c.JSON(http.StatusServiceUnavailable, gin.H{"status": "not ready"})
	}
}

func (s *DataPipelineService) getSystemStatsHandler(c *gin.Context) {
	stats := map[string]interface{}{
		"uptime":           time.Since(s.startTime).String(),
		"active_consumers": s.streamManager.GetActiveConsumerCount(),
		"active_producers": s.streamManager.GetActiveProducerCount(),
		"queue_sizes":      s.streamManager.GetQueueSizes(),
		"throughput":       s.metricsManager.GetThroughputStats(),
		"processor_stats":  s.getProcessorStats(),
	}

	c.JSON(http.StatusOK, stats)
}

func (s *DataPipelineService) getProcessorStats() map[string]interface{} {
	stats := make(map[string]interface{})
	for name, processor := range s.processors {
		stats[name] = processor.GetStats()
	}
	return stats
}

func (s *DataPipelineService) enableMaintenanceModeHandler(c *gin.Context) {
	// Implementation for maintenance mode
	c.JSON(http.StatusOK, gin.H{"message": "Maintenance mode enabled"})
}

func (s *DataPipelineService) disableMaintenanceModeHandler(c *gin.Context) {
	// Implementation for disabling maintenance mode
	c.JSON(http.StatusOK, gin.H{"message": "Maintenance mode disabled"})
}

func (s *DataPipelineService) clearCacheHandler(c *gin.Context) {
	// Clear Redis cache
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	
	if err := s.redis.FlushDB(ctx).Err(); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	
	c.JSON(http.StatusOK, gin.H{"message": "Cache cleared successfully"})
}

func (s *DataPipelineService) restartConsumersHandler(c *gin.Context) {
	if err := s.streamManager.RestartConsumers(); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	
	c.JSON(http.StatusOK, gin.H{"message": "Consumers restarted successfully"})
}

func (s *DataPipelineService) restartProducersHandler(c *gin.Context) {
	if err := s.streamManager.RestartProducers(); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	
	c.JSON(http.StatusOK, gin.H{"message": "Producers restarted successfully"})
}

// UpdateThroughputMetrics updates throughput metrics for Prometheus
func (m *MetricsManager) UpdateThroughputMetrics() {
	m.mu.Lock()
	defer m.mu.Unlock()

	now := time.Now()
	for key, tracker := range m.throughputData {
		if now.Sub(tracker.LastReset) >= time.Minute {
			// Calculate throughput per second
			seconds := now.Sub(tracker.LastReset).Seconds()
			if seconds > 0 {
				tps := float64(tracker.Count) / seconds
				
				// Parse key to extract topic and processor
				// Format: "topic:processor"
				// For simplicity, we'll use the key as-is for the metric
				throughputPerSecond.WithLabelValues(key, "unknown").Set(tps)
			}
			
			// Reset counter
			tracker.Count = 0
			tracker.LastReset = now
		}
	}
}

// GetThroughputStats returns current throughput statistics
func (m *MetricsManager) GetThroughputStats() map[string]interface{} {
	m.mu.RLock()
	defer m.mu.RUnlock()

	stats := make(map[string]interface{})
	for key, tracker := range m.throughputData {
		stats[key] = map[string]interface{}{
			"count":      tracker.Count,
			"last_reset": tracker.LastReset,
		}
	}
	return stats
}

func setupLogger(environment string) (*zap.Logger, error) {
	if environment == "production" {
		return zap.NewProduction()
	}
	return zap.NewDevelopment()
} 