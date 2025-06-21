package services

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"

	"go.uber.org/zap"
	"github.com/prometheus/client_golang/api"
	v1 "github.com/prometheus/client_golang/api/prometheus/v1"
	"github.com/prometheus/common/model"

	"monitoring-service/internal/config"
	"monitoring-service/internal/models"
	"monitoring-service/internal/storage"
	"monitoring-service/pkg/metrics"
	"monitoring-service/pkg/collectors"
)

// MetricsService handles metrics collection and processing
type MetricsService struct {
	config      *config.Config
	logger      *zap.Logger
	storage     storage.Storage
	registry    *metrics.Registry
	collectors  map[string]collectors.Collector
	prometheus  v1.API
	mu          sync.RWMutex
	systems     map[string]*models.System
	isRunning   bool
}

// NewMetricsService creates a new metrics service instance
func NewMetricsService(cfg *config.Config, logger *zap.Logger, storage storage.Storage, registry *metrics.Registry) *MetricsService {
	service := &MetricsService{
		config:     cfg,
		logger:     logger,
		storage:    storage,
		registry:   registry,
		collectors: make(map[string]collectors.Collector),
		systems:    make(map[string]*models.System),
	}

	// Initialize Prometheus client if enabled
	if cfg.Metrics.Prometheus.URL != "" {
		if client, err := api.NewClient(api.Config{
			Address: cfg.Metrics.Prometheus.URL,
		}); err == nil {
			service.prometheus = v1.NewAPI(client)
		} else {
			logger.Error("Failed to initialize Prometheus client", zap.Error(err))
		}
	}

	// Initialize collectors
	service.initializeCollectors()

	return service
}

// StartCollection starts the metrics collection process
func (s *MetricsService) StartCollection(ctx context.Context) error {
	s.mu.Lock()
	if s.isRunning {
		s.mu.Unlock()
		return fmt.Errorf("metrics collection is already running")
	}
	s.isRunning = true
	s.mu.Unlock()

	s.logger.Info("📊 Starting metrics collection service")

	// Start collection ticker
	ticker := time.NewTicker(time.Duration(s.config.Metrics.CollectionInterval) * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			s.logger.Info("📊 Stopping metrics collection service")
			s.mu.Lock()
			s.isRunning = false
			s.mu.Unlock()
			return ctx.Err()

		case <-ticker.C:
			if err := s.collectMetrics(ctx); err != nil {
				s.logger.Error("Failed to collect metrics", zap.Error(err))
			}
		}
	}
}

// collectMetrics performs the actual metrics collection
func (s *MetricsService) collectMetrics(ctx context.Context) error {
	s.logger.Debug("Collecting metrics from all sources")

	var wg sync.WaitGroup
	errChan := make(chan error, len(s.collectors))

	// Collect from all configured sources
	for source, collector := range s.collectors {
		wg.Add(1)
		go func(source string, collector collectors.Collector) {
			defer wg.Done()
			
			if err := collector.Collect(ctx); err != nil {
				s.logger.Error("Failed to collect from source", 
					zap.String("source", source), 
					zap.Error(err))
				errChan <- err
			}
		}(source, collector)
	}

	wg.Wait()
	close(errChan)

	// Check for collection errors
	var collectionErrors []error
	for err := range errChan {
		collectionErrors = append(collectionErrors, err)
	}

	if len(collectionErrors) > 0 {
		s.logger.Warn("Some metric collection sources failed", 
			zap.Int("failed_count", len(collectionErrors)))
	}

	s.logger.Debug("Metrics collection completed successfully")
	return nil
}

// GetSystemMetrics retrieves current metrics for a specific system
func (s *MetricsService) GetSystemMetrics(systemID string) (*models.SystemMetrics, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	// Check if system exists
	system, exists := s.systems[systemID]
	if !exists {
		return nil, fmt.Errorf("system %s not found", systemID)
	}

	// Get metrics from storage
	metrics, err := s.storage.GetSystemMetrics(systemID, time.Now().Add(-1*time.Hour), time.Now())
	if err != nil {
		return nil, fmt.Errorf("failed to get system metrics: %w", err)
	}

	return &models.SystemMetrics{
		SystemID:    systemID,
		SystemName:  system.Name,
		Timestamp:   time.Now(),
		Metrics:     metrics,
		Status:      s.calculateSystemStatus(metrics),
		HealthScore: s.calculateHealthScore(metrics),
	}, nil
}

// GetHistoricalMetrics retrieves historical metrics for a system
func (s *MetricsService) GetHistoricalMetrics(systemID string, startTime, endTime time.Time) ([]*models.MetricPoint, error) {
	return s.storage.GetSystemMetrics(systemID, startTime, endTime)
}

// GetAggregatedMetrics retrieves aggregated metrics across all systems
func (s *MetricsService) GetAggregatedMetrics(timeRange time.Duration) (*models.AggregatedMetrics, error) {
	endTime := time.Now()
	startTime := endTime.Add(-timeRange)

	// Get metrics for all systems
	allMetrics := make(map[string][]*models.MetricPoint)
	
	s.mu.RLock()
	systems := make([]string, 0, len(s.systems))
	for systemID := range s.systems {
		systems = append(systems, systemID)
	}
	s.mu.RUnlock()

	for _, systemID := range systems {
		metrics, err := s.storage.GetSystemMetrics(systemID, startTime, endTime)
		if err != nil {
			s.logger.Error("Failed to get metrics for system", 
				zap.String("system_id", systemID), 
				zap.Error(err))
			continue
		}
		allMetrics[systemID] = metrics
	}

	// Calculate aggregated statistics
	return s.calculateAggregatedMetrics(allMetrics, startTime, endTime), nil
}

// QueryMetrics executes a custom metrics query
func (s *MetricsService) QueryMetrics(query *models.MetricsQuery) (*models.QueryResult, error) {
	s.logger.Info("Executing metrics query", 
		zap.String("query_type", query.Type),
		zap.String("system_id", query.SystemID))

	switch query.Type {
	case "prometheus":
		return s.executePrometheusQuery(query)
	case "aggregation":
		return s.executeAggregationQuery(query)
	case "custom":
		return s.executeCustomQuery(query)
	default:
		return nil, fmt.Errorf("unsupported query type: %s", query.Type)
	}
}

// RegisterSystem registers a new system for monitoring
func (s *MetricsService) RegisterSystem(system *models.System) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.systems[system.ID] = system
	
	// Store in persistent storage
	if err := s.storage.StoreSystem(system); err != nil {
		return fmt.Errorf("failed to store system: %w", err)
	}

	s.logger.Info("System registered for monitoring", 
		zap.String("system_id", system.ID),
		zap.String("system_name", system.Name))

	return nil
}

// UpdateSystem updates system information
func (s *MetricsService) UpdateSystem(systemID string, updates *models.SystemUpdate) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	system, exists := s.systems[systemID]
	if !exists {
		return fmt.Errorf("system %s not found", systemID)
	}

	// Apply updates
	if updates.Name != "" {
		system.Name = updates.Name
	}
	if updates.Description != "" {
		system.Description = updates.Description
	}
	if updates.Tags != nil {
		system.Tags = updates.Tags
	}
	if updates.Thresholds != nil {
		system.Thresholds = updates.Thresholds
	}

	system.UpdatedAt = time.Now()

	// Update in storage
	if err := s.storage.UpdateSystem(system); err != nil {
		return fmt.Errorf("failed to update system: %w", err)
	}

	s.logger.Info("System updated", 
		zap.String("system_id", systemID))

	return nil
}

// RemoveSystem removes a system from monitoring
func (s *MetricsService) RemoveSystem(systemID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, exists := s.systems[systemID]; !exists {
		return fmt.Errorf("system %s not found", systemID)
	}

	delete(s.systems, systemID)

	// Remove from storage
	if err := s.storage.DeleteSystem(systemID); err != nil {
		return fmt.Errorf("failed to delete system: %w", err)
	}

	s.logger.Info("System removed from monitoring", 
		zap.String("system_id", systemID))

	return nil
}

// ListSystems returns all registered systems
func (s *MetricsService) ListSystems() ([]*models.System, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	systems := make([]*models.System, 0, len(s.systems))
	for _, system := range s.systems {
		systems = append(systems, system)
	}

	return systems, nil
}

// GetSystemDetails retrieves detailed information about a system
func (s *MetricsService) GetSystemDetails(systemID string) (*models.SystemDetails, error) {
	s.mu.RLock()
	system, exists := s.systems[systemID]
	s.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("system %s not found", systemID)
	}

	// Get recent metrics
	recentMetrics, err := s.GetSystemMetrics(systemID)
	if err != nil {
		s.logger.Error("Failed to get recent metrics for system details", 
			zap.String("system_id", systemID), 
			zap.Error(err))
	}

	// Get alerts for this system
	alerts, err := s.storage.GetActiveAlerts(systemID)
	if err != nil {
		s.logger.Error("Failed to get alerts for system details", 
			zap.String("system_id", systemID), 
			zap.Error(err))
	}

	return &models.SystemDetails{
		System:        system,
		RecentMetrics: recentMetrics,
		ActiveAlerts:  alerts,
		Status:        s.determineSystemStatus(system, recentMetrics, alerts),
	}, nil
}

// StoreMetrics stores collected metrics
func (s *MetricsService) StoreMetrics(systemID string, metrics []*models.MetricPoint) error {
	if err := s.storage.StoreMetrics(systemID, metrics); err != nil {
		return fmt.Errorf("failed to store metrics: %w", err)
	}

	// Update registry metrics
	s.registry.IncrementMetricsCollected(systemID, len(metrics))

	return nil
}

// initializeCollectors sets up metric collectors for different sources
func (s *MetricsService) initializeCollectors() {
	for _, source := range s.config.Metrics.Sources {
		switch source {
		case "prometheus":
			if s.prometheus != nil {
				collector := collectors.NewPrometheusCollector(s.config, s.logger, s.prometheus, s)
				s.collectors["prometheus"] = collector
			}
		case "system":
			if s.config.Metrics.System.Enabled {
				collector := collectors.NewSystemCollector(s.config, s.logger, s)
				s.collectors["system"] = collector
			}
		case "docker":
			collector := collectors.NewDockerCollector(s.config, s.logger, s)
			s.collectors["docker"] = collector
		case "kubernetes":
			collector := collectors.NewKubernetesCollector(s.config, s.logger, s)
			s.collectors["kubernetes"] = collector
		}
	}

	s.logger.Info("Initialized metric collectors", 
		zap.Strings("sources", s.config.Metrics.Sources),
		zap.Int("collectors_count", len(s.collectors)))
}

// executePrometheusQuery executes a Prometheus query
func (s *MetricsService) executePrometheusQuery(query *models.MetricsQuery) (*models.QueryResult, error) {
	if s.prometheus == nil {
		return nil, fmt.Errorf("prometheus client not configured")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 
		time.Duration(s.config.Metrics.Prometheus.Timeout)*time.Second)
	defer cancel()

	result, warnings, err := s.prometheus.Query(ctx, query.Query, query.Timestamp)
	if err != nil {
		return nil, fmt.Errorf("prometheus query failed: %w", err)
	}

	if len(warnings) > 0 {
		s.logger.Warn("Prometheus query warnings", zap.Strings("warnings", warnings))
	}

	return &models.QueryResult{
		Type:      "prometheus",
		Data:      s.convertPrometheusResult(result),
		Timestamp: query.Timestamp,
		Duration:  time.Since(query.Timestamp),
	}, nil
}

// executeAggregationQuery executes an aggregation query
func (s *MetricsService) executeAggregationQuery(query *models.MetricsQuery) (*models.QueryResult, error) {
	// Implementation for aggregation queries
	// This would involve complex aggregation logic based on the query parameters
	return &models.QueryResult{
		Type:      "aggregation",
		Data:      make(map[string]interface{}),
		Timestamp: query.Timestamp,
		Duration:  time.Since(query.Timestamp),
	}, nil
}

// executeCustomQuery executes a custom query
func (s *MetricsService) executeCustomQuery(query *models.MetricsQuery) (*models.QueryResult, error) {
	// Implementation for custom queries
	// This would involve parsing and executing custom query logic
	return &models.QueryResult{
		Type:      "custom",
		Data:      make(map[string]interface{}),
		Timestamp: query.Timestamp,
		Duration:  time.Since(query.Timestamp),
	}, nil
}

// convertPrometheusResult converts Prometheus result to generic format
func (s *MetricsService) convertPrometheusResult(result model.Value) interface{} {
	switch v := result.(type) {
	case model.Vector:
		points := make([]map[string]interface{}, len(v))
		for i, sample := range v {
			points[i] = map[string]interface{}{
				"labels": sample.Metric,
				"value":  float64(sample.Value),
				"time":   sample.Timestamp.Time(),
			}
		}
		return points
	case model.Matrix:
		series := make([]map[string]interface{}, len(v))
		for i, sampleStream := range v {
			values := make([][]interface{}, len(sampleStream.Values))
			for j, pair := range sampleStream.Values {
				values[j] = []interface{}{pair.Timestamp.Time(), float64(pair.Value)}
			}
			series[i] = map[string]interface{}{
				"labels": sampleStream.Metric,
				"values": values,
			}
		}
		return series
	default:
		return result.String()
	}
}

// calculateSystemStatus determines the overall status of a system
func (s *MetricsService) calculateSystemStatus(metrics []*models.MetricPoint) string {
	if len(metrics) == 0 {
		return "unknown"
	}

	// Simple logic to determine status based on latest metrics
	// In a real implementation, this would be more sophisticated
	for _, metric := range metrics {
		if metric.Name == "cpu_usage" && metric.Value > 90 {
			return "critical"
		}
		if metric.Name == "memory_usage" && metric.Value > 85 {
			return "warning"
		}
		if metric.Name == "error_rate" && metric.Value > 5 {
			return "critical"
		}
	}

	return "healthy"
}

// calculateHealthScore calculates a health score for a system
func (s *MetricsService) calculateHealthScore(metrics []*models.MetricPoint) float64 {
	if len(metrics) == 0 {
		return 0
	}

	// Simple scoring algorithm
	score := 100.0
	
	for _, metric := range metrics {
		switch metric.Name {
		case "cpu_usage":
			if metric.Value > 80 {
				score -= (metric.Value - 80) * 2
			}
		case "memory_usage":
			if metric.Value > 85 {
				score -= (metric.Value - 85) * 3
			}
		case "error_rate":
			score -= metric.Value * 10
		}
	}

	if score < 0 {
		score = 0
	}

	return score
}

// calculateAggregatedMetrics calculates aggregated statistics
func (s *MetricsService) calculateAggregatedMetrics(allMetrics map[string][]*models.MetricPoint, startTime, endTime time.Time) *models.AggregatedMetrics {
	aggregated := &models.AggregatedMetrics{
		StartTime:    startTime,
		EndTime:      endTime,
		SystemCount:  len(allMetrics),
		MetricCounts: make(map[string]int),
		Averages:     make(map[string]float64),
		Maximums:     make(map[string]float64),
		Minimums:     make(map[string]float64),
	}

	metricSums := make(map[string]float64)
	metricCounts := make(map[string]int)
	metricMins := make(map[string]float64)
	metricMaxs := make(map[string]float64)

	// Process all metrics
	for _, systemMetrics := range allMetrics {
		for _, metric := range systemMetrics {
			// Count
			metricCounts[metric.Name]++
			
			// Sum for average
			metricSums[metric.Name] += metric.Value
			
			// Min/Max
			if count := metricCounts[metric.Name]; count == 1 {
				metricMins[metric.Name] = metric.Value
				metricMaxs[metric.Name] = metric.Value
			} else {
				if metric.Value < metricMins[metric.Name] {
					metricMins[metric.Name] = metric.Value
				}
				if metric.Value > metricMaxs[metric.Name] {
					metricMaxs[metric.Name] = metric.Value
				}
			}
		}
	}

	// Calculate averages and set final values
	for metricName, count := range metricCounts {
		aggregated.MetricCounts[metricName] = count
		aggregated.Averages[metricName] = metricSums[metricName] / float64(count)
		aggregated.Minimums[metricName] = metricMins[metricName]
		aggregated.Maximums[metricName] = metricMaxs[metricName]
	}

	return aggregated
}

// determineSystemStatus determines the overall status of a system
func (s *MetricsService) determineSystemStatus(system *models.System, metrics *models.SystemMetrics, alerts []*models.Alert) string {
	// Priority: alerts > metrics > default
	
	if len(alerts) > 0 {
		for _, alert := range alerts {
			if alert.Severity == "critical" {
				return "critical"
			}
		}
		return "warning"
	}

	if metrics != nil {
		return metrics.Status
	}

	return "unknown"
} 