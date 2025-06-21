package services

import (
	"context"
	"fmt"
	"sync"
	"time"

	"go.uber.org/zap"

	"monitoring-service/internal/config"
	"monitoring-service/internal/models"
	"monitoring-service/internal/storage"
	"monitoring-service/pkg/notifications"
)

// AlertService handles alert processing and notifications
type AlertService struct {
	config      *config.Config
	logger      *zap.Logger
	storage     storage.Storage
	notifier    *notifications.Manager
	rules       map[string]*models.AlertRule
	activeRules sync.Map
	mu          sync.RWMutex
	isRunning   bool
}

// NewAlertService creates a new alert service instance
func NewAlertService(cfg *config.Config, logger *zap.Logger, storage storage.Storage) *AlertService {
	service := &AlertService{
		config:  cfg,
		logger:  logger,
		storage: storage,
		rules:   make(map[string]*models.AlertRule),
	}

	// Initialize notification manager
	service.notifier = notifications.NewManager(cfg, logger)

	return service
}

// StartProcessing starts the alert processing engine
func (s *AlertService) StartProcessing(ctx context.Context) error {
	s.mu.Lock()
	if s.isRunning {
		s.mu.Unlock()
		return fmt.Errorf("alert processing is already running")
	}
	s.isRunning = true
	s.mu.Unlock()

	s.logger.Info("🚨 Starting alert processing service")

	// Load existing alert rules
	if err := s.loadAlertRules(); err != nil {
		s.logger.Error("Failed to load alert rules", zap.Error(err))
	}

	// Start alert processing ticker
	ticker := time.NewTicker(time.Duration(s.config.Alerts.CheckInterval) * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			s.logger.Info("🚨 Stopping alert processing service")
			s.mu.Lock()
			s.isRunning = false
			s.mu.Unlock()
			return ctx.Err()

		case <-ticker.C:
			if err := s.processAlerts(ctx); err != nil {
				s.logger.Error("Failed to process alerts", zap.Error(err))
			}
		}
	}
}

// loadAlertRules loads all alert rules from storage
func (s *AlertService) loadAlertRules() error {
	// Get all systems to load their rules
	systems, err := s.storage.ListSystems()
	if err != nil {
		return fmt.Errorf("failed to list systems: %w", err)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	// Clear existing rules
	s.rules = make(map[string]*models.AlertRule)

	// Load rules for each system
	for _, system := range systems {
		rules, err := s.storage.ListAlertRules(system.ID)
		if err != nil {
			s.logger.Error("Failed to load alert rules for system",
				zap.String("system_id", system.ID), zap.Error(err))
			continue
		}

		for _, rule := range rules {
			if rule.Enabled {
				s.rules[rule.ID] = rule
			}
		}
	}

	s.logger.Info("Loaded alert rules", zap.Int("count", len(s.rules)))
	return nil
}

// processAlerts checks all active rules against current metrics
func (s *AlertService) processAlerts(ctx context.Context) error {
	s.mu.RLock()
	rules := make([]*models.AlertRule, 0, len(s.rules))
	for _, rule := range s.rules {
		rules = append(rules, rule)
	}
	s.mu.RUnlock()

	if len(rules) == 0 {
		return nil
	}

	s.logger.Debug("Processing alert rules", zap.Int("rule_count", len(rules)))

	// Process rules in parallel
	var wg sync.WaitGroup
	semaphore := make(chan struct{}, 10) // Limit concurrent rule processing

	for _, rule := range rules {
		wg.Add(1)
		go func(rule *models.AlertRule) {
			defer wg.Done()
			semaphore <- struct{}{}
			defer func() { <-semaphore }()

			if err := s.evaluateRule(ctx, rule); err != nil {
				s.logger.Error("Failed to evaluate rule",
					zap.String("rule_id", rule.ID),
					zap.String("rule_name", rule.Name),
					zap.Error(err))
			}
		}(rule)
	}

	wg.Wait()
	s.logger.Debug("Alert rule processing completed")
	return nil
}

// evaluateRule evaluates a single alert rule
func (s *AlertService) evaluateRule(ctx context.Context, rule *models.AlertRule) error {
	// Get latest metrics for the system
	metrics, err := s.storage.GetLatestMetrics(rule.SystemID, 1)
	if err != nil {
		return fmt.Errorf("failed to get metrics for system %s: %w", rule.SystemID, err)
	}

	// Find the metric this rule is monitoring
	var targetMetric *models.MetricPoint
	for _, metric := range metrics {
		if metric.Name == rule.MetricName {
			targetMetric = metric
			break
		}
	}

	if targetMetric == nil {
		s.logger.Debug("Metric not found for rule",
			zap.String("rule_id", rule.ID),
			zap.String("metric_name", rule.MetricName))
		return nil
	}

	// Evaluate the condition
	triggered := s.evaluateCondition(targetMetric.Value, rule.Operator, rule.Threshold)

	if triggered {
		return s.handleTriggeredRule(ctx, rule, targetMetric)
	}

	// Check if we need to resolve any existing alerts
	return s.resolveAlertsIfNeeded(ctx, rule, targetMetric)
}

// evaluateCondition evaluates if a metric value triggers an alert condition
func (s *AlertService) evaluateCondition(value float64, operator string, threshold float64) bool {
	switch operator {
	case "gt", ">":
		return value > threshold
	case "gte", ">=":
		return value >= threshold
	case "lt", "<":
		return value < threshold
	case "lte", "<=":
		return value <= threshold
	case "eq", "==":
		return value == threshold
	case "ne", "!=":
		return value != threshold
	default:
		return false
	}
}

// handleTriggeredRule handles a rule that has been triggered
func (s *AlertService) handleTriggeredRule(ctx context.Context, rule *models.AlertRule, metric *models.MetricPoint) error {
	// Check if there's already an active alert for this rule
	activeAlerts, err := s.storage.GetActiveAlerts(rule.SystemID)
	if err != nil {
		return fmt.Errorf("failed to get active alerts: %w", err)
	}

	// Look for existing alert for this rule
	for _, alert := range activeAlerts {
		if alert.MetricName == rule.MetricName && alert.Status == "active" {
			// Alert already exists and is active
			s.logger.Debug("Alert already active for rule",
				zap.String("rule_id", rule.ID),
				zap.String("alert_id", alert.ID))
			return nil
		}
	}

	// Create new alert
	alert := &models.Alert{
		ID:          generateAlertID(),
		SystemID:    rule.SystemID,
		Name:        fmt.Sprintf("Alert: %s", rule.Name),
		Description: rule.Description,
		Severity:    rule.Severity,
		Status:      "active",
		MetricName:  metric.Name,
		MetricValue: metric.Value,
		Threshold:   rule.Threshold,
		Operator:    rule.Operator,
		Labels:      combineLabels(rule.Labels, metric.Labels),
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}

	// Store the alert
	if err := s.storage.StoreAlert(alert); err != nil {
		return fmt.Errorf("failed to store alert: %w", err)
	}

	s.logger.Info("Alert triggered",
		zap.String("alert_id", alert.ID),
		zap.String("system_id", alert.SystemID),
		zap.String("metric_name", alert.MetricName),
		zap.Float64("metric_value", alert.MetricValue),
		zap.Float64("threshold", alert.Threshold),
		zap.String("severity", alert.Severity))

	// Send notifications
	if err := s.sendNotifications(alert, rule); err != nil {
		s.logger.Error("Failed to send notifications for alert",
			zap.String("alert_id", alert.ID), zap.Error(err))
	}

	return nil
}

// resolveAlertsIfNeeded resolves alerts if conditions are no longer met
func (s *AlertService) resolveAlertsIfNeeded(ctx context.Context, rule *models.AlertRule, metric *models.MetricPoint) error {
	// Get active alerts for this rule
	activeAlerts, err := s.storage.GetActiveAlerts(rule.SystemID)
	if err != nil {
		return fmt.Errorf("failed to get active alerts: %w", err)
	}

	for _, alert := range activeAlerts {
		if alert.MetricName == rule.MetricName && alert.Status == "active" {
			// Check if condition is still met
			if !s.evaluateCondition(metric.Value, rule.Operator, rule.Threshold) {
				// Resolve the alert
				now := time.Now()
				alert.Status = "resolved"
				alert.UpdatedAt = now
				alert.ResolvedAt = &now

				if err := s.storage.UpdateAlert(alert); err != nil {
					return fmt.Errorf("failed to update resolved alert: %w", err)
				}

				s.logger.Info("Alert resolved",
					zap.String("alert_id", alert.ID),
					zap.String("system_id", alert.SystemID),
					zap.String("metric_name", alert.MetricName))

				// Send resolution notification
				if err := s.sendResolutionNotification(alert, rule); err != nil {
					s.logger.Error("Failed to send resolution notification",
						zap.String("alert_id", alert.ID), zap.Error(err))
				}
			}
		}
	}

	return nil
}

// sendNotifications sends alert notifications through configured channels
func (s *AlertService) sendNotifications(alert *models.Alert, rule *models.AlertRule) error {
	if !s.config.Alerts.Enabled {
		return nil
	}

	notification := &notifications.AlertNotification{
		Alert:    alert,
		Rule:     rule,
		Type:     "triggered",
		Channels: rule.Channels,
	}

	return s.notifier.SendAlert(notification)
}

// sendResolutionNotification sends alert resolution notification
func (s *AlertService) sendResolutionNotification(alert *models.Alert, rule *models.AlertRule) error {
	if !s.config.Alerts.Enabled {
		return nil
	}

	notification := &notifications.AlertNotification{
		Alert:    alert,
		Rule:     rule,
		Type:     "resolved",
		Channels: rule.Channels,
	}

	return s.notifier.SendAlert(notification)
}

// CreateAlert creates a new alert manually
func (s *AlertService) CreateAlert(alert *models.Alert) error {
	alert.ID = generateAlertID()
	alert.CreatedAt = time.Now()
	alert.UpdatedAt = time.Now()

	if err := s.storage.StoreAlert(alert); err != nil {
		return fmt.Errorf("failed to create alert: %w", err)
	}

	s.logger.Info("Manual alert created",
		zap.String("alert_id", alert.ID),
		zap.String("system_id", alert.SystemID))

	return nil
}

// GetAlert retrieves an alert by ID
func (s *AlertService) GetAlert(alertID string) (*models.Alert, error) {
	return s.storage.GetAlert(alertID)
}

// UpdateAlert updates an existing alert
func (s *AlertService) UpdateAlert(alertID string, updates *models.AlertUpdate) error {
	alert, err := s.storage.GetAlert(alertID)
	if err != nil {
		return err
	}

	// Apply updates
	if updates.Status != "" {
		alert.Status = updates.Status
		if updates.Status == "resolved" {
			now := time.Now()
			alert.ResolvedAt = &now
		}
	}
	if updates.Description != "" {
		alert.Description = updates.Description
	}

	alert.UpdatedAt = time.Now()

	if err := s.storage.UpdateAlert(alert); err != nil {
		return fmt.Errorf("failed to update alert: %w", err)
	}

	s.logger.Info("Alert updated",
		zap.String("alert_id", alertID),
		zap.String("status", alert.Status))

	return nil
}

// DeleteAlert deletes an alert
func (s *AlertService) DeleteAlert(alertID string) error {
	if err := s.storage.DeleteAlert(alertID); err != nil {
		return fmt.Errorf("failed to delete alert: %w", err)
	}

	s.logger.Info("Alert deleted", zap.String("alert_id", alertID))
	return nil
}

// ListAlerts retrieves alerts for a system
func (s *AlertService) ListAlerts(systemID string, limit int) ([]*models.Alert, error) {
	return s.storage.ListAlerts(systemID, limit)
}

// AcknowledgeAlert acknowledges an alert
func (s *AlertService) AcknowledgeAlert(alertID string) error {
	alert, err := s.storage.GetAlert(alertID)
	if err != nil {
		return err
	}

	if alert.Status != "active" {
		return fmt.Errorf("alert is not in active state")
	}

	alert.Status = "acknowledged"
	alert.UpdatedAt = time.Now()

	if err := s.storage.UpdateAlert(alert); err != nil {
		return fmt.Errorf("failed to acknowledge alert: %w", err)
	}

	s.logger.Info("Alert acknowledged",
		zap.String("alert_id", alertID),
		zap.String("system_id", alert.SystemID))

	return nil
}

// CreateAlertRule creates a new alert rule
func (s *AlertService) CreateAlertRule(rule *models.AlertRule) error {
	rule.ID = generateRuleID()
	rule.CreatedAt = time.Now()
	rule.UpdatedAt = time.Now()

	if err := s.storage.StoreAlertRule(rule); err != nil {
		return fmt.Errorf("failed to create alert rule: %w", err)
	}

	// Add to active rules if enabled
	if rule.Enabled {
		s.mu.Lock()
		s.rules[rule.ID] = rule
		s.mu.Unlock()
	}

	s.logger.Info("Alert rule created",
		zap.String("rule_id", rule.ID),
		zap.String("system_id", rule.SystemID),
		zap.String("metric_name", rule.MetricName))

	return nil
}

// UpdateAlertRule updates an existing alert rule
func (s *AlertService) UpdateAlertRule(ruleID string, updates *models.AlertRule) error {
	rule, err := s.storage.GetAlertRule(ruleID)
	if err != nil {
		return err
	}

	// Apply updates
	updates.ID = ruleID
	updates.UpdatedAt = time.Now()
	if updates.CreatedAt.IsZero() {
		updates.CreatedAt = rule.CreatedAt
	}

	if err := s.storage.UpdateAlertRule(updates); err != nil {
		return fmt.Errorf("failed to update alert rule: %w", err)
	}

	// Update in-memory rules
	s.mu.Lock()
	if updates.Enabled {
		s.rules[ruleID] = updates
	} else {
		delete(s.rules, ruleID)
	}
	s.mu.Unlock()

	s.logger.Info("Alert rule updated",
		zap.String("rule_id", ruleID))

	return nil
}

// DeleteAlertRule deletes an alert rule
func (s *AlertService) DeleteAlertRule(ruleID string) error {
	if err := s.storage.DeleteAlertRule(ruleID); err != nil {
		return fmt.Errorf("failed to delete alert rule: %w", err)
	}

	// Remove from active rules
	s.mu.Lock()
	delete(s.rules, ruleID)
	s.mu.Unlock()

	s.logger.Info("Alert rule deleted", zap.String("rule_id", ruleID))
	return nil
}

// ListAlertRules retrieves alert rules for a system
func (s *AlertService) ListAlertRules(systemID string) ([]*models.AlertRule, error) {
	return s.storage.ListAlertRules(systemID)
}

// Helper functions

// generateAlertID generates a unique alert ID
func generateAlertID() string {
	return fmt.Sprintf("alert_%d", time.Now().UnixNano())
}

// generateRuleID generates a unique rule ID
func generateRuleID() string {
	return fmt.Sprintf("rule_%d", time.Now().UnixNano())
}

// combineLabels combines rule labels with metric labels
func combineLabels(ruleLabels, metricLabels map[string]string) map[string]string {
	combined := make(map[string]string)
	
	// Add metric labels first
	for k, v := range metricLabels {
		combined[k] = v
	}
	
	// Add rule labels (override metric labels if conflicts)
	for k, v := range ruleLabels {
		combined[k] = v
	}
	
	return combined
} 