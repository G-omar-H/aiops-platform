package storage

import (
	"context"
	"time"

	"monitoring-service/internal/models"
)

// Storage defines the interface for data persistence
type Storage interface {
	// System management
	StoreSystem(system *models.System) error
	GetSystem(systemID string) (*models.System, error)
	UpdateSystem(system *models.System) error
	DeleteSystem(systemID string) error
	ListSystems() ([]*models.System, error)

	// Metrics storage and retrieval
	StoreMetrics(systemID string, metrics []*models.MetricPoint) error
	GetSystemMetrics(systemID string, startTime, endTime time.Time) ([]*models.MetricPoint, error)
	GetLatestMetrics(systemID string, limit int) ([]*models.MetricPoint, error)

	// Alert management
	StoreAlert(alert *models.Alert) error
	GetAlert(alertID string) (*models.Alert, error)
	UpdateAlert(alert *models.Alert) error
	DeleteAlert(alertID string) error
	GetActiveAlerts(systemID string) ([]*models.Alert, error)
	ListAlerts(systemID string, limit int) ([]*models.Alert, error)

	// Alert rules
	StoreAlertRule(rule *models.AlertRule) error
	GetAlertRule(ruleID string) (*models.AlertRule, error)
	UpdateAlertRule(rule *models.AlertRule) error
	DeleteAlertRule(ruleID string) error
	ListAlertRules(systemID string) ([]*models.AlertRule, error)

	// Dashboard management
	StoreDashboard(dashboard *models.Dashboard) error
	GetDashboard(dashboardID string) (*models.Dashboard, error)
	UpdateDashboard(dashboard *models.Dashboard) error
	DeleteDashboard(dashboardID string) error
	ListDashboards(userID string) ([]*models.Dashboard, error)

	// Health and status
	HealthCheck() error
	Close() error
} 