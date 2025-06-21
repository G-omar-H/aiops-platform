package storage

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"go.uber.org/zap"

	"monitoring-service/internal/config"
	"monitoring-service/internal/models"
)

// PostgresStorage implements the Storage interface using PostgreSQL
type PostgresStorage struct {
	db     *pgxpool.Pool
	logger *zap.Logger
}

// NewPostgresStorage creates a new PostgreSQL storage instance
func NewPostgresStorage(cfg *config.Config, logger *zap.Logger) (*PostgresStorage, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	dbConfig, err := pgxpool.ParseConfig(cfg.Storage.PostgreSQL.GetDSN())
	if err != nil {
		return nil, fmt.Errorf("failed to parse database config: %w", err)
	}

	dbConfig.MaxConns = int32(cfg.Storage.PostgreSQL.MaxConns)
	dbConfig.MinConns = int32(cfg.Storage.PostgreSQL.MinConns)
	dbConfig.MaxConnLifetime = time.Duration(cfg.Storage.PostgreSQL.MaxConnTime) * time.Second

	db, err := pgxpool.NewWithConfig(ctx, dbConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create connection pool: %w", err)
	}

	// Test connection
	if err := db.Ping(ctx); err != nil {
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}

	storage := &PostgresStorage{
		db:     db,
		logger: logger,
	}

	// Initialize database schema
	if err := storage.initSchema(ctx); err != nil {
		return nil, fmt.Errorf("failed to initialize database schema: %w", err)
	}

	logger.Info("PostgreSQL storage initialized successfully")
	return storage, nil
}

// initSchema creates the necessary database tables
func (s *PostgresStorage) initSchema(ctx context.Context) error {
	queries := []string{
		`CREATE EXTENSION IF NOT EXISTS "uuid-ossp"`,
		
		// Systems table
		`CREATE TABLE IF NOT EXISTS systems (
			id VARCHAR(255) PRIMARY KEY,
			name VARCHAR(255) NOT NULL,
			description TEXT,
			type VARCHAR(100) NOT NULL,
			tags JSONB,
			endpoints JSONB,
			thresholds JSONB,
			status VARCHAR(50) DEFAULT 'unknown',
			created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
			updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
		)`,

		// Metrics table
		`CREATE TABLE IF NOT EXISTS metrics (
			id BIGSERIAL PRIMARY KEY,
			system_id VARCHAR(255) REFERENCES systems(id) ON DELETE CASCADE,
			name VARCHAR(255) NOT NULL,
			value DOUBLE PRECISION NOT NULL,
			unit VARCHAR(50),
			labels JSONB,
			timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
			UNIQUE(system_id, name, timestamp, labels)
		)`,

		// Alerts table
		`CREATE TABLE IF NOT EXISTS alerts (
			id VARCHAR(255) PRIMARY KEY DEFAULT gen_random_uuid()::text,
			system_id VARCHAR(255) REFERENCES systems(id) ON DELETE CASCADE,
			name VARCHAR(255) NOT NULL,
			description TEXT,
			severity VARCHAR(50) NOT NULL,
			status VARCHAR(50) DEFAULT 'active',
			metric_name VARCHAR(255),
			metric_value DOUBLE PRECISION,
			threshold DOUBLE PRECISION,
			operator VARCHAR(10),
			labels JSONB,
			created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
			updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
			resolved_at TIMESTAMP WITH TIME ZONE
		)`,

		// Alert rules table
		`CREATE TABLE IF NOT EXISTS alert_rules (
			id VARCHAR(255) PRIMARY KEY DEFAULT gen_random_uuid()::text,
			system_id VARCHAR(255) REFERENCES systems(id) ON DELETE CASCADE,
			name VARCHAR(255) NOT NULL,
			description TEXT,
			enabled BOOLEAN DEFAULT true,
			metric_name VARCHAR(255) NOT NULL,
			operator VARCHAR(10) NOT NULL,
			threshold DOUBLE PRECISION NOT NULL,
			severity VARCHAR(50) NOT NULL,
			duration INTERVAL,
			labels JSONB,
			channels JSONB,
			created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
			updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
		)`,

		// Dashboards table
		`CREATE TABLE IF NOT EXISTS dashboards (
			id VARCHAR(255) PRIMARY KEY DEFAULT gen_random_uuid()::text,
			name VARCHAR(255) NOT NULL,
			description TEXT,
			layout JSONB NOT NULL,
			filters JSONB,
			shared BOOLEAN DEFAULT false,
			created_by VARCHAR(255),
			created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
			updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
		)`,

		// Indexes for performance
		`CREATE INDEX IF NOT EXISTS idx_metrics_system_timestamp ON metrics(system_id, timestamp DESC)`,
		`CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(name)`,
		`CREATE INDEX IF NOT EXISTS idx_alerts_system_status ON alerts(system_id, status)`,
		`CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts(created_at DESC)`,
		`CREATE INDEX IF NOT EXISTS idx_alert_rules_system ON alert_rules(system_id)`,
	}

	for _, query := range queries {
		if _, err := s.db.Exec(ctx, query); err != nil {
			return fmt.Errorf("failed to execute query: %s, error: %w", query, err)
		}
	}

	return nil
}

// StoreSystem stores a new system
func (s *PostgresStorage) StoreSystem(system *models.System) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	tagsJSON, _ := json.Marshal(system.Tags)
	endpointsJSON, _ := json.Marshal(system.Endpoints)
	thresholdsJSON, _ := json.Marshal(system.Thresholds)

	query := `
		INSERT INTO systems (id, name, description, type, tags, endpoints, thresholds, status, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
		ON CONFLICT (id) DO UPDATE SET
			name = EXCLUDED.name,
			description = EXCLUDED.description,
			type = EXCLUDED.type,
			tags = EXCLUDED.tags,
			endpoints = EXCLUDED.endpoints,
			thresholds = EXCLUDED.thresholds,
			status = EXCLUDED.status,
			updated_at = EXCLUDED.updated_at
	`

	_, err := s.db.Exec(ctx, query,
		system.ID, system.Name, system.Description, system.Type,
		tagsJSON, endpointsJSON, thresholdsJSON, system.Status,
		system.CreatedAt, system.UpdatedAt)

	return err
}

// GetSystem retrieves a system by ID
func (s *PostgresStorage) GetSystem(systemID string) (*models.System, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	query := `
		SELECT id, name, description, type, tags, endpoints, thresholds, status, created_at, updated_at
		FROM systems WHERE id = $1
	`

	row := s.db.QueryRow(ctx, query, systemID)

	var system models.System
	var tagsJSON, endpointsJSON, thresholdsJSON []byte

	err := row.Scan(
		&system.ID, &system.Name, &system.Description, &system.Type,
		&tagsJSON, &endpointsJSON, &thresholdsJSON, &system.Status,
		&system.CreatedAt, &system.UpdatedAt,
	)

	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("system not found: %s", systemID)
		}
		return nil, err
	}

	// Unmarshal JSON fields
	json.Unmarshal(tagsJSON, &system.Tags)
	json.Unmarshal(endpointsJSON, &system.Endpoints)
	json.Unmarshal(thresholdsJSON, &system.Thresholds)

	return &system, nil
}

// UpdateSystem updates an existing system
func (s *PostgresStorage) UpdateSystem(system *models.System) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	tagsJSON, _ := json.Marshal(system.Tags)
	endpointsJSON, _ := json.Marshal(system.Endpoints)
	thresholdsJSON, _ := json.Marshal(system.Thresholds)

	query := `
		UPDATE systems SET 
			name = $2, description = $3, type = $4, tags = $5, 
			endpoints = $6, thresholds = $7, status = $8, updated_at = $9
		WHERE id = $1
	`

	result, err := s.db.Exec(ctx, query,
		system.ID, system.Name, system.Description, system.Type,
		tagsJSON, endpointsJSON, thresholdsJSON, system.Status,
		system.UpdatedAt)

	if err != nil {
		return err
	}

	if result.RowsAffected() == 0 {
		return fmt.Errorf("system not found: %s", system.ID)
	}

	return nil
}

// DeleteSystem deletes a system
func (s *PostgresStorage) DeleteSystem(systemID string) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	query := `DELETE FROM systems WHERE id = $1`
	result, err := s.db.Exec(ctx, query, systemID)

	if err != nil {
		return err
	}

	if result.RowsAffected() == 0 {
		return fmt.Errorf("system not found: %s", systemID)
	}

	return nil
}

// ListSystems retrieves all systems
func (s *PostgresStorage) ListSystems() ([]*models.System, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	query := `
		SELECT id, name, description, type, tags, endpoints, thresholds, status, created_at, updated_at
		FROM systems ORDER BY created_at DESC
	`

	rows, err := s.db.Query(ctx, query)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var systems []*models.System
	for rows.Next() {
		var system models.System
		var tagsJSON, endpointsJSON, thresholdsJSON []byte

		err := rows.Scan(
			&system.ID, &system.Name, &system.Description, &system.Type,
			&tagsJSON, &endpointsJSON, &thresholdsJSON, &system.Status,
			&system.CreatedAt, &system.UpdatedAt,
		)
		if err != nil {
			return nil, err
		}

		// Unmarshal JSON fields
		json.Unmarshal(tagsJSON, &system.Tags)
		json.Unmarshal(endpointsJSON, &system.Endpoints)
		json.Unmarshal(thresholdsJSON, &system.Thresholds)

		systems = append(systems, &system)
	}

	return systems, rows.Err()
}

// StoreMetrics stores multiple metric points
func (s *PostgresStorage) StoreMetrics(systemID string, metrics []*models.MetricPoint) error {
	if len(metrics) == 0 {
		return nil
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	tx, err := s.db.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)

	query := `
		INSERT INTO metrics (system_id, name, value, unit, labels, timestamp)
		VALUES ($1, $2, $3, $4, $5, $6)
		ON CONFLICT (system_id, name, timestamp, labels) DO UPDATE SET
			value = EXCLUDED.value,
			unit = EXCLUDED.unit
	`

	for _, metric := range metrics {
		labelsJSON, _ := json.Marshal(metric.Labels)
		_, err := tx.Exec(ctx, query,
			systemID, metric.Name, metric.Value, metric.Unit,
			labelsJSON, metric.Timestamp)
		if err != nil {
			return err
		}
	}

	return tx.Commit(ctx)
}

// GetSystemMetrics retrieves metrics for a system within a time range
func (s *PostgresStorage) GetSystemMetrics(systemID string, startTime, endTime time.Time) ([]*models.MetricPoint, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	query := `
		SELECT system_id, name, value, unit, labels, timestamp
		FROM metrics
		WHERE system_id = $1 AND timestamp BETWEEN $2 AND $3
		ORDER BY timestamp DESC
	`

	rows, err := s.db.Query(ctx, query, systemID, startTime, endTime)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var metrics []*models.MetricPoint
	for rows.Next() {
		var metric models.MetricPoint
		var labelsJSON []byte

		err := rows.Scan(
			&metric.SystemID, &metric.Name, &metric.Value, &metric.Unit,
			&labelsJSON, &metric.Timestamp,
		)
		if err != nil {
			return nil, err
		}

		json.Unmarshal(labelsJSON, &metric.Labels)
		metrics = append(metrics, &metric)
	}

	return metrics, rows.Err()
}

// GetLatestMetrics retrieves the latest metrics for a system
func (s *PostgresStorage) GetLatestMetrics(systemID string, limit int) ([]*models.MetricPoint, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	query := `
		SELECT DISTINCT ON (name) system_id, name, value, unit, labels, timestamp
		FROM metrics
		WHERE system_id = $1
		ORDER BY name, timestamp DESC
		LIMIT $2
	`

	rows, err := s.db.Query(ctx, query, systemID, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var metrics []*models.MetricPoint
	for rows.Next() {
		var metric models.MetricPoint
		var labelsJSON []byte

		err := rows.Scan(
			&metric.SystemID, &metric.Name, &metric.Value, &metric.Unit,
			&labelsJSON, &metric.Timestamp,
		)
		if err != nil {
			return nil, err
		}

		json.Unmarshal(labelsJSON, &metric.Labels)
		metrics = append(metrics, &metric)
	}

	return metrics, rows.Err()
}

// StoreAlert stores a new alert
func (s *PostgresStorage) StoreAlert(alert *models.Alert) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	labelsJSON, _ := json.Marshal(alert.Labels)

	query := `
		INSERT INTO alerts (id, system_id, name, description, severity, status, 
			metric_name, metric_value, threshold, operator, labels, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
		ON CONFLICT (id) DO UPDATE SET
			status = EXCLUDED.status,
			updated_at = EXCLUDED.updated_at,
			resolved_at = EXCLUDED.resolved_at
	`

	_, err := s.db.Exec(ctx, query,
		alert.ID, alert.SystemID, alert.Name, alert.Description, alert.Severity,
		alert.Status, alert.MetricName, alert.MetricValue, alert.Threshold,
		alert.Operator, labelsJSON, alert.CreatedAt, alert.UpdatedAt)

	return err
}

// GetAlert retrieves an alert by ID
func (s *PostgresStorage) GetAlert(alertID string) (*models.Alert, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	query := `
		SELECT id, system_id, name, description, severity, status, metric_name,
			metric_value, threshold, operator, labels, created_at, updated_at, resolved_at
		FROM alerts WHERE id = $1
	`

	row := s.db.QueryRow(ctx, query, alertID)

	var alert models.Alert
	var labelsJSON []byte

	err := row.Scan(
		&alert.ID, &alert.SystemID, &alert.Name, &alert.Description,
		&alert.Severity, &alert.Status, &alert.MetricName, &alert.MetricValue,
		&alert.Threshold, &alert.Operator, &labelsJSON, &alert.CreatedAt,
		&alert.UpdatedAt, &alert.ResolvedAt,
	)

	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("alert not found: %s", alertID)
		}
		return nil, err
	}

	json.Unmarshal(labelsJSON, &alert.Labels)
	return &alert, nil
}

// UpdateAlert updates an existing alert
func (s *PostgresStorage) UpdateAlert(alert *models.Alert) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	labelsJSON, _ := json.Marshal(alert.Labels)

	query := `
		UPDATE alerts SET 
			status = $2, description = $3, labels = $4, updated_at = $5, resolved_at = $6
		WHERE id = $1
	`

	result, err := s.db.Exec(ctx, query,
		alert.ID, alert.Status, alert.Description, labelsJSON,
		alert.UpdatedAt, alert.ResolvedAt)

	if err != nil {
		return err
	}

	if result.RowsAffected() == 0 {
		return fmt.Errorf("alert not found: %s", alert.ID)
	}

	return nil
}

// DeleteAlert deletes an alert
func (s *PostgresStorage) DeleteAlert(alertID string) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	query := `DELETE FROM alerts WHERE id = $1`
	result, err := s.db.Exec(ctx, query, alertID)

	if err != nil {
		return err
	}

	if result.RowsAffected() == 0 {
		return fmt.Errorf("alert not found: %s", alertID)
	}

	return nil
}

// GetActiveAlerts retrieves active alerts for a system
func (s *PostgresStorage) GetActiveAlerts(systemID string) ([]*models.Alert, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	query := `
		SELECT id, system_id, name, description, severity, status, metric_name,
			metric_value, threshold, operator, labels, created_at, updated_at, resolved_at
		FROM alerts 
		WHERE system_id = $1 AND status IN ('active', 'acknowledged')
		ORDER BY created_at DESC
	`

	rows, err := s.db.Query(ctx, query, systemID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var alerts []*models.Alert
	for rows.Next() {
		var alert models.Alert
		var labelsJSON []byte

		err := rows.Scan(
			&alert.ID, &alert.SystemID, &alert.Name, &alert.Description,
			&alert.Severity, &alert.Status, &alert.MetricName, &alert.MetricValue,
			&alert.Threshold, &alert.Operator, &labelsJSON, &alert.CreatedAt,
			&alert.UpdatedAt, &alert.ResolvedAt,
		)
		if err != nil {
			return nil, err
		}

		json.Unmarshal(labelsJSON, &alert.Labels)
		alerts = append(alerts, &alert)
	}

	return alerts, rows.Err()
}

// ListAlerts retrieves alerts for a system with pagination
func (s *PostgresStorage) ListAlerts(systemID string, limit int) ([]*models.Alert, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	query := `
		SELECT id, system_id, name, description, severity, status, metric_name,
			metric_value, threshold, operator, labels, created_at, updated_at, resolved_at
		FROM alerts 
		WHERE system_id = $1
		ORDER BY created_at DESC
		LIMIT $2
	`

	rows, err := s.db.Query(ctx, query, systemID, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var alerts []*models.Alert
	for rows.Next() {
		var alert models.Alert
		var labelsJSON []byte

		err := rows.Scan(
			&alert.ID, &alert.SystemID, &alert.Name, &alert.Description,
			&alert.Severity, &alert.Status, &alert.MetricName, &alert.MetricValue,
			&alert.Threshold, &alert.Operator, &labelsJSON, &alert.CreatedAt,
			&alert.UpdatedAt, &alert.ResolvedAt,
		)
		if err != nil {
			return nil, err
		}

		json.Unmarshal(labelsJSON, &alert.Labels)
		alerts = append(alerts, &alert)
	}

	return alerts, rows.Err()
}

// Implement remaining methods for alert rules and dashboards...
// (truncated for brevity, but would include full CRUD operations)

// StoreAlertRule stores a new alert rule
func (s *PostgresStorage) StoreAlertRule(rule *models.AlertRule) error {
	// Implementation here
	return nil
}

// GetAlertRule retrieves an alert rule by ID
func (s *PostgresStorage) GetAlertRule(ruleID string) (*models.AlertRule, error) {
	// Implementation here
	return nil, nil
}

// UpdateAlertRule updates an existing alert rule
func (s *PostgresStorage) UpdateAlertRule(rule *models.AlertRule) error {
	// Implementation here
	return nil
}

// DeleteAlertRule deletes an alert rule
func (s *PostgresStorage) DeleteAlertRule(ruleID string) error {
	// Implementation here
	return nil
}

// ListAlertRules retrieves alert rules for a system
func (s *PostgresStorage) ListAlertRules(systemID string) ([]*models.AlertRule, error) {
	// Implementation here
	return nil, nil
}

// StoreDashboard stores a new dashboard
func (s *PostgresStorage) StoreDashboard(dashboard *models.Dashboard) error {
	// Implementation here
	return nil
}

// GetDashboard retrieves a dashboard by ID
func (s *PostgresStorage) GetDashboard(dashboardID string) (*models.Dashboard, error) {
	// Implementation here
	return nil, nil
}

// UpdateDashboard updates an existing dashboard
func (s *PostgresStorage) UpdateDashboard(dashboard *models.Dashboard) error {
	// Implementation here
	return nil
}

// DeleteDashboard deletes a dashboard
func (s *PostgresStorage) DeleteDashboard(dashboardID string) error {
	// Implementation here
	return nil
}

// ListDashboards retrieves dashboards for a user
func (s *PostgresStorage) ListDashboards(userID string) ([]*models.Dashboard, error) {
	// Implementation here
	return nil, nil
}

// HealthCheck performs a health check on the database
func (s *PostgresStorage) HealthCheck() error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	return s.db.Ping(ctx)
}

// Close closes the database connection
func (s *PostgresStorage) Close() error {
	s.db.Close()
	return nil
}

// NewStorage creates a new storage instance based on configuration
func NewStorage(cfg *config.Config, logger *zap.Logger) (Storage, error) {
	switch cfg.Storage.Type {
	case "postgresql":
		return NewPostgresStorage(cfg, logger)
	default:
		return nil, fmt.Errorf("unsupported storage type: %s", cfg.Storage.Type)
	}
} 