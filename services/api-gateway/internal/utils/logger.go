package utils

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
)

// Logger interface defines logging methods
type Logger interface {
	Debug(msg string, fields ...interface{})
	Info(msg string, fields ...interface{})
	Warn(msg string, fields ...interface{})
	Error(msg string, fields ...interface{})
	Fatal(msg string, fields ...interface{})
	WithContext(ctx context.Context) Logger
	WithFields(fields map[string]interface{}) Logger
}

// StructuredLogger implements the Logger interface using logrus
type StructuredLogger struct {
	logger *logrus.Logger
	entry  *logrus.Entry
}

// NewLogger creates a new structured logger
func NewLogger(level string) Logger {
	logger := logrus.New()

	// Set log level
	logLevel, err := logrus.ParseLevel(strings.ToLower(level))
	if err != nil {
		logLevel = logrus.InfoLevel
	}
	logger.SetLevel(logLevel)

	// Set formatter based on environment
	if os.Getenv("ENVIRONMENT") == "production" {
		// JSON formatter for production
		logger.SetFormatter(&logrus.JSONFormatter{
			TimestampFormat: time.RFC3339,
			FieldMap: logrus.FieldMap{
				logrus.FieldKeyTime:  "timestamp",
				logrus.FieldKeyLevel: "level",
				logrus.FieldKeyMsg:   "message",
				logrus.FieldKeyFunc:  "function",
				logrus.FieldKeyFile:  "file",
			},
		})
	} else {
		// Text formatter for development
		logger.SetFormatter(&logrus.TextFormatter{
			TimestampFormat: "2006-01-02 15:04:05",
			FullTimestamp:   true,
			ForceColors:     true,
		})
	}

	// Set output to stdout
	logger.SetOutput(os.Stdout)

	// Add default fields
	entry := logger.WithFields(logrus.Fields{
		"service": "api-gateway",
		"version": "1.0.0",
	})

	return &StructuredLogger{
		logger: logger,
		entry:  entry,
	}
}

// Debug logs a debug message
func (l *StructuredLogger) Debug(msg string, fields ...interface{}) {
	entry := l.addFields(fields...)
	entry.Debug(msg)
}

// Info logs an info message
func (l *StructuredLogger) Info(msg string, fields ...interface{}) {
	entry := l.addFields(fields...)
	entry.Info(msg)
}

// Warn logs a warning message
func (l *StructuredLogger) Warn(msg string, fields ...interface{}) {
	entry := l.addFields(fields...)
	entry.Warn(msg)
}

// Error logs an error message
func (l *StructuredLogger) Error(msg string, fields ...interface{}) {
	entry := l.addFields(fields...)
	entry.Error(msg)
}

// Fatal logs a fatal message and exits
func (l *StructuredLogger) Fatal(msg string, fields ...interface{}) {
	entry := l.addFields(fields...)
	entry.Fatal(msg)
}

// WithContext adds context information to the logger
func (l *StructuredLogger) WithContext(ctx context.Context) Logger {
	entry := l.entry

	// Extract common context values
	if requestID := ctx.Value("request_id"); requestID != nil {
		entry = entry.WithField("request_id", requestID)
	}

	if userID := ctx.Value("user_id"); userID != nil {
		entry = entry.WithField("user_id", userID)
	}

	if traceID := ctx.Value("trace_id"); traceID != nil {
		entry = entry.WithField("trace_id", traceID)
	}

	return &StructuredLogger{
		logger: l.logger,
		entry:  entry,
	}
}

// WithFields adds structured fields to the logger
func (l *StructuredLogger) WithFields(fields map[string]interface{}) Logger {
	entry := l.entry.WithFields(logrus.Fields(fields))

	return &StructuredLogger{
		logger: l.logger,
		entry:  entry,
	}
}

// addFields converts field pairs to logrus fields
func (l *StructuredLogger) addFields(fields ...interface{}) *logrus.Entry {
	entry := l.entry

	if len(fields)%2 != 0 {
		// If odd number of fields, log a warning and ignore the last field
		entry.Warn("Odd number of fields provided to logger")
		fields = fields[:len(fields)-1]
	}

	// Convert field pairs to map
	logFields := make(logrus.Fields)
	for i := 0; i < len(fields); i += 2 {
		key, ok := fields[i].(string)
		if !ok {
			key = fmt.Sprintf("field_%d", i)
		}
		logFields[key] = fields[i+1]
	}

	if len(logFields) > 0 {
		entry = entry.WithFields(logFields)
	}

	return entry
}

// Performance logger for measuring execution time
type PerformanceLogger struct {
	logger Logger
	start  time.Time
	name   string
	fields map[string]interface{}
}

// NewPerformanceLogger creates a new performance logger
func NewPerformanceLogger(logger Logger, name string) *PerformanceLogger {
	return &PerformanceLogger{
		logger: logger,
		start:  time.Now(),
		name:   name,
		fields: make(map[string]interface{}),
	}
}

// AddField adds a field to the performance logger
func (p *PerformanceLogger) AddField(key string, value interface{}) *PerformanceLogger {
	p.fields[key] = value
	return p
}

// End logs the performance metrics
func (p *PerformanceLogger) End() {
	duration := time.Since(p.start)
	
	fields := []interface{}{
		"operation", p.name,
		"duration_ms", duration.Milliseconds(),
		"duration", duration.String(),
	}

	// Add custom fields
	for k, v := range p.fields {
		fields = append(fields, k, v)
	}

	p.logger.Info("Performance metric", fields...)
}

// RequestLogger logs HTTP request details
func LogHTTPRequest(logger Logger, method, path, userAgent, clientIP string, statusCode int, duration time.Duration, requestID string) {
	logger.Info("HTTP Request",
		"method", method,
		"path", path,
		"status_code", statusCode,
		"duration_ms", duration.Milliseconds(),
		"user_agent", userAgent,
		"client_ip", clientIP,
		"request_id", requestID,
	)
}

// ErrorLogger logs errors with stack trace and context
func LogError(logger Logger, err error, operation string, fields ...interface{}) {
	allFields := []interface{}{
		"error", err.Error(),
		"operation", operation,
	}
	allFields = append(allFields, fields...)
	
	logger.Error("Operation failed", allFields...)
}

// SecurityLogger logs security-related events
func LogSecurityEvent(logger Logger, event, userID, clientIP, details string) {
	logger.Warn("Security Event",
		"event_type", event,
		"user_id", userID,
		"client_ip", clientIP,
		"details", details,
		"timestamp", time.Now().UTC().Format(time.RFC3339),
	)
}

// AuditLogger logs audit events for compliance
func LogAuditEvent(logger Logger, action, resource, userID, result string, metadata map[string]interface{}) {
	fields := []interface{}{
		"audit_action", action,
		"resource", resource,
		"user_id", userID,
		"result", result,
		"timestamp", time.Now().UTC().Format(time.RFC3339),
	}

	// Add metadata fields
	for k, v := range metadata {
		fields = append(fields, k, v)
	}

	logger.Info("Audit Event", fields...)
}

// MetricsLogger logs business metrics
func LogMetric(logger Logger, metricName string, value interface{}, unit string, tags map[string]string) {
	fields := []interface{}{
		"metric_name", metricName,
		"value", value,
		"unit", unit,
		"timestamp", time.Now().UTC().Format(time.RFC3339),
	}

	// Add tags
	for k, v := range tags {
		fields = append(fields, k, v)
	}

	logger.Info("Metric", fields...)
}

// BusinessEventLogger logs business events for analytics
func LogBusinessEvent(logger Logger, eventType string, data map[string]interface{}) {
	fields := []interface{}{
		"event_type", eventType,
		"timestamp", time.Now().UTC().Format(time.RFC3339),
	}

	// Add event data
	for k, v := range data {
		fields = append(fields, k, v)
	}

	logger.Info("Business Event", fields...)
} 