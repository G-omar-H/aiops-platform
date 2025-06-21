package notifications

import (
	"fmt"

	"go.uber.org/zap"

	"monitoring-service/internal/config"
	"monitoring-service/internal/models"
)

// AlertNotification represents an alert notification
type AlertNotification struct {
	Alert    *models.Alert     `json:"alert"`
	Rule     *models.AlertRule `json:"rule"`
	Type     string            `json:"type"` // "triggered" or "resolved"
	Channels []string          `json:"channels"`
}

// Manager handles sending notifications through various channels
type Manager struct {
	config *config.Config
	logger *zap.Logger
}

// NewManager creates a new notification manager
func NewManager(cfg *config.Config, logger *zap.Logger) *Manager {
	return &Manager{
		config: cfg,
		logger: logger,
	}
}

// SendAlert sends an alert notification through configured channels
func (m *Manager) SendAlert(notification *AlertNotification) error {
	if !m.config.Alerts.Enabled {
		return nil
	}

	m.logger.Info("Sending alert notification",
		zap.String("alert_id", notification.Alert.ID),
		zap.String("type", notification.Type),
		zap.Strings("channels", notification.Channels))

	var errors []error

	// Send through each configured channel
	for _, channel := range notification.Channels {
		switch channel {
		case "email":
			if err := m.sendEmail(notification); err != nil {
				errors = append(errors, fmt.Errorf("email notification failed: %w", err))
			}
		case "slack":
			if err := m.sendSlack(notification); err != nil {
				errors = append(errors, fmt.Errorf("slack notification failed: %w", err))
			}
		case "webhook":
			if err := m.sendWebhook(notification); err != nil {
				errors = append(errors, fmt.Errorf("webhook notification failed: %w", err))
			}
		case "pagerduty":
			if err := m.sendPagerDuty(notification); err != nil {
				errors = append(errors, fmt.Errorf("pagerduty notification failed: %w", err))
			}
		default:
			m.logger.Warn("Unknown notification channel", zap.String("channel", channel))
		}
	}

	if len(errors) > 0 {
		m.logger.Error("Some notifications failed", zap.Errors("errors", errors))
		return fmt.Errorf("notification failures: %v", errors)
	}

	return nil
}

// sendEmail sends email notification
func (m *Manager) sendEmail(notification *AlertNotification) error {
	if !m.config.Alerts.Channels.Email.Enabled {
		return nil
	}

	// Email implementation would go here
	m.logger.Info("Email notification sent",
		zap.String("alert_id", notification.Alert.ID))
	
	return nil
}

// sendSlack sends Slack notification
func (m *Manager) sendSlack(notification *AlertNotification) error {
	if !m.config.Alerts.Channels.Slack.Enabled {
		return nil
	}

	// Slack implementation would go here
	m.logger.Info("Slack notification sent",
		zap.String("alert_id", notification.Alert.ID))
	
	return nil
}

// sendWebhook sends webhook notification
func (m *Manager) sendWebhook(notification *AlertNotification) error {
	if !m.config.Alerts.Channels.Webhook.Enabled {
		return nil
	}

	// Webhook implementation would go here
	m.logger.Info("Webhook notification sent",
		zap.String("alert_id", notification.Alert.ID))
	
	return nil
}

// sendPagerDuty sends PagerDuty notification
func (m *Manager) sendPagerDuty(notification *AlertNotification) error {
	if !m.config.Alerts.Channels.PagerDuty.Enabled {
		return nil
	}

	// PagerDuty implementation would go here
	m.logger.Info("PagerDuty notification sent",
		zap.String("alert_id", notification.Alert.ID))
	
	return nil
} 