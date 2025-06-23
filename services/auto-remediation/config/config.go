package config

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/spf13/viper"
	"gopkg.in/yaml.v3"
)

// Config represents the application configuration
type Config struct {
	Environment string         `mapstructure:"environment" yaml:"environment"`
	Server      ServerConfig   `mapstructure:"server" yaml:"server"`
	Database    DatabaseConfig `mapstructure:"database" yaml:"database"`
	Redis       RedisConfig    `mapstructure:"redis" yaml:"redis"`
	Security    SecurityConfig `mapstructure:"security" yaml:"security"`
	RateLimit   RateLimitConfig `mapstructure:"rate_limit" yaml:"rate_limit"`
	Workflow    WorkflowConfig `mapstructure:"workflow" yaml:"workflow"`
	Safety      SafetyConfig   `mapstructure:"safety" yaml:"safety"`
	Integrations IntegrationsConfig `mapstructure:"integrations" yaml:"integrations"`
	Monitoring  MonitoringConfig `mapstructure:"monitoring" yaml:"monitoring"`
	Features    FeatureFlags   `mapstructure:"features" yaml:"features"`
}

// ServerConfig holds server configuration
type ServerConfig struct {
	Port           int    `mapstructure:"port" yaml:"port"`
	Host           string `mapstructure:"host" yaml:"host"`
	ReadTimeout    int    `mapstructure:"read_timeout" yaml:"read_timeout"`
	WriteTimeout   int    `mapstructure:"write_timeout" yaml:"write_timeout"`
	IdleTimeout    int    `mapstructure:"idle_timeout" yaml:"idle_timeout"`
	MaxHeaderBytes int    `mapstructure:"max_header_bytes" yaml:"max_header_bytes"`
	EnableSwagger  bool   `mapstructure:"enable_swagger" yaml:"enable_swagger"`
	EnableMetrics  bool   `mapstructure:"enable_metrics" yaml:"enable_metrics"`
}

// DatabaseConfig holds database configuration
type DatabaseConfig struct {
	URL                string `mapstructure:"url" yaml:"url"`
	MaxOpenConnections int    `mapstructure:"max_open_connections" yaml:"max_open_connections"`
	MaxIdleConnections int    `mapstructure:"max_idle_connections" yaml:"max_idle_connections"`
	ConnectionMaxLife  int    `mapstructure:"connection_max_life" yaml:"connection_max_life"`
	EnableLogging      bool   `mapstructure:"enable_logging" yaml:"enable_logging"`
	MigrationPath      string `mapstructure:"migration_path" yaml:"migration_path"`
}

// RedisConfig holds Redis configuration
type RedisConfig struct {
	URL         string `mapstructure:"url" yaml:"url"`
	Password    string `mapstructure:"password" yaml:"password"`
	Database    int    `mapstructure:"database" yaml:"database"`
	MaxRetries  int    `mapstructure:"max_retries" yaml:"max_retries"`
	PoolSize    int    `mapstructure:"pool_size" yaml:"pool_size"`
	PoolTimeout int    `mapstructure:"pool_timeout" yaml:"pool_timeout"`
	EnableTLS   bool   `mapstructure:"enable_tls" yaml:"enable_tls"`
}

// SecurityConfig holds security configuration
type SecurityConfig struct {
	JWTSecret           string   `mapstructure:"jwt_secret" yaml:"jwt_secret"`
	JWTExpirationHours  int      `mapstructure:"jwt_expiration_hours" yaml:"jwt_expiration_hours"`
	EncryptionKey       string   `mapstructure:"encryption_key" yaml:"encryption_key"`
	AllowedOrigins      []string `mapstructure:"allowed_origins" yaml:"allowed_origins"`
	RequireHTTPS        bool     `mapstructure:"require_https" yaml:"require_https"`
	EnableCSRF          bool     `mapstructure:"enable_csrf" yaml:"enable_csrf"`
	CSRFTokenLength     int      `mapstructure:"csrf_token_length" yaml:"csrf_token_length"`
	SessionTimeout      int      `mapstructure:"session_timeout" yaml:"session_timeout"`
	MaxLoginAttempts    int      `mapstructure:"max_login_attempts" yaml:"max_login_attempts"`
	LockoutDuration     int      `mapstructure:"lockout_duration" yaml:"lockout_duration"`
}

// RateLimitConfig holds rate limiting configuration
type RateLimitConfig struct {
	RequestsPerMinute   int               `mapstructure:"requests_per_minute" yaml:"requests_per_minute"`
	BurstSize           int               `mapstructure:"burst_size" yaml:"burst_size"`
	EnableRateLimit     bool              `mapstructure:"enable_rate_limit" yaml:"enable_rate_limit"`
	EndpointLimits      map[string]int    `mapstructure:"endpoint_limits" yaml:"endpoint_limits"`
	WhitelistIPs        []string          `mapstructure:"whitelist_ips" yaml:"whitelist_ips"`
	CustomLimits        map[string]string `mapstructure:"custom_limits" yaml:"custom_limits"`
}

// WorkflowConfig holds workflow engine configuration
type WorkflowConfig struct {
	MaxConcurrentExecutions int               `mapstructure:"max_concurrent_executions" yaml:"max_concurrent_executions"`
	DefaultTimeout          int               `mapstructure:"default_timeout" yaml:"default_timeout"`
	MaxRetryAttempts        int               `mapstructure:"max_retry_attempts" yaml:"max_retry_attempts"`
	RetryBackoffMultiplier  float64           `mapstructure:"retry_backoff_multiplier" yaml:"retry_backoff_multiplier"`
	EnableParallelExecution bool              `mapstructure:"enable_parallel_execution" yaml:"enable_parallel_execution"`
	WorkflowStoragePath     string            `mapstructure:"workflow_storage_path" yaml:"workflow_storage_path"`
	TemplatesPath           string            `mapstructure:"templates_path" yaml:"templates_path"`
	CustomActionPaths       []string          `mapstructure:"custom_action_paths" yaml:"custom_action_paths"`
	EnvironmentVariables    map[string]string `mapstructure:"environment_variables" yaml:"environment_variables"`
}

// SafetyConfig holds safety and compliance configuration
type SafetyConfig struct {
	RequireApproval           bool              `mapstructure:"require_approval" yaml:"require_approval"`
	ApprovalTimeout           int               `mapstructure:"approval_timeout" yaml:"approval_timeout"`
	EnableRollback            bool              `mapstructure:"enable_rollback" yaml:"enable_rollback"`
	RollbackTimeout           int               `mapstructure:"rollback_timeout" yaml:"rollback_timeout"`
	MaxActionsPerHour         int               `mapstructure:"max_actions_per_hour" yaml:"max_actions_per_hour"`
	EnableSafetyChecks        bool              `mapstructure:"enable_safety_checks" yaml:"enable_safety_checks"`
	RequiredSafetyChecks      []string          `mapstructure:"required_safety_checks" yaml:"required_safety_checks"`
	EmergencyStopEnabled      bool              `mapstructure:"emergency_stop_enabled" yaml:"emergency_stop_enabled"`
	MaintenanceMode           bool              `mapstructure:"maintenance_mode" yaml:"maintenance_mode"`
	AllowedActionTypes        []string          `mapstructure:"allowed_action_types" yaml:"allowed_action_types"`
	RestrictedEnvironments    []string          `mapstructure:"restricted_environments" yaml:"restricted_environments"`
	ComplianceMode            string            `mapstructure:"compliance_mode" yaml:"compliance_mode"`
	AuditLevel                string            `mapstructure:"audit_level" yaml:"audit_level"`
	SafetyThresholds          map[string]float64 `mapstructure:"safety_thresholds" yaml:"safety_thresholds"`
}

// IntegrationsConfig holds external integrations configuration
type IntegrationsConfig struct {
	AIService          AIServiceConfig        `mapstructure:"ai_service" yaml:"ai_service"`
	MonitoringService  MonitoringServiceConfig `mapstructure:"monitoring_service" yaml:"monitoring_service"`
	Notifications      NotificationsConfig    `mapstructure:"notifications" yaml:"notifications"`
	CloudProviders     CloudProvidersConfig   `mapstructure:"cloud_providers" yaml:"cloud_providers"`
	ContainerPlatforms ContainerPlatformsConfig `mapstructure:"container_platforms" yaml:"container_platforms"`
	ITSM               ITSMConfig             `mapstructure:"itsm" yaml:"itsm"`
}

// AIServiceConfig holds AI service integration configuration
type AIServiceConfig struct {
	URL                 string            `mapstructure:"url" yaml:"url"`
	APIKey              string            `mapstructure:"api_key" yaml:"api_key"`
	Timeout             int               `mapstructure:"timeout" yaml:"timeout"`
	MaxRetries          int               `mapstructure:"max_retries" yaml:"max_retries"`
	EnablePredictions   bool              `mapstructure:"enable_predictions" yaml:"enable_predictions"`
	PredictionThreshold float64           `mapstructure:"prediction_threshold" yaml:"prediction_threshold"`
	CustomHeaders       map[string]string `mapstructure:"custom_headers" yaml:"custom_headers"`
}

// MonitoringServiceConfig holds monitoring service configuration
type MonitoringServiceConfig struct {
	URL                 string            `mapstructure:"url" yaml:"url"`
	APIKey              string            `mapstructure:"api_key" yaml:"api_key"`
	Timeout             int               `mapstructure:"timeout" yaml:"timeout"`
	EnableMetrics       bool              `mapstructure:"enable_metrics" yaml:"enable_metrics"`
	MetricsInterval     int               `mapstructure:"metrics_interval" yaml:"metrics_interval"`
	AlertingEndpoint    string            `mapstructure:"alerting_endpoint" yaml:"alerting_endpoint"`
	CustomHeaders       map[string]string `mapstructure:"custom_headers" yaml:"custom_headers"`
}

// NotificationsConfig holds notification service configuration
type NotificationsConfig struct {
	Slack     SlackConfig     `mapstructure:"slack" yaml:"slack"`
	PagerDuty PagerDutyConfig `mapstructure:"pagerduty" yaml:"pagerduty"`
	Email     EmailConfig     `mapstructure:"email" yaml:"email"`
	Webhook   WebhookConfig   `mapstructure:"webhook" yaml:"webhook"`
}

// SlackConfig holds Slack configuration
type SlackConfig struct {
	Token                string   `mapstructure:"token" yaml:"token"`
	Channel              string   `mapstructure:"channel" yaml:"channel"`
	EnableNotifications  bool     `mapstructure:"enable_notifications" yaml:"enable_notifications"`
	NotificationLevels   []string `mapstructure:"notification_levels" yaml:"notification_levels"`
	ThreadNotifications  bool     `mapstructure:"thread_notifications" yaml:"thread_notifications"`
}

// PagerDutyConfig holds PagerDuty configuration
type PagerDutyConfig struct {
	APIKey              string `mapstructure:"api_key" yaml:"api_key"`
	ServiceKey          string `mapstructure:"service_key" yaml:"service_key"`
	EnableNotifications bool   `mapstructure:"enable_notifications" yaml:"enable_notifications"`
	EscalationPolicy    string `mapstructure:"escalation_policy" yaml:"escalation_policy"`
}

// EmailConfig holds email configuration
type EmailConfig struct {
	SMTPHost            string   `mapstructure:"smtp_host" yaml:"smtp_host"`
	SMTPPort            int      `mapstructure:"smtp_port" yaml:"smtp_port"`
	Username            string   `mapstructure:"username" yaml:"username"`
	Password            string   `mapstructure:"password" yaml:"password"`
	FromEmail           string   `mapstructure:"from_email" yaml:"from_email"`
	EnableNotifications bool     `mapstructure:"enable_notifications" yaml:"enable_notifications"`
	Recipients          []string `mapstructure:"recipients" yaml:"recipients"`
	EnableTLS           bool     `mapstructure:"enable_tls" yaml:"enable_tls"`
}

// WebhookConfig holds webhook configuration
type WebhookConfig struct {
	URLs                []string          `mapstructure:"urls" yaml:"urls"`
	EnableNotifications bool              `mapstructure:"enable_notifications" yaml:"enable_notifications"`
	Timeout             int               `mapstructure:"timeout" yaml:"timeout"`
	MaxRetries          int               `mapstructure:"max_retries" yaml:"max_retries"`
	CustomHeaders       map[string]string `mapstructure:"custom_headers" yaml:"custom_headers"`
}

// CloudProvidersConfig holds cloud provider configurations
type CloudProvidersConfig struct {
	AWS   AWSConfig   `mapstructure:"aws" yaml:"aws"`
	Azure AzureConfig `mapstructure:"azure" yaml:"azure"`
	GCP   GCPConfig   `mapstructure:"gcp" yaml:"gcp"`
}

// AWSConfig holds AWS configuration
type AWSConfig struct {
	Region          string `mapstructure:"region" yaml:"region"`
	AccessKeyID     string `mapstructure:"access_key_id" yaml:"access_key_id"`
	SecretAccessKey string `mapstructure:"secret_access_key" yaml:"secret_access_key"`
	SessionToken    string `mapstructure:"session_token" yaml:"session_token"`
	EnableIntegration bool  `mapstructure:"enable_integration" yaml:"enable_integration"`
}

// AzureConfig holds Azure configuration
type AzureConfig struct {
	TenantID         string `mapstructure:"tenant_id" yaml:"tenant_id"`
	ClientID         string `mapstructure:"client_id" yaml:"client_id"`
	ClientSecret     string `mapstructure:"client_secret" yaml:"client_secret"`
	SubscriptionID   string `mapstructure:"subscription_id" yaml:"subscription_id"`
	EnableIntegration bool  `mapstructure:"enable_integration" yaml:"enable_integration"`
}

// GCPConfig holds GCP configuration
type GCPConfig struct {
	ProjectID         string `mapstructure:"project_id" yaml:"project_id"`
	CredentialsPath   string `mapstructure:"credentials_path" yaml:"credentials_path"`
	Zone              string `mapstructure:"zone" yaml:"zone"`
	EnableIntegration bool   `mapstructure:"enable_integration" yaml:"enable_integration"`
}

// ContainerPlatformsConfig holds container platform configurations
type ContainerPlatformsConfig struct {
	Kubernetes KubernetesConfig `mapstructure:"kubernetes" yaml:"kubernetes"`
	Docker     DockerConfig     `mapstructure:"docker" yaml:"docker"`
}

// KubernetesConfig holds Kubernetes configuration
type KubernetesConfig struct {
	ConfigPath        string   `mapstructure:"config_path" yaml:"config_path"`
	Context           string   `mapstructure:"context" yaml:"context"`
	Namespace         string   `mapstructure:"namespace" yaml:"namespace"`
	EnableIntegration bool     `mapstructure:"enable_integration" yaml:"enable_integration"`
	AllowedNamespaces []string `mapstructure:"allowed_namespaces" yaml:"allowed_namespaces"`
}

// DockerConfig holds Docker configuration
type DockerConfig struct {
	Host              string `mapstructure:"host" yaml:"host"`
	APIVersion        string `mapstructure:"api_version" yaml:"api_version"`
	EnableIntegration bool   `mapstructure:"enable_integration" yaml:"enable_integration"`
	TLSVerify         bool   `mapstructure:"tls_verify" yaml:"tls_verify"`
	CertPath          string `mapstructure:"cert_path" yaml:"cert_path"`
}

// ITSMConfig holds ITSM integration configuration
type ITSMConfig struct {
	ServiceNow ServiceNowConfig `mapstructure:"servicenow" yaml:"servicenow"`
	Jira       JiraConfig       `mapstructure:"jira" yaml:"jira"`
}

// ServiceNowConfig holds ServiceNow configuration
type ServiceNowConfig struct {
	Instance          string `mapstructure:"instance" yaml:"instance"`
	Username          string `mapstructure:"username" yaml:"username"`
	Password          string `mapstructure:"password" yaml:"password"`
	EnableIntegration bool   `mapstructure:"enable_integration" yaml:"enable_integration"`
	Table             string `mapstructure:"table" yaml:"table"`
}

// JiraConfig holds Jira configuration
type JiraConfig struct {
	URL               string `mapstructure:"url" yaml:"url"`
	Username          string `mapstructure:"username" yaml:"username"`
	APIToken          string `mapstructure:"api_token" yaml:"api_token"`
	EnableIntegration bool   `mapstructure:"enable_integration" yaml:"enable_integration"`
	Project           string `mapstructure:"project" yaml:"project"`
	IssueType         string `mapstructure:"issue_type" yaml:"issue_type"`
}

// MonitoringConfig holds monitoring and observability configuration
type MonitoringConfig struct {
	LogLevel        string            `mapstructure:"log_level" yaml:"log_level"`
	LogFormat       string            `mapstructure:"log_format" yaml:"log_format"`
	LogFile         string            `mapstructure:"log_file" yaml:"log_file"`
	EnableMetrics   bool              `mapstructure:"enable_metrics" yaml:"enable_metrics"`
	MetricsPort     int               `mapstructure:"metrics_port" yaml:"metrics_port"`
	EnableTracing   bool              `mapstructure:"enable_tracing" yaml:"enable_tracing"`
	TracingEndpoint string            `mapstructure:"tracing_endpoint" yaml:"tracing_endpoint"`
	HealthCheckPath string            `mapstructure:"health_check_path" yaml:"health_check_path"`
	CustomMetrics   map[string]string `mapstructure:"custom_metrics" yaml:"custom_metrics"`
}

// FeatureFlags holds feature flag configuration
type FeatureFlags struct {
	EnableWorkflowEngine     bool `mapstructure:"enable_workflow_engine" yaml:"enable_workflow_engine"`
	EnableAdvancedSafety     bool `mapstructure:"enable_advanced_safety" yaml:"enable_advanced_safety"`
	EnableMLPredictions      bool `mapstructure:"enable_ml_predictions" yaml:"enable_ml_predictions"`
	EnableCloudIntegrations  bool `mapstructure:"enable_cloud_integrations" yaml:"enable_cloud_integrations"`
	EnableContainerSupport   bool `mapstructure:"enable_container_support" yaml:"enable_container_support"`
	EnableITSMIntegrations   bool `mapstructure:"enable_itsm_integrations" yaml:"enable_itsm_integrations"`
	EnableAdvancedAuditing   bool `mapstructure:"enable_advanced_auditing" yaml:"enable_advanced_auditing"`
	EnableCustomActions      bool `mapstructure:"enable_custom_actions" yaml:"enable_custom_actions"`
	EnableBatchOperations    bool `mapstructure:"enable_batch_operations" yaml:"enable_batch_operations"`
	EnableScheduledActions   bool `mapstructure:"enable_scheduled_actions" yaml:"enable_scheduled_actions"`
}

// Load loads configuration from environment variables and config files
func Load() (*Config, error) {
	config := &Config{}

	// Set default values
	setDefaults(config)

	// Configure viper
	viper.SetConfigName("config")
	viper.SetConfigType("yaml")
	viper.AddConfigPath("./config")
	viper.AddConfigPath(".")
	viper.AutomaticEnv()
	viper.SetEnvKeyReplacer(strings.NewReplacer(".", "_"))

	// Read config file if it exists
	if err := viper.ReadInConfig(); err != nil {
		if _, ok := err.(viper.ConfigFileNotFoundError); !ok {
			return nil, fmt.Errorf("failed to read config file: %w", err)
		}
	}

	// Override with environment variables
	loadFromEnv(config)

	// Unmarshal config
	if err := viper.Unmarshal(config); err != nil {
		return nil, fmt.Errorf("failed to unmarshal config: %w", err)
	}

	// Validate configuration
	if err := validate(config); err != nil {
		return nil, fmt.Errorf("config validation failed: %w", err)
	}

	return config, nil
}

// setDefaults sets default configuration values
func setDefaults(config *Config) {
	config.Environment = "development"
	config.Server = ServerConfig{
		Port:           8080,
		Host:           "0.0.0.0",
		ReadTimeout:    30,
		WriteTimeout:   30,
		IdleTimeout:    60,
		MaxHeaderBytes: 1 << 20, // 1MB
		EnableSwagger:  true,
		EnableMetrics:  true,
	}
	
	config.Database = DatabaseConfig{
		MaxOpenConnections: 25,
		MaxIdleConnections: 5,
		ConnectionMaxLife:  300,
		EnableLogging:      false,
		MigrationPath:      "./migrations",
	}
	
	config.Redis = RedisConfig{
		Database:    0,
		MaxRetries:  3,
		PoolSize:    10,
		PoolTimeout: 30,
		EnableTLS:   false,
	}
	
	config.Security = SecurityConfig{
		JWTExpirationHours: 24,
		RequireHTTPS:       false,
		EnableCSRF:         true,
		CSRFTokenLength:    32,
		SessionTimeout:     3600,
		MaxLoginAttempts:   5,
		LockoutDuration:    300,
	}
	
	config.RateLimit = RateLimitConfig{
		RequestsPerMinute: 100,
		BurstSize:         200,
		EnableRateLimit:   true,
		EndpointLimits: map[string]int{
			"/api/v1/remediation/execute": 20,
			"/api/v1/remediation/batch":   5,
		},
	}
	
	config.Workflow = WorkflowConfig{
		MaxConcurrentExecutions: 10,
		DefaultTimeout:          300,
		MaxRetryAttempts:        3,
		RetryBackoffMultiplier:  2.0,
		EnableParallelExecution: true,
		WorkflowStoragePath:     "./workflows",
		TemplatesPath:           "./templates",
	}
	
	config.Safety = SafetyConfig{
		RequireApproval:        false,
		ApprovalTimeout:        1800,
		EnableRollback:         true,
		RollbackTimeout:        300,
		MaxActionsPerHour:      50,
		EnableSafetyChecks:     true,
		EmergencyStopEnabled:   true,
		MaintenanceMode:        false,
		ComplianceMode:         "standard",
		AuditLevel:             "info",
		RequiredSafetyChecks:   []string{"system_health", "resource_availability"},
		AllowedActionTypes:     []string{"restart", "scale", "update_config"},
		SafetyThresholds: map[string]float64{
			"cpu_usage":    80.0,
			"memory_usage": 85.0,
			"disk_usage":   90.0,
		},
	}
	
	config.Monitoring = MonitoringConfig{
		LogLevel:        "info",
		LogFormat:       "json",
		EnableMetrics:   true,
		MetricsPort:     9090,
		EnableTracing:   false,
		HealthCheckPath: "/health",
	}
	
	config.Features = FeatureFlags{
		EnableWorkflowEngine:    true,
		EnableAdvancedSafety:    true,
		EnableMLPredictions:     true,
		EnableCloudIntegrations: true,
		EnableContainerSupport:  true,
		EnableAdvancedAuditing:  true,
		EnableCustomActions:     true,
		EnableBatchOperations:   true,
		EnableScheduledActions:  true,
	}
}

// loadFromEnv loads configuration from environment variables
func loadFromEnv(config *Config) {
	if val := os.Getenv("ENVIRONMENT"); val != "" {
		config.Environment = val
	}
	
	if val := os.Getenv("PORT"); val != "" {
		if port, err := strconv.Atoi(val); err == nil {
			config.Server.Port = port
		}
	}
	
	if val := os.Getenv("DATABASE_URL"); val != "" {
		config.Database.URL = val
	}
	
	if val := os.Getenv("REDIS_URL"); val != "" {
		config.Redis.URL = val
	}
	
	if val := os.Getenv("JWT_SECRET"); val != "" {
		config.Security.JWTSecret = val
	}
	
	if val := os.Getenv("ENCRYPTION_KEY"); val != "" {
		config.Security.EncryptionKey = val
	}
	
	// AI Service configuration
	if val := os.Getenv("AI_SERVICE_URL"); val != "" {
		config.Integrations.AIService.URL = val
	}
	
	if val := os.Getenv("AI_SERVICE_API_KEY"); val != "" {
		config.Integrations.AIService.APIKey = val
	}
	
	// Monitoring Service configuration
	if val := os.Getenv("MONITORING_SERVICE_URL"); val != "" {
		config.Integrations.MonitoringService.URL = val
	}
	
	// Cloud provider configurations
	if val := os.Getenv("AWS_REGION"); val != "" {
		config.Integrations.CloudProviders.AWS.Region = val
	}
	
	if val := os.Getenv("AWS_ACCESS_KEY_ID"); val != "" {
		config.Integrations.CloudProviders.AWS.AccessKeyID = val
	}
	
	if val := os.Getenv("AWS_SECRET_ACCESS_KEY"); val != "" {
		config.Integrations.CloudProviders.AWS.SecretAccessKey = val
	}
	
	// Kubernetes configuration
	if val := os.Getenv("KUBECONFIG"); val != "" {
		config.Integrations.ContainerPlatforms.Kubernetes.ConfigPath = val
	}
	
	// Notification configurations
	if val := os.Getenv("SLACK_TOKEN"); val != "" {
		config.Integrations.Notifications.Slack.Token = val
	}
	
	if val := os.Getenv("PAGERDUTY_API_KEY"); val != "" {
		config.Integrations.Notifications.PagerDuty.APIKey = val
	}
}

// validate validates the configuration
func validate(config *Config) error {
	if config.Database.URL == "" {
		return fmt.Errorf("database URL is required")
	}
	
	if config.Security.JWTSecret == "" {
		return fmt.Errorf("JWT secret is required")
	}
	
	if config.Server.Port <= 0 || config.Server.Port > 65535 {
		return fmt.Errorf("server port must be between 1 and 65535")
	}
	
	if config.Workflow.MaxConcurrentExecutions <= 0 {
		return fmt.Errorf("max concurrent executions must be greater than 0")
	}
	
	if config.Safety.MaxActionsPerHour <= 0 {
		return fmt.Errorf("max actions per hour must be greater than 0")
	}
	
	return nil
}

// ToYAML exports configuration to YAML format
func (c *Config) ToYAML() ([]byte, error) {
	return yaml.Marshal(c)
}

// IsProduction returns true if running in production environment
func (c *Config) IsProduction() bool {
	return c.Environment == "production"
}

// IsDevelopment returns true if running in development environment
func (c *Config) IsDevelopment() bool {
	return c.Environment == "development"
} 