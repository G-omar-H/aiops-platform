package config

import (
	"fmt"
	"strings"
	"time"

	"github.com/spf13/viper"
)

// Config holds all configuration for the monitoring service
type Config struct {
	Environment string          `mapstructure:"environment"`
	Server      ServerConfig    `mapstructure:"server"`
	Log         LogConfig       `mapstructure:"log"`
	Security    SecurityConfig  `mapstructure:"security"`
	Storage     StorageConfig   `mapstructure:"storage"`
	Metrics     MetricsConfig   `mapstructure:"metrics"`
	Alerts      AlertsConfig    `mapstructure:"alerts"`
	Integration IntegrationConfig `mapstructure:"integration"`
}

// ServerConfig holds HTTP server configuration
type ServerConfig struct {
	Port           int `mapstructure:"port"`
	ReadTimeout    int `mapstructure:"read_timeout"`
	WriteTimeout   int `mapstructure:"write_timeout"`
	IdleTimeout    int `mapstructure:"idle_timeout"`
	MaxHeaderBytes int `mapstructure:"max_header_bytes"`
}

// LogConfig holds logging configuration
type LogConfig struct {
	Level  string `mapstructure:"level"`
	Format string `mapstructure:"format"`
	File   string `mapstructure:"file"`
}

// SecurityConfig holds security-related configuration
type SecurityConfig struct {
	JWTSecret     string   `mapstructure:"jwt_secret"`
	CORSOrigins   []string `mapstructure:"cors_origins"`
	RateLimit     int      `mapstructure:"rate_limit"`
	AllowedHosts  []string `mapstructure:"allowed_hosts"`
	TLSEnabled    bool     `mapstructure:"tls_enabled"`
	TLSCertFile   string   `mapstructure:"tls_cert_file"`
	TLSKeyFile    string   `mapstructure:"tls_key_file"`
}

// StorageConfig holds storage backend configuration
type StorageConfig struct {
	Type        string         `mapstructure:"type"`
	PostgreSQL  PostgreSQLConfig `mapstructure:"postgresql"`
	Redis       RedisConfig    `mapstructure:"redis"`
	InfluxDB    InfluxDBConfig `mapstructure:"influxdb"`
	Elasticsearch ElasticsearchConfig `mapstructure:"elasticsearch"`
}

// PostgreSQLConfig holds PostgreSQL configuration
type PostgreSQLConfig struct {
	Host         string `mapstructure:"host"`
	Port         int    `mapstructure:"port"`
	Database     string `mapstructure:"database"`
	Username     string `mapstructure:"username"`
	Password     string `mapstructure:"password"`
	SSLMode      string `mapstructure:"ssl_mode"`
	MaxConns     int    `mapstructure:"max_conns"`
	MinConns     int    `mapstructure:"min_conns"`
	MaxConnTime  int    `mapstructure:"max_conn_time"`
}

// RedisConfig holds Redis configuration
type RedisConfig struct {
	Host        string `mapstructure:"host"`
	Port        int    `mapstructure:"port"`
	Password    string `mapstructure:"password"`
	Database    int    `mapstructure:"database"`
	MaxRetries  int    `mapstructure:"max_retries"`
	PoolSize    int    `mapstructure:"pool_size"`
	IdleTimeout int    `mapstructure:"idle_timeout"`
}

// InfluxDBConfig holds InfluxDB configuration
type InfluxDBConfig struct {
	URL         string `mapstructure:"url"`
	Token       string `mapstructure:"token"`
	Organization string `mapstructure:"organization"`
	Bucket      string `mapstructure:"bucket"`
	BatchSize   int    `mapstructure:"batch_size"`
	FlushInterval int  `mapstructure:"flush_interval"`
}

// ElasticsearchConfig holds Elasticsearch configuration
type ElasticsearchConfig struct {
	URLs     []string `mapstructure:"urls"`
	Username string   `mapstructure:"username"`
	Password string   `mapstructure:"password"`
	Index    string   `mapstructure:"index"`
}

// MetricsConfig holds metrics collection configuration
type MetricsConfig struct {
	CollectionInterval int      `mapstructure:"collection_interval"`
	RetentionDays      int      `mapstructure:"retention_days"`
	Sources            []string `mapstructure:"sources"`
	Prometheus         PrometheusConfig `mapstructure:"prometheus"`
	System             SystemMetricsConfig `mapstructure:"system"`
}

// PrometheusConfig holds Prometheus configuration
type PrometheusConfig struct {
	URL     string            `mapstructure:"url"`
	Queries map[string]string `mapstructure:"queries"`
	Timeout int               `mapstructure:"timeout"`
}

// SystemMetricsConfig holds system metrics configuration
type SystemMetricsConfig struct {
	Enabled   bool     `mapstructure:"enabled"`
	Interval  int      `mapstructure:"interval"`
	Metrics   []string `mapstructure:"metrics"`
}

// AlertsConfig holds alerting configuration
type AlertsConfig struct {
	Enabled         bool              `mapstructure:"enabled"`
	CheckInterval   int               `mapstructure:"check_interval"`
	DefaultSeverity string            `mapstructure:"default_severity"`
	Channels        AlertChannelsConfig `mapstructure:"channels"`
}

// AlertChannelsConfig holds alert channel configuration
type AlertChannelsConfig struct {
	Email    EmailConfig    `mapstructure:"email"`
	Slack    SlackConfig    `mapstructure:"slack"`
	Webhook  WebhookConfig  `mapstructure:"webhook"`
	PagerDuty PagerDutyConfig `mapstructure:"pagerduty"`
}

// EmailConfig holds email configuration
type EmailConfig struct {
	Enabled  bool     `mapstructure:"enabled"`
	SMTPHost string   `mapstructure:"smtp_host"`
	SMTPPort int      `mapstructure:"smtp_port"`
	Username string   `mapstructure:"username"`
	Password string   `mapstructure:"password"`
	From     string   `mapstructure:"from"`
	To       []string `mapstructure:"to"`
}

// SlackConfig holds Slack configuration
type SlackConfig struct {
	Enabled   bool   `mapstructure:"enabled"`
	WebhookURL string `mapstructure:"webhook_url"`
	Channel   string `mapstructure:"channel"`
	Username  string `mapstructure:"username"`
}

// WebhookConfig holds webhook configuration
type WebhookConfig struct {
	Enabled bool   `mapstructure:"enabled"`
	URL     string `mapstructure:"url"`
	Secret  string `mapstructure:"secret"`
	Timeout int    `mapstructure:"timeout"`
}

// PagerDutyConfig holds PagerDuty configuration
type PagerDutyConfig struct {
	Enabled        bool   `mapstructure:"enabled"`
	IntegrationKey string `mapstructure:"integration_key"`
	Severity       string `mapstructure:"severity"`
}

// IntegrationConfig holds external integration configuration
type IntegrationConfig struct {
	AIPrediction   AIPredictionConfig `mapstructure:"ai_prediction"`
	APIGateway     APIGatewayConfig   `mapstructure:"api_gateway"`
	Kafka          KafkaConfig        `mapstructure:"kafka"`
}

// AIPredictionConfig holds AI prediction service configuration
type AIPredictionConfig struct {
	Enabled  bool   `mapstructure:"enabled"`
	URL      string `mapstructure:"url"`
	APIKey   string `mapstructure:"api_key"`
	Timeout  int    `mapstructure:"timeout"`
}

// APIGatewayConfig holds API gateway configuration
type APIGatewayConfig struct {
	URL     string `mapstructure:"url"`
	APIKey  string `mapstructure:"api_key"`
	Timeout int    `mapstructure:"timeout"`
}

// KafkaConfig holds Kafka configuration
type KafkaConfig struct {
	Enabled     bool     `mapstructure:"enabled"`
	Brokers     []string `mapstructure:"brokers"`
	Topics      map[string]string `mapstructure:"topics"`
	GroupID     string   `mapstructure:"group_id"`
	Compression string   `mapstructure:"compression"`
}

// Load loads configuration from environment variables and config files
func Load() (*Config, error) {
	viper.SetConfigName("config")
	viper.SetConfigType("yaml")
	viper.AddConfigPath("./config")
	viper.AddConfigPath("/app/config")
	viper.AddConfigPath(".")

	// Set environment variable prefix
	viper.SetEnvPrefix("MONITORING")
	viper.AutomaticEnv()
	viper.SetEnvKeyReplacer(strings.NewReplacer(".", "_"))

	// Set defaults
	setDefaults()

	// Read config file if it exists
	if err := viper.ReadInConfig(); err != nil {
		if _, ok := err.(viper.ConfigFileNotFoundError); !ok {
			return nil, fmt.Errorf("failed to read config file: %w", err)
		}
	}

	var config Config
	if err := viper.Unmarshal(&config); err != nil {
		return nil, fmt.Errorf("failed to unmarshal config: %w", err)
	}

	return &config, nil
}

// setDefaults sets default configuration values
func setDefaults() {
	// Environment
	viper.SetDefault("environment", "development")

	// Server defaults
	viper.SetDefault("server.port", 8080)
	viper.SetDefault("server.read_timeout", 30)
	viper.SetDefault("server.write_timeout", 30)
	viper.SetDefault("server.idle_timeout", 120)
	viper.SetDefault("server.max_header_bytes", 1048576) // 1MB

	// Log defaults
	viper.SetDefault("log.level", "info")
	viper.SetDefault("log.format", "json")

	// Security defaults
	viper.SetDefault("security.cors_origins", []string{"*"})
	viper.SetDefault("security.rate_limit", 100)
	viper.SetDefault("security.tls_enabled", false)

	// Storage defaults
	viper.SetDefault("storage.type", "postgresql")
	
	// PostgreSQL defaults
	viper.SetDefault("storage.postgresql.host", "localhost")
	viper.SetDefault("storage.postgresql.port", 5432)
	viper.SetDefault("storage.postgresql.database", "aiops_monitoring")
	viper.SetDefault("storage.postgresql.username", "aiops")
	viper.SetDefault("storage.postgresql.ssl_mode", "disable")
	viper.SetDefault("storage.postgresql.max_conns", 20)
	viper.SetDefault("storage.postgresql.min_conns", 5)
	viper.SetDefault("storage.postgresql.max_conn_time", 300)

	// Redis defaults
	viper.SetDefault("storage.redis.host", "localhost")
	viper.SetDefault("storage.redis.port", 6379)
	viper.SetDefault("storage.redis.database", 0)
	viper.SetDefault("storage.redis.max_retries", 3)
	viper.SetDefault("storage.redis.pool_size", 10)
	viper.SetDefault("storage.redis.idle_timeout", 300)

	// InfluxDB defaults
	viper.SetDefault("storage.influxdb.url", "http://localhost:8086")
	viper.SetDefault("storage.influxdb.organization", "aiops")
	viper.SetDefault("storage.influxdb.bucket", "metrics")
	viper.SetDefault("storage.influxdb.batch_size", 1000)
	viper.SetDefault("storage.influxdb.flush_interval", 10)

	// Metrics defaults
	viper.SetDefault("metrics.collection_interval", 30)
	viper.SetDefault("metrics.retention_days", 30)
	viper.SetDefault("metrics.sources", []string{"prometheus", "system"})
	viper.SetDefault("metrics.prometheus.url", "http://localhost:9090")
	viper.SetDefault("metrics.prometheus.timeout", 30)
	viper.SetDefault("metrics.system.enabled", true)
	viper.SetDefault("metrics.system.interval", 10)
	viper.SetDefault("metrics.system.metrics", []string{"cpu", "memory", "disk", "network"})

	// Alerts defaults
	viper.SetDefault("alerts.enabled", true)
	viper.SetDefault("alerts.check_interval", 60)
	viper.SetDefault("alerts.default_severity", "medium")

	// Integration defaults
	viper.SetDefault("integration.ai_prediction.enabled", true)
	viper.SetDefault("integration.ai_prediction.url", "http://ai-prediction:8000")
	viper.SetDefault("integration.ai_prediction.timeout", 30)
	viper.SetDefault("integration.api_gateway.url", "http://api-gateway:8080")
	viper.SetDefault("integration.api_gateway.timeout", 30)
	viper.SetDefault("integration.kafka.enabled", false)
	viper.SetDefault("integration.kafka.brokers", []string{"localhost:9092"})
	viper.SetDefault("integration.kafka.group_id", "monitoring-service")
	viper.SetDefault("integration.kafka.compression", "snappy")
}

// Validate validates the configuration
func (c *Config) Validate() error {
	if c.Security.JWTSecret == "" {
		return fmt.Errorf("JWT secret is required")
	}

	if c.Server.Port <= 0 || c.Server.Port > 65535 {
		return fmt.Errorf("invalid server port: %d", c.Server.Port)
	}

	if c.Storage.Type == "" {
		return fmt.Errorf("storage type is required")
	}

	return nil
}

// GetDSN returns database connection string
func (c *PostgreSQLConfig) GetDSN() string {
	return fmt.Sprintf(
		"host=%s port=%d user=%s password=%s dbname=%s sslmode=%s",
		c.Host, c.Port, c.Username, c.Password, c.Database, c.SSLMode,
	)
}

// GetRedisAddr returns Redis address
func (c *RedisConfig) GetRedisAddr() string {
	return fmt.Sprintf("%s:%d", c.Host, c.Port)
} 