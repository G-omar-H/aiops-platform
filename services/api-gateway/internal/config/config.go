package config

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

// Config holds all configuration for the API Gateway
type Config struct {
	// Server configuration
	Environment string `json:"environment"`
	Port        string `json:"port"`
	LogLevel    string `json:"log_level"`
	Debug       bool   `json:"debug"`

	// Database configuration
	Database DatabaseConfig `json:"database"`

	// Redis configuration
	Redis RedisConfig `json:"redis"`

	// Kafka configuration
	Kafka KafkaConfig `json:"kafka"`

	// JWT configuration
	JWT JWTConfig `json:"jwt"`

	// CORS configuration
	CORS CORSConfig `json:"cors"`

	// Rate limiting configuration
	RateLimit RateLimitConfig `json:"rate_limit"`

	// External services
	Services ServicesConfig `json:"services"`

	// Security configuration
	Security SecurityConfig `json:"security"`

	// Swagger configuration
	Swagger SwaggerConfig `json:"swagger"`

	// Monitoring configuration
	Monitoring MonitoringConfig `json:"monitoring"`

	// AI/ML configuration
	AI AIConfig `json:"ai"`

	// Feature flags
	Features FeatureFlags `json:"features"`
}

// DatabaseConfig holds database configuration
type DatabaseConfig struct {
	URL             string        `json:"url"`
	Host            string        `json:"host"`
	Port            int           `json:"port"`
	Name            string        `json:"name"`
	User            string        `json:"user"`
	Password        string        `json:"password"`
	SSLMode         string        `json:"ssl_mode"`
	MaxConnections  int           `json:"max_connections"`
	IdleConnections int           `json:"idle_connections"`
	ConnMaxLifetime time.Duration `json:"conn_max_lifetime"`
}

// RedisConfig holds Redis configuration
type RedisConfig struct {
	URL      string `json:"url"`
	Host     string `json:"host"`
	Port     int    `json:"port"`
	Password string `json:"password"`
	DB       int    `json:"db"`
	PoolSize int    `json:"pool_size"`
}

// KafkaConfig holds Kafka configuration
type KafkaConfig struct {
	Brokers       []string `json:"brokers"`
	GroupID       string   `json:"group_id"`
	ClientID      string   `json:"client_id"`
	RetryMax      int      `json:"retry_max"`
	BatchSize     int      `json:"batch_size"`
	FlushInterval int      `json:"flush_interval"`
}

// JWTConfig holds JWT configuration
type JWTConfig struct {
	Secret     string        `json:"secret"`
	Expiration time.Duration `json:"expiration"`
	Issuer     string        `json:"issuer"`
	Algorithm  string        `json:"algorithm"`
}

// CORSConfig holds CORS configuration
type CORSConfig struct {
	AllowedOrigins []string `json:"allowed_origins"`
	AllowedMethods []string `json:"allowed_methods"`
	AllowedHeaders []string `json:"allowed_headers"`
}

// RateLimitConfig holds rate limiting configuration
type RateLimitConfig struct {
	RequestsPerMinute int           `json:"requests_per_minute"`
	Burst             int           `json:"burst"`
	Window            time.Duration `json:"window"`
}

// ServicesConfig holds external services configuration
type ServicesConfig struct {
	MonitoringService  ServiceEndpoint `json:"monitoring_service"`
	PredictionService  ServiceEndpoint `json:"prediction_service"`
	RemediationService ServiceEndpoint `json:"remediation_service"`
	IncidentService    ServiceEndpoint `json:"incident_service"`
	ComplianceService  ServiceEndpoint `json:"compliance_service"`
	NotificationService ServiceEndpoint `json:"notification_service"`
	
	// External services
	Prometheus ServiceEndpoint `json:"prometheus"`
	Grafana    ServiceEndpoint `json:"grafana"`
	Kibana     ServiceEndpoint `json:"kibana"`
	MLflow     ServiceEndpoint `json:"mlflow"`
}

// ServiceEndpoint holds service endpoint configuration
type ServiceEndpoint struct {
	URL     string `json:"url"`
	Timeout int    `json:"timeout"`
	Retries int    `json:"retries"`
}

// SecurityConfig holds security configuration
type SecurityConfig struct {
	EncryptionKey    string `json:"encryption_key"`
	APIKeyHeader     string `json:"api_key_header"`
	TrustedProxies   []string `json:"trusted_proxies"`
	CSRFEnabled      bool   `json:"csrf_enabled"`
	SecurityHeaders  bool   `json:"security_headers"`
	ContentSecurityPolicy string `json:"content_security_policy"`
}

// SwaggerConfig holds Swagger configuration
type SwaggerConfig struct {
	Enabled bool   `json:"enabled"`
	Host    string `json:"host"`
	Path    string `json:"path"`
}

// MonitoringConfig holds monitoring configuration
type MonitoringConfig struct {
	MetricsEnabled bool   `json:"metrics_enabled"`
	TracingEnabled bool   `json:"tracing_enabled"`
	JaegerURL      string `json:"jaeger_url"`
	PrometheusURL  string `json:"prometheus_url"`
}

// AIConfig holds AI/ML configuration
type AIConfig struct {
	OpenAIAPIKey      string `json:"openai_api_key"`
	AnthropicAPIKey   string `json:"anthropic_api_key"`
	HuggingFaceToken  string `json:"huggingface_token"`
	DefaultModel      string `json:"default_model"`
	MaxTokens         int    `json:"max_tokens"`
	Temperature       float64 `json:"temperature"`
}

// FeatureFlags holds feature flag configuration
type FeatureFlags struct {
	PredictiveAnalysis    bool `json:"predictive_analysis"`
	AutoRemediation       bool `json:"auto_remediation"`
	ComplianceAutomation  bool `json:"compliance_automation"`
	MultiCloud            bool `json:"multi_cloud"`
	SelfLearning          bool `json:"self_learning"`
	AdvancedAnalytics     bool `json:"advanced_analytics"`
	AIInsights            bool `json:"ai_insights"`
	RealTimeCollaboration bool `json:"real_time_collaboration"`
}

// Load loads configuration from environment variables
func Load() (*Config, error) {
	config := &Config{
		Environment: getEnv("ENVIRONMENT", "development"),
		Port:        getEnv("PORT", "8080"),
		LogLevel:    getEnv("LOG_LEVEL", "info"),
		Debug:       getBoolEnv("DEBUG", false),

		Database: DatabaseConfig{
			URL:             getEnv("DATABASE_URL", ""),
			Host:            getEnv("POSTGRES_HOST", "localhost"),
			Port:            getIntEnv("POSTGRES_PORT", 5432),
			Name:            getEnv("POSTGRES_DB", "aiops_platform"),
			User:            getEnv("POSTGRES_USER", "aiops_user"),
			Password:        getEnv("POSTGRES_PASSWORD", "aiops_password"),
			SSLMode:         getEnv("POSTGRES_SSL_MODE", "disable"),
			MaxConnections:  getIntEnv("DB_MAX_CONNECTIONS", 25),
			IdleConnections: getIntEnv("DB_IDLE_CONNECTIONS", 5),
			ConnMaxLifetime: getDurationEnv("DB_CONN_MAX_LIFETIME", 5*time.Minute),
		},

		Redis: RedisConfig{
			URL:      getEnv("REDIS_URL", "redis://localhost:6379"),
			Host:     getEnv("REDIS_HOST", "localhost"),
			Port:     getIntEnv("REDIS_PORT", 6379),
			Password: getEnv("REDIS_PASSWORD", ""),
			DB:       getIntEnv("REDIS_DB", 0),
			PoolSize: getIntEnv("REDIS_POOL_SIZE", 10),
		},

		Kafka: KafkaConfig{
			Brokers:       getSliceEnv("KAFKA_BROKERS", []string{"localhost:9092"}),
			GroupID:       getEnv("KAFKA_GROUP_ID", "aiops-platform"),
			ClientID:      getEnv("KAFKA_CLIENT_ID", "api-gateway"),
			RetryMax:      getIntEnv("KAFKA_RETRY_MAX", 3),
			BatchSize:     getIntEnv("KAFKA_BATCH_SIZE", 100),
			FlushInterval: getIntEnv("KAFKA_FLUSH_INTERVAL", 1000),
		},

		JWT: JWTConfig{
			Secret:     getEnv("JWT_SECRET", "your-secret-key"),
			Expiration: getDurationEnv("JWT_EXPIRATION", 24*time.Hour),
			Issuer:     getEnv("JWT_ISSUER", "aiops-platform"),
			Algorithm:  getEnv("JWT_ALGORITHM", "HS256"),
		},

		CORS: CORSConfig{
			AllowedOrigins: getSliceEnv("CORS_ALLOWED_ORIGINS", []string{"*"}),
			AllowedMethods: getSliceEnv("CORS_ALLOWED_METHODS", []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"}),
			AllowedHeaders: getSliceEnv("CORS_ALLOWED_HEADERS", []string{"*"}),
		},

		RateLimit: RateLimitConfig{
			RequestsPerMinute: getIntEnv("RATE_LIMIT_REQUESTS_PER_MINUTE", 1000),
			Burst:             getIntEnv("RATE_LIMIT_BURST", 100),
			Window:            getDurationEnv("RATE_LIMIT_WINDOW", time.Minute),
		},

		Services: ServicesConfig{
			MonitoringService: ServiceEndpoint{
				URL:     getEnv("MONITORING_SERVICE_URL", "http://localhost:8081"),
				Timeout: getIntEnv("MONITORING_SERVICE_TIMEOUT", 30),
				Retries: getIntEnv("MONITORING_SERVICE_RETRIES", 3),
			},
			PredictionService: ServiceEndpoint{
				URL:     getEnv("PREDICTION_SERVICE_URL", "http://localhost:8082"),
				Timeout: getIntEnv("PREDICTION_SERVICE_TIMEOUT", 60),
				Retries: getIntEnv("PREDICTION_SERVICE_RETRIES", 3),
			},
			RemediationService: ServiceEndpoint{
				URL:     getEnv("REMEDIATION_SERVICE_URL", "http://localhost:8083"),
				Timeout: getIntEnv("REMEDIATION_SERVICE_TIMEOUT", 120),
				Retries: getIntEnv("REMEDIATION_SERVICE_RETRIES", 3),
			},
			IncidentService: ServiceEndpoint{
				URL:     getEnv("INCIDENT_SERVICE_URL", "http://localhost:8084"),
				Timeout: getIntEnv("INCIDENT_SERVICE_TIMEOUT", 30),
				Retries: getIntEnv("INCIDENT_SERVICE_RETRIES", 3),
			},
			ComplianceService: ServiceEndpoint{
				URL:     getEnv("COMPLIANCE_SERVICE_URL", "http://localhost:8085"),
				Timeout: getIntEnv("COMPLIANCE_SERVICE_TIMEOUT", 60),
				Retries: getIntEnv("COMPLIANCE_SERVICE_RETRIES", 3),
			},
			NotificationService: ServiceEndpoint{
				URL:     getEnv("NOTIFICATION_SERVICE_URL", "http://localhost:8086"),
				Timeout: getIntEnv("NOTIFICATION_SERVICE_TIMEOUT", 30),
				Retries: getIntEnv("NOTIFICATION_SERVICE_RETRIES", 3),
			},
			Prometheus: ServiceEndpoint{
				URL:     getEnv("PROMETHEUS_URL", "http://localhost:9090"),
				Timeout: getIntEnv("PROMETHEUS_TIMEOUT", 30),
				Retries: getIntEnv("PROMETHEUS_RETRIES", 3),
			},
			Grafana: ServiceEndpoint{
				URL:     getEnv("GRAFANA_URL", "http://localhost:3000"),
				Timeout: getIntEnv("GRAFANA_TIMEOUT", 30),
				Retries: getIntEnv("GRAFANA_RETRIES", 3),
			},
			Kibana: ServiceEndpoint{
				URL:     getEnv("KIBANA_URL", "http://localhost:5601"),
				Timeout: getIntEnv("KIBANA_TIMEOUT", 30),
				Retries: getIntEnv("KIBANA_RETRIES", 3),
			},
			MLflow: ServiceEndpoint{
				URL:     getEnv("MLFLOW_URL", "http://localhost:5000"),
				Timeout: getIntEnv("MLFLOW_TIMEOUT", 60),
				Retries: getIntEnv("MLFLOW_RETRIES", 3),
			},
		},

		Security: SecurityConfig{
			EncryptionKey:         getEnv("ENCRYPTION_KEY", "your-encryption-key"),
			APIKeyHeader:          getEnv("API_KEY_HEADER", "X-API-Key"),
			TrustedProxies:        getSliceEnv("TRUSTED_PROXIES", []string{"127.0.0.1"}),
			CSRFEnabled:           getBoolEnv("CSRF_ENABLED", true),
			SecurityHeaders:       getBoolEnv("SECURITY_HEADERS_ENABLED", true),
			ContentSecurityPolicy: getEnv("CONTENT_SECURITY_POLICY", "default-src 'self'"),
		},

		Swagger: SwaggerConfig{
			Enabled: getBoolEnv("SWAGGER_ENABLED", true),
			Host:    getEnv("SWAGGER_HOST", "localhost:8080"),
			Path:    getEnv("SWAGGER_PATH", "/swagger"),
		},

		Monitoring: MonitoringConfig{
			MetricsEnabled: getBoolEnv("METRICS_ENABLED", true),
			TracingEnabled: getBoolEnv("TRACING_ENABLED", true),
			JaegerURL:      getEnv("JAEGER_URL", "http://localhost:14268"),
			PrometheusURL:  getEnv("PROMETHEUS_URL", "http://localhost:9090"),
		},

		AI: AIConfig{
			OpenAIAPIKey:     getEnv("OPENAI_API_KEY", ""),
			AnthropicAPIKey:  getEnv("ANTHROPIC_API_KEY", ""),
			HuggingFaceToken: getEnv("HUGGINGFACE_API_TOKEN", ""),
			DefaultModel:     getEnv("AI_DEFAULT_MODEL", "gpt-4"),
			MaxTokens:        getIntEnv("AI_MAX_TOKENS", 4096),
			Temperature:      getFloatEnv("AI_TEMPERATURE", 0.7),
		},

		Features: FeatureFlags{
			PredictiveAnalysis:    getBoolEnv("FEATURE_PREDICTIVE_ANALYSIS", true),
			AutoRemediation:       getBoolEnv("FEATURE_AUTO_REMEDIATION", true),
			ComplianceAutomation:  getBoolEnv("FEATURE_COMPLIANCE_AUTOMATION", true),
			MultiCloud:            getBoolEnv("FEATURE_MULTI_CLOUD", true),
			SelfLearning:          getBoolEnv("FEATURE_SELF_LEARNING", true),
			AdvancedAnalytics:     getBoolEnv("FEATURE_ADVANCED_ANALYTICS", true),
			AIInsights:            getBoolEnv("FEATURE_AI_INSIGHTS", true),
			RealTimeCollaboration: getBoolEnv("FEATURE_REAL_TIME_COLLABORATION", true),
		},
	}

	// Validate configuration
	if err := config.validate(); err != nil {
		return nil, fmt.Errorf("config validation failed: %w", err)
	}

	return config, nil
}

// validate validates the configuration
func (c *Config) validate() error {
	if c.JWT.Secret == "" || c.JWT.Secret == "your-secret-key" {
		return fmt.Errorf("JWT secret must be set and not use default value")
	}

	if c.Database.URL == "" && (c.Database.Host == "" || c.Database.Name == "") {
		return fmt.Errorf("database configuration is incomplete")
	}

	if len(c.Kafka.Brokers) == 0 {
		return fmt.Errorf("at least one Kafka broker must be configured")
	}

	return nil
}

// Helper functions for environment variable parsing

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getIntEnv(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if intValue, err := strconv.Atoi(value); err == nil {
			return intValue
		}
	}
	return defaultValue
}

func getBoolEnv(key string, defaultValue bool) bool {
	if value := os.Getenv(key); value != "" {
		if boolValue, err := strconv.ParseBool(value); err == nil {
			return boolValue
		}
	}
	return defaultValue
}

func getFloatEnv(key string, defaultValue float64) float64 {
	if value := os.Getenv(key); value != "" {
		if floatValue, err := strconv.ParseFloat(value, 64); err == nil {
			return floatValue
		}
	}
	return defaultValue
}

func getDurationEnv(key string, defaultValue time.Duration) time.Duration {
	if value := os.Getenv(key); value != "" {
		if duration, err := time.ParseDuration(value); err == nil {
			return duration
		}
	}
	return defaultValue
}

func getSliceEnv(key string, defaultValue []string) []string {
	if value := os.Getenv(key); value != "" {
		return strings.Split(value, ",")
	}
	return defaultValue
} 