package config

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/spf13/viper"
)

// Config represents the complete configuration for the data pipeline service
type Config struct {
	Environment string         `mapstructure:"environment"`
	LogLevel    string         `mapstructure:"log_level"`
	Server      ServerConfig   `mapstructure:"server"`
	Database    DatabaseConfig `mapstructure:"database"`
	Redis       RedisConfig    `mapstructure:"redis"`
	Kafka       KafkaConfig    `mapstructure:"kafka"`
	Streams     StreamConfig   `mapstructure:"streams"`
	Pipeline    PipelineConfig `mapstructure:"pipeline"`
	Processors  ProcessorsConfig `mapstructure:"processors"`
	Monitoring  MonitoringConfig `mapstructure:"monitoring"`
	Security    SecurityConfig `mapstructure:"security"`
	Cloud       CloudConfig    `mapstructure:"cloud"`
	Storage     StorageConfig  `mapstructure:"storage"`
	Features    FeatureFlags   `mapstructure:"features"`
}

// ServerConfig contains HTTP server configuration
type ServerConfig struct {
	Port           int `mapstructure:"port"`
	ReadTimeout    int `mapstructure:"read_timeout"`
	WriteTimeout   int `mapstructure:"write_timeout"`
	IdleTimeout    int `mapstructure:"idle_timeout"`
	MaxHeaderBytes int `mapstructure:"max_header_bytes"`
}

// DatabaseConfig contains database connection settings
type DatabaseConfig struct {
	URL              string `mapstructure:"url"`
	MaxOpenConns     int    `mapstructure:"max_open_conns"`
	MaxIdleConns     int    `mapstructure:"max_idle_conns"`
	ConnMaxLifetime  int    `mapstructure:"conn_max_lifetime"`
	ConnMaxIdleTime  int    `mapstructure:"conn_max_idle_time"`
	EnableLogging    bool   `mapstructure:"enable_logging"`
	SlowQueryThreshold int  `mapstructure:"slow_query_threshold"`
}

// RedisConfig contains Redis connection settings
type RedisConfig struct {
	URL          string `mapstructure:"url"`
	Password     string `mapstructure:"password"`
	DB           int    `mapstructure:"db"`
	PoolSize     int    `mapstructure:"pool_size"`
	DialTimeout  int    `mapstructure:"dial_timeout"`
	ReadTimeout  int    `mapstructure:"read_timeout"`
	WriteTimeout int    `mapstructure:"write_timeout"`
	IdleTimeout  int    `mapstructure:"idle_timeout"`
	MaxRetries   int    `mapstructure:"max_retries"`
}

// KafkaConfig contains Kafka configuration
type KafkaConfig struct {
	Brokers           []string          `mapstructure:"brokers"`
	SecurityProtocol  string            `mapstructure:"security_protocol"`
	SASLMechanism     string            `mapstructure:"sasl_mechanism"`
	SASLUsername      string            `mapstructure:"sasl_username"`
	SASLPassword      string            `mapstructure:"sasl_password"`
	SSLCALocation     string            `mapstructure:"ssl_ca_location"`
	SSLCertLocation   string            `mapstructure:"ssl_cert_location"`
	SSLKeyLocation    string            `mapstructure:"ssl_key_location"`
	ConsumerGroups    map[string]string `mapstructure:"consumer_groups"`
	Topics            KafkaTopics       `mapstructure:"topics"`
	Producer          ProducerConfig    `mapstructure:"producer"`
	Consumer          ConsumerConfig    `mapstructure:"consumer"`
	SchemaRegistry    SchemaRegistryConfig `mapstructure:"schema_registry"`
}

// KafkaTopics defines topic configuration
type KafkaTopics struct {
	Metrics    string `mapstructure:"metrics"`
	Logs       string `mapstructure:"logs"`
	Events     string `mapstructure:"events"`
	Traces     string `mapstructure:"traces"`
	Alerts     string `mapstructure:"alerts"`
	Errors     string `mapstructure:"errors"`
	Audit      string `mapstructure:"audit"`
	DeadLetter string `mapstructure:"dead_letter"`
}

// ProducerConfig contains Kafka producer settings
type ProducerConfig struct {
	Acks             int    `mapstructure:"acks"`
	Retries          int    `mapstructure:"retries"`
	BatchSize        int    `mapstructure:"batch_size"`
	LingerMS         int    `mapstructure:"linger_ms"`
	BufferMemory     int    `mapstructure:"buffer_memory"`
	CompressionType  string `mapstructure:"compression_type"`
	MaxInFlightReqs  int    `mapstructure:"max_in_flight_requests"`
	RequestTimeoutMS int    `mapstructure:"request_timeout_ms"`
	EnableIdempotence bool  `mapstructure:"enable_idempotence"`
}

// ConsumerConfig contains Kafka consumer settings
type ConsumerConfig struct {
	GroupID              string `mapstructure:"group_id"`
	AutoOffsetReset      string `mapstructure:"auto_offset_reset"`
	EnableAutoCommit     bool   `mapstructure:"enable_auto_commit"`
	AutoCommitIntervalMS int    `mapstructure:"auto_commit_interval_ms"`
	SessionTimeoutMS     int    `mapstructure:"session_timeout_ms"`
	HeartbeatIntervalMS  int    `mapstructure:"heartbeat_interval_ms"`
	MaxPollRecords       int    `mapstructure:"max_poll_records"`
	MaxPollIntervalMS    int    `mapstructure:"max_poll_interval_ms"`
	FetchMinBytes        int    `mapstructure:"fetch_min_bytes"`
	FetchMaxWaitMS       int    `mapstructure:"fetch_max_wait_ms"`
}

// SchemaRegistryConfig contains schema registry settings
type SchemaRegistryConfig struct {
	URL      string `mapstructure:"url"`
	Username string `mapstructure:"username"`
	Password string `mapstructure:"password"`
	Enabled  bool   `mapstructure:"enabled"`
}

// StreamConfig contains stream processing configuration
type StreamConfig struct {
	BufferSize       int                  `mapstructure:"buffer_size"`
	Workers          int                  `mapstructure:"workers"`
	BatchSize        int                  `mapstructure:"batch_size"`
	FlushInterval    int                  `mapstructure:"flush_interval"`
	MaxRetries       int                  `mapstructure:"max_retries"`
	BackoffStrategy  string               `mapstructure:"backoff_strategy"`
	DeadLetterQueue  bool                 `mapstructure:"dead_letter_queue"`
	Windowing        WindowingConfig      `mapstructure:"windowing"`
	StateStore       StateStoreConfig     `mapstructure:"state_store"`
	Watermarking     WatermarkingConfig   `mapstructure:"watermarking"`
}

// WindowingConfig contains windowing settings
type WindowingConfig struct {
	Enabled        bool   `mapstructure:"enabled"`
	Type           string `mapstructure:"type"` // tumbling, sliding, session
	Size           int    `mapstructure:"size"`
	SlideInterval  int    `mapstructure:"slide_interval"`
	SessionTimeout int    `mapstructure:"session_timeout"`
	AllowLateness  int    `mapstructure:"allow_lateness"`
}

// StateStoreConfig contains state store settings
type StateStoreConfig struct {
	Type        string `mapstructure:"type"` // memory, rocksdb, redis
	Directory   string `mapstructure:"directory"`
	MaxSize     int    `mapstructure:"max_size"`
	TTL         int    `mapstructure:"ttl"`
	Compression bool   `mapstructure:"compression"`
}

// WatermarkingConfig contains watermarking settings
type WatermarkingConfig struct {
	Enabled       bool   `mapstructure:"enabled"`
	Strategy      string `mapstructure:"strategy"` // periodic, punctuated
	Interval      int    `mapstructure:"interval"`
	MaxOutOfOrder int    `mapstructure:"max_out_of_order"`
}

// PipelineConfig contains pipeline execution settings
type PipelineConfig struct {
	MaxConcurrency    int                `mapstructure:"max_concurrency"`
	Timeout           int                `mapstructure:"timeout"`
	RetryPolicy       RetryPolicyConfig  `mapstructure:"retry_policy"`
	CircuitBreaker    CircuitBreakerConfig `mapstructure:"circuit_breaker"`
	RateLimiting      RateLimitConfig    `mapstructure:"rate_limiting"`
	HealthCheck       HealthCheckConfig  `mapstructure:"health_check"`
	MetricsCollection MetricsConfig      `mapstructure:"metrics"`
}

// RetryPolicyConfig contains retry policy settings
type RetryPolicyConfig struct {
	MaxRetries      int    `mapstructure:"max_retries"`
	InitialDelay    int    `mapstructure:"initial_delay"`
	MaxDelay        int    `mapstructure:"max_delay"`
	BackoffFactor   float64 `mapstructure:"backoff_factor"`
	RetryableErrors []string `mapstructure:"retryable_errors"`
}

// CircuitBreakerConfig contains circuit breaker settings
type CircuitBreakerConfig struct {
	Enabled             bool   `mapstructure:"enabled"`
	FailureThreshold    int    `mapstructure:"failure_threshold"`
	SuccessThreshold    int    `mapstructure:"success_threshold"`
	Timeout             int    `mapstructure:"timeout"`
	HalfOpenMaxCalls    int    `mapstructure:"half_open_max_calls"`
	HalfOpenSuccessThreshold int `mapstructure:"half_open_success_threshold"`
}

// RateLimitConfig contains rate limiting settings
type RateLimitConfig struct {
	Enabled    bool   `mapstructure:"enabled"`
	Rate       int    `mapstructure:"rate"`
	Burst      int    `mapstructure:"burst"`
	WindowSize int    `mapstructure:"window_size"`
	Algorithm  string `mapstructure:"algorithm"` // token_bucket, leaky_bucket, sliding_window
}

// HealthCheckConfig contains health check settings
type HealthCheckConfig struct {
	Interval    int `mapstructure:"interval"`
	Timeout     int `mapstructure:"timeout"`
	Threshold   int `mapstructure:"threshold"`
	GracePeriod int `mapstructure:"grace_period"`
}

// MetricsConfig contains metrics collection settings
type MetricsConfig struct {
	Enabled        bool     `mapstructure:"enabled"`
	Interval       int      `mapstructure:"interval"`
	Tags           []string `mapstructure:"tags"`
	CustomMetrics  bool     `mapstructure:"custom_metrics"`
	HistogramBuckets []float64 `mapstructure:"histogram_buckets"`
}

// ProcessorsConfig contains processor-specific settings
type ProcessorsConfig struct {
	Metrics MetricProcessorConfig `mapstructure:"metrics"`
	Logs    LogProcessorConfig    `mapstructure:"logs"`
	Events  EventProcessorConfig  `mapstructure:"events"`
	Traces  TraceProcessorConfig  `mapstructure:"traces"`
	Alerts  AlertProcessorConfig  `mapstructure:"alerts"`
}

// MetricProcessorConfig contains metric processor settings
type MetricProcessorConfig struct {
	Enabled             bool     `mapstructure:"enabled"`
	AggregationWindow   int      `mapstructure:"aggregation_window"`
	SupportedTypes      []string `mapstructure:"supported_types"`
	MaxCardinality      int      `mapstructure:"max_cardinality"`
	SamplingRate        float64  `mapstructure:"sampling_rate"`
	CompressionEnabled  bool     `mapstructure:"compression_enabled"`
	DownsamplingRules   []DownsamplingRule `mapstructure:"downsampling_rules"`
}

// DownsamplingRule defines downsampling configuration
type DownsamplingRule struct {
	Resolution string `mapstructure:"resolution"`
	Retention  string `mapstructure:"retention"`
	Aggregator string `mapstructure:"aggregator"`
}

// LogProcessorConfig contains log processor settings
type LogProcessorConfig struct {
	Enabled         bool     `mapstructure:"enabled"`
	ParseJSON       bool     `mapstructure:"parse_json"`
	ExtractFields   []string `mapstructure:"extract_fields"`
	FilterPatterns  []string `mapstructure:"filter_patterns"`
	SensitiveFields []string `mapstructure:"sensitive_fields"`
	IndexFields     []string `mapstructure:"index_fields"`
	MaxLogSize      int      `mapstructure:"max_log_size"`
	CompressionType string   `mapstructure:"compression_type"`
}

// EventProcessorConfig contains event processor settings
type EventProcessorConfig struct {
	Enabled           bool                `mapstructure:"enabled"`
	SchemaValidation  bool                `mapstructure:"schema_validation"`
	Deduplication     bool                `mapstructure:"deduplication"`
	DeduplicationTTL  int                 `mapstructure:"deduplication_ttl"`
	Enrichment        EventEnrichmentConfig `mapstructure:"enrichment"`
	Ordering          EventOrderingConfig `mapstructure:"ordering"`
}

// EventEnrichmentConfig contains event enrichment settings
type EventEnrichmentConfig struct {
	Enabled     bool              `mapstructure:"enabled"`
	Sources     []string          `mapstructure:"sources"`
	Fields      map[string]string `mapstructure:"fields"`
	CacheSize   int               `mapstructure:"cache_size"`
	CacheTTL    int               `mapstructure:"cache_ttl"`
}

// EventOrderingConfig contains event ordering settings
type EventOrderingConfig struct {
	Enabled       bool   `mapstructure:"enabled"`
	BufferSize    int    `mapstructure:"buffer_size"`
	MaxLateness   int    `mapstructure:"max_lateness"`
	OrderingField string `mapstructure:"ordering_field"`
}

// TraceProcessorConfig contains trace processor settings
type TraceProcessorConfig struct {
	Enabled        bool     `mapstructure:"enabled"`
	SamplingRate   float64  `mapstructure:"sampling_rate"`
	MaxSpanSize    int      `mapstructure:"max_span_size"`
	SpanBatchSize  int      `mapstructure:"span_batch_size"`
	ExportTimeout  int      `mapstructure:"export_timeout"`
	TraceIDHeader  string   `mapstructure:"trace_id_header"`
	ServiceMapping map[string]string `mapstructure:"service_mapping"`
}

// AlertProcessorConfig contains alert processor settings
type AlertProcessorConfig struct {
	Enabled            bool              `mapstructure:"enabled"`
	EvaluationInterval int               `mapstructure:"evaluation_interval"`
	RuleEngine         RuleEngineConfig  `mapstructure:"rule_engine"`
	NotificationChannels []NotificationChannel `mapstructure:"notification_channels"`
	Suppressions       []SuppressionRule `mapstructure:"suppressions"`
}

// RuleEngineConfig contains rule engine settings
type RuleEngineConfig struct {
	Type           string `mapstructure:"type"` // simple, complex, ml
	MaxRules       int    `mapstructure:"max_rules"`
	EvalConcurrency int   `mapstructure:"eval_concurrency"`
	CacheEnabled   bool   `mapstructure:"cache_enabled"`
}

// NotificationChannel defines notification settings
type NotificationChannel struct {
	Type       string            `mapstructure:"type"`
	Enabled    bool              `mapstructure:"enabled"`
	Config     map[string]string `mapstructure:"config"`
	Priority   int               `mapstructure:"priority"`
	Filters    []string          `mapstructure:"filters"`
}

// SuppressionRule defines alert suppression
type SuppressionRule struct {
	Pattern   string `mapstructure:"pattern"`
	Duration  int    `mapstructure:"duration"`
	Condition string `mapstructure:"condition"`
}

// MonitoringConfig contains monitoring and observability settings
type MonitoringConfig struct {
	Prometheus PrometheusConfig `mapstructure:"prometheus"`
	Jaeger     JaegerConfig     `mapstructure:"jaeger"`
	DataDog    DataDogConfig    `mapstructure:"datadog"`
	NewRelic   NewRelicConfig   `mapstructure:"newrelic"`
	OpenTelemetry OpenTelemetryConfig `mapstructure:"opentelemetry"`
}

// PrometheusConfig contains Prometheus settings
type PrometheusConfig struct {
	Enabled      bool   `mapstructure:"enabled"`
	PushGateway  string `mapstructure:"push_gateway"`
	JobName      string `mapstructure:"job_name"`
	PushInterval int    `mapstructure:"push_interval"`
	Namespace    string `mapstructure:"namespace"`
}

// JaegerConfig contains Jaeger tracing settings
type JaegerConfig struct {
	Enabled        bool   `mapstructure:"enabled"`
	CollectorURL   string `mapstructure:"collector_url"`
	AgentEndpoint  string `mapstructure:"agent_endpoint"`
	ServiceName    string `mapstructure:"service_name"`
	SamplingRate   float64 `mapstructure:"sampling_rate"`
}

// DataDogConfig contains DataDog monitoring settings
type DataDogConfig struct {
	Enabled    bool   `mapstructure:"enabled"`
	APIKey     string `mapstructure:"api_key"`
	AppKey     string `mapstructure:"app_key"`
	AgentHost  string `mapstructure:"agent_host"`
	AgentPort  int    `mapstructure:"agent_port"`
	Namespace  string `mapstructure:"namespace"`
	Tags       []string `mapstructure:"tags"`
}

// NewRelicConfig contains New Relic monitoring settings
type NewRelicConfig struct {
	Enabled    bool   `mapstructure:"enabled"`
	LicenseKey string `mapstructure:"license_key"`
	AppName    string `mapstructure:"app_name"`
}

// OpenTelemetryConfig contains OpenTelemetry settings
type OpenTelemetryConfig struct {
	Enabled      bool   `mapstructure:"enabled"`
	Endpoint     string `mapstructure:"endpoint"`
	ServiceName  string `mapstructure:"service_name"`
	ResourceAttributes map[string]string `mapstructure:"resource_attributes"`
}

// SecurityConfig contains security settings
type SecurityConfig struct {
	JWTSecret          string            `mapstructure:"jwt_secret"`
	JWTExpiration      int               `mapstructure:"jwt_expiration"`
	APIKeys            map[string]string `mapstructure:"api_keys"`
	TLSEnabled         bool              `mapstructure:"tls_enabled"`
	TLSCertPath        string            `mapstructure:"tls_cert_path"`
	TLSKeyPath         string            `mapstructure:"tls_key_path"`
	RateLimiting       SecurityRateLimit `mapstructure:"rate_limiting"`
	Encryption         EncryptionConfig  `mapstructure:"encryption"`
	Authentication     AuthConfig        `mapstructure:"authentication"`
}

// SecurityRateLimit contains security rate limiting
type SecurityRateLimit struct {
	Enabled      bool              `mapstructure:"enabled"`
	Global       RateLimitRule     `mapstructure:"global"`
	PerEndpoint  map[string]RateLimitRule `mapstructure:"per_endpoint"`
	PerUser      RateLimitRule     `mapstructure:"per_user"`
}

// RateLimitRule defines rate limiting rules
type RateLimitRule struct {
	Requests int `mapstructure:"requests"`
	Window   int `mapstructure:"window"`
}

// EncryptionConfig contains encryption settings
type EncryptionConfig struct {
	Enabled     bool   `mapstructure:"enabled"`
	Algorithm   string `mapstructure:"algorithm"`
	KeyPath     string `mapstructure:"key_path"`
	KeyRotation int    `mapstructure:"key_rotation"`
}

// AuthConfig contains authentication settings
type AuthConfig struct {
	Providers []AuthProvider `mapstructure:"providers"`
	JWT       JWTConfig      `mapstructure:"jwt"`
	OAuth     OAuthConfig    `mapstructure:"oauth"`
}

// AuthProvider defines authentication providers
type AuthProvider struct {
	Name     string            `mapstructure:"name"`
	Type     string            `mapstructure:"type"`
	Enabled  bool              `mapstructure:"enabled"`
	Config   map[string]string `mapstructure:"config"`
}

// JWTConfig contains JWT settings
type JWTConfig struct {
	Algorithm      string `mapstructure:"algorithm"`
	PublicKeyPath  string `mapstructure:"public_key_path"`
	PrivateKeyPath string `mapstructure:"private_key_path"`
	Issuer         string `mapstructure:"issuer"`
	Audience       string `mapstructure:"audience"`
}

// OAuthConfig contains OAuth settings
type OAuthConfig struct {
	Enabled      bool   `mapstructure:"enabled"`
	ClientID     string `mapstructure:"client_id"`
	ClientSecret string `mapstructure:"client_secret"`
	RedirectURL  string `mapstructure:"redirect_url"`
	Scopes       []string `mapstructure:"scopes"`
}

// CloudConfig contains cloud provider settings
type CloudConfig struct {
	AWS   AWSConfig   `mapstructure:"aws"`
	Azure AzureConfig `mapstructure:"azure"`
	GCP   GCPConfig   `mapstructure:"gcp"`
}

// AWSConfig contains AWS-specific settings
type AWSConfig struct {
	Region          string            `mapstructure:"region"`
	AccessKeyID     string            `mapstructure:"access_key_id"`
	SecretAccessKey string            `mapstructure:"secret_access_key"`
	SessionToken    string            `mapstructure:"session_token"`
	Kinesis         KinesisConfig     `mapstructure:"kinesis"`
	S3              S3Config          `mapstructure:"s3"`
	SQS             SQSConfig         `mapstructure:"sqs"`
	SNS             SNSConfig         `mapstructure:"sns"`
	Tags            map[string]string `mapstructure:"tags"`
}

// KinesisConfig contains AWS Kinesis settings
type KinesisConfig struct {
	Enabled     bool   `mapstructure:"enabled"`
	StreamName  string `mapstructure:"stream_name"`
	ShardCount  int    `mapstructure:"shard_count"`
	RetentionHours int `mapstructure:"retention_hours"`
}

// S3Config contains AWS S3 settings
type S3Config struct {
	Enabled    bool   `mapstructure:"enabled"`
	BucketName string `mapstructure:"bucket_name"`
	Prefix     string `mapstructure:"prefix"`
	StorageClass string `mapstructure:"storage_class"`
}

// SQSConfig contains AWS SQS settings
type SQSConfig struct {
	Enabled          bool `mapstructure:"enabled"`
	QueueURL         string `mapstructure:"queue_url"`
	VisibilityTimeout int `mapstructure:"visibility_timeout"`
	MaxMessages      int `mapstructure:"max_messages"`
}

// SNSConfig contains AWS SNS settings
type SNSConfig struct {
	Enabled  bool   `mapstructure:"enabled"`
	TopicARN string `mapstructure:"topic_arn"`
}

// AzureConfig contains Azure-specific settings
type AzureConfig struct {
	TenantID       string              `mapstructure:"tenant_id"`
	ClientID       string              `mapstructure:"client_id"`
	ClientSecret   string              `mapstructure:"client_secret"`
	SubscriptionID string              `mapstructure:"subscription_id"`
	EventHubs      EventHubsConfig     `mapstructure:"event_hubs"`
	BlobStorage    BlobStorageConfig   `mapstructure:"blob_storage"`
	Tags           map[string]string   `mapstructure:"tags"`
}

// EventHubsConfig contains Azure Event Hubs settings
type EventHubsConfig struct {
	Enabled         bool   `mapstructure:"enabled"`
	ConnectionString string `mapstructure:"connection_string"`
	HubName         string `mapstructure:"hub_name"`
	ConsumerGroup   string `mapstructure:"consumer_group"`
}

// BlobStorageConfig contains Azure Blob Storage settings
type BlobStorageConfig struct {
	Enabled         bool   `mapstructure:"enabled"`
	ConnectionString string `mapstructure:"connection_string"`
	ContainerName   string `mapstructure:"container_name"`
}

// GCPConfig contains GCP-specific settings
type GCPConfig struct {
	ProjectID       string              `mapstructure:"project_id"`
	CredentialsPath string              `mapstructure:"credentials_path"`
	PubSub          PubSubConfig        `mapstructure:"pubsub"`
	BigQuery        BigQueryConfig      `mapstructure:"bigquery"`
	CloudStorage    CloudStorageConfig  `mapstructure:"cloud_storage"`
	Labels          map[string]string   `mapstructure:"labels"`
}

// PubSubConfig contains GCP Pub/Sub settings
type PubSubConfig struct {
	Enabled      bool   `mapstructure:"enabled"`
	TopicName    string `mapstructure:"topic_name"`
	Subscription string `mapstructure:"subscription"`
}

// BigQueryConfig contains GCP BigQuery settings
type BigQueryConfig struct {
	Enabled   bool   `mapstructure:"enabled"`
	Dataset   string `mapstructure:"dataset"`
	Table     string `mapstructure:"table"`
	Location  string `mapstructure:"location"`
}

// CloudStorageConfig contains GCP Cloud Storage settings
type CloudStorageConfig struct {
	Enabled    bool   `mapstructure:"enabled"`
	BucketName string `mapstructure:"bucket_name"`
	Prefix     string `mapstructure:"prefix"`
}

// StorageConfig contains storage backend settings
type StorageConfig struct {
	ClickHouse  ClickHouseConfig  `mapstructure:"clickhouse"`
	InfluxDB    InfluxDBConfig    `mapstructure:"influxdb"`
	Elasticsearch ElasticsearchConfig `mapstructure:"elasticsearch"`
	TimeSeries  TimeSeriesConfig  `mapstructure:"timeseries"`
	ObjectStore ObjectStoreConfig `mapstructure:"object_store"`
}

// ClickHouseConfig contains ClickHouse settings
type ClickHouseConfig struct {
	Enabled     bool   `mapstructure:"enabled"`
	DSN         string `mapstructure:"dsn"`
	Database    string `mapstructure:"database"`
	MaxOpenConns int   `mapstructure:"max_open_conns"`
	MaxIdleConns int   `mapstructure:"max_idle_conns"`
	Compression bool   `mapstructure:"compression"`
}

// InfluxDBConfig contains InfluxDB settings
type InfluxDBConfig struct {
	Enabled     bool   `mapstructure:"enabled"`
	URL         string `mapstructure:"url"`
	Token       string `mapstructure:"token"`
	Org         string `mapstructure:"org"`
	Bucket      string `mapstructure:"bucket"`
	BatchSize   int    `mapstructure:"batch_size"`
	FlushInterval int  `mapstructure:"flush_interval"`
}

// ElasticsearchConfig contains Elasticsearch settings
type ElasticsearchConfig struct {
	Enabled     bool     `mapstructure:"enabled"`
	URLs        []string `mapstructure:"urls"`
	Username    string   `mapstructure:"username"`
	Password    string   `mapstructure:"password"`
	IndexPrefix string   `mapstructure:"index_prefix"`
	MaxRetries  int      `mapstructure:"max_retries"`
}

// TimeSeriesConfig contains time series storage settings
type TimeSeriesConfig struct {
	DefaultRetention   string `mapstructure:"default_retention"`
	CompressionEnabled bool   `mapstructure:"compression_enabled"`
	ShardingStrategy   string `mapstructure:"sharding_strategy"`
	ReplicationFactor  int    `mapstructure:"replication_factor"`
}

// ObjectStoreConfig contains object storage settings
type ObjectStoreConfig struct {
	Provider     string `mapstructure:"provider"`
	Endpoint     string `mapstructure:"endpoint"`
	AccessKey    string `mapstructure:"access_key"`
	SecretKey    string `mapstructure:"secret_key"`
	BucketName   string `mapstructure:"bucket_name"`
	Region       string `mapstructure:"region"`
	UseSSL       bool   `mapstructure:"use_ssl"`
}

// FeatureFlags contains feature toggle settings
type FeatureFlags struct {
	AdvancedAnalytics   bool `mapstructure:"advanced_analytics"`
	MLPredictions      bool `mapstructure:"ml_predictions"`
	RealTimeAlerts     bool `mapstructure:"realtime_alerts"`
	SchemaEvolution    bool `mapstructure:"schema_evolution"`
	AutoScaling        bool `mapstructure:"auto_scaling"`
	CompressionOptimization bool `mapstructure:"compression_optimization"`
	SmartRouting       bool `mapstructure:"smart_routing"`
	DataLineage        bool `mapstructure:"data_lineage"`
	CostOptimization   bool `mapstructure:"cost_optimization"`
	ExperimentalFeatures bool `mapstructure:"experimental_features"`
}

// Load reads configuration from environment variables and config files
func Load() (*Config, error) {
	config := &Config{}

	// Set default values
	setDefaults()

	// Read from config file
	viper.SetConfigName("config")
	viper.SetConfigType("yaml")
	viper.AddConfigPath(".")
	viper.AddConfigPath("./config")
	viper.AddConfigPath("/etc/data-pipeline")

	// Allow config file to be optional
	if err := viper.ReadInConfig(); err != nil {
		if _, ok := err.(viper.ConfigFileNotFoundError); !ok {
			return nil, fmt.Errorf("failed to read config file: %w", err)
		}
	}

	// Override with environment variables
	viper.AutomaticEnv()
	viper.SetEnvKeyReplacer(strings.NewReplacer(".", "_"))

	// Unmarshal into config struct
	if err := viper.Unmarshal(config); err != nil {
		return nil, fmt.Errorf("failed to unmarshal config: %w", err)
	}

	// Override specific fields from environment
	overrideFromEnv(config)

	// Validate configuration
	if err := validate(config); err != nil {
		return nil, fmt.Errorf("config validation failed: %w", err)
	}

	return config, nil
}

// setDefaults sets default configuration values
func setDefaults() {
	// Server defaults
	viper.SetDefault("server.port", 8080)
	viper.SetDefault("server.read_timeout", 30)
	viper.SetDefault("server.write_timeout", 30)
	viper.SetDefault("server.idle_timeout", 60)
	viper.SetDefault("server.max_header_bytes", 1048576)

	// Database defaults
	viper.SetDefault("database.max_open_conns", 25)
	viper.SetDefault("database.max_idle_conns", 5)
	viper.SetDefault("database.conn_max_lifetime", 300)
	viper.SetDefault("database.conn_max_idle_time", 60)
	viper.SetDefault("database.enable_logging", false)
	viper.SetDefault("database.slow_query_threshold", 1000)

	// Redis defaults
	viper.SetDefault("redis.db", 0)
	viper.SetDefault("redis.pool_size", 10)
	viper.SetDefault("redis.dial_timeout", 5)
	viper.SetDefault("redis.read_timeout", 3)
	viper.SetDefault("redis.write_timeout", 3)
	viper.SetDefault("redis.idle_timeout", 60)
	viper.SetDefault("redis.max_retries", 3)

	// Kafka defaults
	viper.SetDefault("kafka.brokers", []string{"localhost:9092"})
	viper.SetDefault("kafka.security_protocol", "PLAINTEXT")
	viper.SetDefault("kafka.consumer_groups.default", "data-pipeline")
	viper.SetDefault("kafka.topics.metrics", "metrics")
	viper.SetDefault("kafka.topics.logs", "logs")
	viper.SetDefault("kafka.topics.events", "events")
	viper.SetDefault("kafka.topics.traces", "traces")
	viper.SetDefault("kafka.topics.alerts", "alerts")
	viper.SetDefault("kafka.topics.errors", "errors")
	viper.SetDefault("kafka.topics.audit", "audit")
	viper.SetDefault("kafka.topics.dead_letter", "dead-letter")

	// Producer defaults
	viper.SetDefault("kafka.producer.acks", 1)
	viper.SetDefault("kafka.producer.retries", 3)
	viper.SetDefault("kafka.producer.batch_size", 16384)
	viper.SetDefault("kafka.producer.linger_ms", 5)
	viper.SetDefault("kafka.producer.buffer_memory", 33554432)
	viper.SetDefault("kafka.producer.compression_type", "snappy")
	viper.SetDefault("kafka.producer.enable_idempotence", true)

	// Consumer defaults
	viper.SetDefault("kafka.consumer.group_id", "data-pipeline-consumer")
	viper.SetDefault("kafka.consumer.auto_offset_reset", "latest")
	viper.SetDefault("kafka.consumer.enable_auto_commit", true)
	viper.SetDefault("kafka.consumer.auto_commit_interval_ms", 5000)
	viper.SetDefault("kafka.consumer.session_timeout_ms", 30000)
	viper.SetDefault("kafka.consumer.heartbeat_interval_ms", 3000)
	viper.SetDefault("kafka.consumer.max_poll_records", 500)

	// Stream processing defaults
	viper.SetDefault("streams.buffer_size", 1000)
	viper.SetDefault("streams.workers", 4)
	viper.SetDefault("streams.batch_size", 100)
	viper.SetDefault("streams.flush_interval", 5)
	viper.SetDefault("streams.max_retries", 3)
	viper.SetDefault("streams.backoff_strategy", "exponential")
	viper.SetDefault("streams.dead_letter_queue", true)

	// Pipeline defaults
	viper.SetDefault("pipeline.max_concurrency", 10)
	viper.SetDefault("pipeline.timeout", 30)
	viper.SetDefault("pipeline.retry_policy.max_retries", 3)
	viper.SetDefault("pipeline.retry_policy.initial_delay", 1)
	viper.SetDefault("pipeline.retry_policy.max_delay", 30)
	viper.SetDefault("pipeline.retry_policy.backoff_factor", 2.0)

	// Feature flags defaults
	viper.SetDefault("features.advanced_analytics", true)
	viper.SetDefault("features.ml_predictions", true)
	viper.SetDefault("features.realtime_alerts", true)
	viper.SetDefault("features.schema_evolution", true)
	viper.SetDefault("features.auto_scaling", false)
	viper.SetDefault("features.experimental_features", false)

	// Environment defaults
	viper.SetDefault("environment", "development")
	viper.SetDefault("log_level", "info")
}

// overrideFromEnv overrides specific config fields from environment variables
func overrideFromEnv(config *Config) {
	if val := os.Getenv("ENVIRONMENT"); val != "" {
		config.Environment = val
	}
	if val := os.Getenv("LOG_LEVEL"); val != "" {
		config.LogLevel = val
	}
	if val := os.Getenv("SERVER_PORT"); val != "" {
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
	if val := os.Getenv("KAFKA_BROKERS"); val != "" {
		config.Kafka.Brokers = strings.Split(val, ",")
	}
	if val := os.Getenv("JWT_SECRET"); val != "" {
		config.Security.JWTSecret = val
	}
}

// validate performs configuration validation
func validate(config *Config) error {
	if config.Database.URL == "" {
		return fmt.Errorf("database URL is required")
	}
	if config.Redis.URL == "" {
		return fmt.Errorf("redis URL is required")
	}
	if len(config.Kafka.Brokers) == 0 {
		return fmt.Errorf("at least one Kafka broker is required")
	}
	if config.Security.JWTSecret == "" {
		return fmt.Errorf("JWT secret is required")
	}
	if config.Server.Port <= 0 || config.Server.Port > 65535 {
		return fmt.Errorf("invalid server port: %d", config.Server.Port)
	}
	return nil
}

// GetKafkaBrokerList returns comma-separated list of Kafka brokers
func (c *Config) GetKafkaBrokerList() string {
	return strings.Join(c.Kafka.Brokers, ",")
}

// IsProduction returns true if running in production environment
func (c *Config) IsProduction() bool {
	return strings.ToLower(c.Environment) == "production"
}

// IsDevelopment returns true if running in development environment
func (c *Config) IsDevelopment() bool {
	return strings.ToLower(c.Environment) == "development"
}

// GetDatabaseTimeout returns database timeout duration
func (c *Config) GetDatabaseTimeout() time.Duration {
	return time.Duration(c.Database.ConnMaxLifetime) * time.Second
}

// GetRedisTimeout returns Redis timeout duration
func (c *Config) GetRedisTimeout() time.Duration {
	return time.Duration(c.Redis.DialTimeout) * time.Second
} 