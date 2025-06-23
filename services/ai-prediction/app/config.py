"""
Configuration management for AIOps AI Prediction Service
Handles environment variables, model parameters, and service settings
"""

import os
from functools import lru_cache
from typing import List, Dict, Any, Optional
from pydantic import BaseSettings, Field, validator


class MLModelConfig(BaseSettings):
    """Machine Learning model configuration"""
    
    # Model storage and paths
    model_store_path: str = Field(default="./models", env="ML_MODEL_STORE_PATH")
    pretrained_models_path: str = Field(default="./pretrained", env="ML_PRETRAINED_MODELS_PATH")
    
    # Model types and algorithms
    supported_algorithms: List[str] = [
        "lstm", "transformer", "random_forest", "isolation_forest",
        "autoencoder", "svm", "xgboost", "prophet"
    ]
    
    # Default model parameters
    default_sequence_length: int = Field(default=60, env="ML_SEQUENCE_LENGTH")
    default_prediction_horizon: int = Field(default=12, env="ML_PREDICTION_HORIZON")
    default_confidence_threshold: float = Field(default=0.85, env="ML_CONFIDENCE_THRESHOLD")
    
    # Training configuration
    batch_size: int = Field(default=64, env="ML_BATCH_SIZE")
    epochs: int = Field(default=100, env="ML_EPOCHS")
    learning_rate: float = Field(default=0.001, env="ML_LEARNING_RATE")
    validation_split: float = Field(default=0.2, env="ML_VALIDATION_SPLIT")
    
    # Model performance thresholds
    min_accuracy: float = Field(default=0.8, env="ML_MIN_ACCURACY")
    max_training_time: int = Field(default=3600, env="ML_MAX_TRAINING_TIME")  # seconds
    
    # Auto-retraining configuration
    auto_retrain_enabled: bool = Field(default=True, env="ML_AUTO_RETRAIN_ENABLED")
    retrain_threshold_accuracy: float = Field(default=0.75, env="ML_RETRAIN_THRESHOLD")
    retrain_interval_hours: int = Field(default=24, env="ML_RETRAIN_INTERVAL_HOURS")
    
    class Config:
        env_prefix = "ML_"


class DatabaseConfig(BaseSettings):
    """Database configuration"""
    
    # Primary database
    database_url: str = Field(env="DATABASE_URL")
    pool_size: int = Field(default=20, env="DB_POOL_SIZE")
    max_overflow: int = Field(default=30, env="DB_MAX_OVERFLOW")
    pool_timeout: int = Field(default=30, env="DB_POOL_TIMEOUT")
    
    # Time series database (InfluxDB)
    influxdb_url: str = Field(default="http://influxdb:8086", env="INFLUXDB_URL")
    influxdb_token: str = Field(env="INFLUXDB_TOKEN")
    influxdb_org: str = Field(default="aiops", env="INFLUXDB_ORG")
    influxdb_bucket: str = Field(default="metrics", env="INFLUXDB_BUCKET")
    
    # Redis for caching and session management
    redis_url: str = Field(default="redis://redis:6379/0", env="REDIS_URL")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")  # seconds
    
    class Config:
        env_prefix = "DB_"


class SecurityConfig(BaseSettings):
    """Security and authentication configuration"""
    
    # JWT configuration
    secret_key: str = Field(env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # API security
    api_key_header: str = Field(default="X-API-Key", env="API_KEY_HEADER")
    allowed_hosts: List[str] = Field(default=["*"], env="ALLOWED_HOSTS")
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    
    # Rate limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")  # seconds
    
    # Encryption
    encryption_key: Optional[str] = Field(default=None, env="ENCRYPTION_KEY")
    
    @validator('secret_key')
    def secret_key_must_be_set(cls, v):
        if not v:
            raise ValueError('SECRET_KEY must be set')
        return v
    
    class Config:
        env_prefix = "SECURITY_"


class MonitoringConfig(BaseSettings):
    """Monitoring and observability configuration"""
    
    # Prometheus metrics
    metrics_enabled: bool = Field(default=True, env="METRICS_ENABLED")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")
    
    # Tracing
    jaeger_enabled: bool = Field(default=False, env="JAEGER_ENABLED")
    jaeger_endpoint: Optional[str] = Field(default=None, env="JAEGER_ENDPOINT")
    
    # Health checks
    health_check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")
    
    class Config:
        env_prefix = "MONITORING_"


class ExternalIntegrationConfig(BaseSettings):
    """External service integration configuration"""
    
    # API Gateway
    api_gateway_url: str = Field(default="http://api-gateway:8080", env="API_GATEWAY_URL")
    api_gateway_timeout: int = Field(default=30, env="API_GATEWAY_TIMEOUT")
    
    # Monitoring service
    monitoring_service_url: str = Field(default="http://monitoring:8080", env="MONITORING_SERVICE_URL")
    monitoring_service_timeout: int = Field(default=30, env="MONITORING_SERVICE_TIMEOUT")
    
    # Auto-remediation service
    auto_remediation_url: str = Field(default="http://auto-remediation:8080", env="AUTO_REMEDIATION_URL")
    auto_remediation_enabled: bool = Field(default=True, env="AUTO_REMEDIATION_ENABLED")
    
    # Notification service
    notification_service_url: str = Field(default="http://notification:8080", env="NOTIFICATION_SERVICE_URL")
    
    # External ML services
    huggingface_api_key: Optional[str] = Field(default=None, env="HUGGINGFACE_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    
    class Config:
        env_prefix = "INTEGRATION_"


class Settings(BaseSettings):
    """Main application settings"""
    
    # Application configuration
    app_name: str = Field(default="AIOps AI Prediction Service", env="APP_NAME")
    version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # Server configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")
    
    # Sub-configurations
    ml: MLModelConfig = MLModelConfig()
    database: DatabaseConfig = DatabaseConfig()
    security: SecurityConfig = SecurityConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    integrations: ExternalIntegrationConfig = ExternalIntegrationConfig()
    
    # Feature flags
    feature_flags: Dict[str, bool] = {
        "advanced_anomaly_detection": True,
        "auto_model_selection": True,
        "real_time_predictions": True,
        "batch_processing": True,
        "model_explainability": True,
        "federated_learning": False,
        "quantum_ml": False,
    }
    
    # Performance settings
    max_concurrent_predictions: int = Field(default=100, env="MAX_CONCURRENT_PREDICTIONS")
    prediction_timeout: int = Field(default=30, env="PREDICTION_TIMEOUT")
    model_cache_size: int = Field(default=10, env="MODEL_CACHE_SIZE")
    
    # Data processing
    max_batch_size: int = Field(default=1000, env="MAX_BATCH_SIZE")
    data_retention_days: int = Field(default=90, env="DATA_RETENTION_DAYS")
    
    @validator('environment')
    def validate_environment(cls, v):
        allowed = ['development', 'staging', 'production']
        if v not in allowed:
            raise ValueError(f'Environment must be one of {allowed}')
        return v
    
    @validator('workers')
    def validate_workers(cls, v):
        if v < 1:
            raise ValueError('Workers must be at least 1')
        return v
    
    @property
    def METRICS_PORT(self) -> int:
        return self.monitoring.metrics_port
    
    @property
    def PORT(self) -> int:
        return self.port
    
    @property
    def DEBUG(self) -> bool:
        return self.debug
    
    @property
    def is_production(self) -> bool:
        return self.environment == "production"
    
    @property
    def database_url(self) -> str:
        return self.database.database_url
    
    @property
    def redis_url(self) -> str:
        return self.database.redis_url
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Environment-specific configurations
def get_development_settings() -> Settings:
    """Development environment settings"""
    settings = get_settings()
    settings.debug = True
    settings.ml.auto_retrain_enabled = False
    settings.monitoring.log_level = "DEBUG"
    return settings


def get_production_settings() -> Settings:
    """Production environment settings"""
    settings = get_settings()
    settings.debug = False
    settings.workers = max(4, os.cpu_count() or 1)
    settings.monitoring.log_level = "INFO"
    settings.ml.auto_retrain_enabled = True
    return settings


def get_test_settings() -> Settings:
    """Test environment settings"""
    settings = get_settings()
    settings.debug = True
    settings.database.database_url = "sqlite:///./test.db"
    settings.database.redis_url = "redis://localhost:6379/1"
    settings.ml.auto_retrain_enabled = False
    return settings 