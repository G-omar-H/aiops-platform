"""
Configuration Management for AI Prediction Service

Handles all environment variables, settings, and configuration validation.
"""

import os
from functools import lru_cache
from typing import List, Optional, Dict, Any

from pydantic import BaseSettings, validator, Field
from pydantic_settings import BaseSettings as PydanticBaseSettings


class Settings(PydanticBaseSettings):
    """Application settings with validation and type safety."""
    
    # Application settings
    APP_NAME: str = "AIOps AI Prediction Service"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    WORKERS: int = Field(default=4, env="WORKERS")
    
    # Security settings
    SECRET_KEY: str = Field(..., env="SECRET_KEY")
    ALLOWED_HOSTS: List[str] = Field(default=["*"], env="ALLOWED_HOSTS")
    CORS_ORIGINS: List[str] = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:8080", 
            "https://dashboard.aiops.local"
        ],
        env="CORS_ORIGINS"
    )
    
    # Database settings
    POSTGRES_HOST: str = Field(default="postgres", env="POSTGRES_HOST")
    POSTGRES_PORT: int = Field(default=5432, env="POSTGRES_PORT")
    POSTGRES_USER: str = Field(default="aiops", env="POSTGRES_USER")
    POSTGRES_PASSWORD: str = Field(..., env="POSTGRES_PASSWORD")
    POSTGRES_DB: str = Field(default="aiops_prediction", env="POSTGRES_DB")
    DATABASE_URL: Optional[str] = Field(default=None, env="DATABASE_URL")
    
    # Redis settings
    REDIS_HOST: str = Field(default="redis", env="REDIS_HOST")
    REDIS_PORT: int = Field(default=6379, env="REDIS_PORT")
    REDIS_PASSWORD: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    REDIS_DB: int = Field(default=0, env="REDIS_DB")
    REDIS_URL: Optional[str] = Field(default=None, env="REDIS_URL")
    
    # Machine Learning settings
    ML_MODELS_PATH: str = Field(default="./models", env="ML_MODELS_PATH")
    ML_MODEL_CACHE_SIZE: int = Field(default=10, env="ML_MODEL_CACHE_SIZE")
    ML_PREDICTION_BATCH_SIZE: int = Field(default=100, env="ML_PREDICTION_BATCH_SIZE")
    ML_RETRAIN_INTERVAL_HOURS: int = Field(default=24, env="ML_RETRAIN_INTERVAL_HOURS")
    ML_DEFAULT_LOOKBACK_MINUTES: int = Field(default=60, env="ML_DEFAULT_LOOKBACK_MINUTES")
    
    # Model specific settings
    LSTM_SEQUENCE_LENGTH: int = Field(default=50, env="LSTM_SEQUENCE_LENGTH")
    LSTM_EPOCHS: int = Field(default=100, env="LSTM_EPOCHS")
    LSTM_BATCH_SIZE: int = Field(default=32, env="LSTM_BATCH_SIZE")
    RF_N_ESTIMATORS: int = Field(default=100, env="RF_N_ESTIMATORS")
    RF_MAX_DEPTH: int = Field(default=10, env="RF_MAX_DEPTH")
    
    # Metrics and monitoring
    METRICS_RETENTION_DAYS: int = Field(default=30, env="METRICS_RETENTION_DAYS")
    METRICS_COLLECTION_INTERVAL: int = Field(default=30, env="METRICS_COLLECTION_INTERVAL")
    PROMETHEUS_URL: str = Field(default="http://prometheus:9090", env="PROMETHEUS_URL")
    GRAFANA_URL: str = Field(default="http://grafana:3000", env="GRAFANA_URL")
    
    # Kafka settings (for real-time data streaming)
    KAFKA_BOOTSTRAP_SERVERS: str = Field(default="kafka:9092", env="KAFKA_BOOTSTRAP_SERVERS")
    KAFKA_TOPIC_METRICS: str = Field(default="system-metrics", env="KAFKA_TOPIC_METRICS")
    KAFKA_TOPIC_PREDICTIONS: str = Field(default="predictions", env="KAFKA_TOPIC_PREDICTIONS")
    KAFKA_GROUP_ID: str = Field(default="ai-prediction-service", env="KAFKA_GROUP_ID")
    
    # Logging settings
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = Field(default="json", env="LOG_FORMAT")
    LOG_FILE: Optional[str] = Field(default=None, env="LOG_FILE")
    
    # API Gateway integration
    API_GATEWAY_URL: str = Field(default="http://api-gateway:8080", env="API_GATEWAY_URL")
    API_GATEWAY_TOKEN: str = Field(..., env="API_GATEWAY_TOKEN")
    
    # OpenAI/External AI settings
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    OPENAI_MODEL: str = Field(default="gpt-4", env="OPENAI_MODEL")
    
    # Performance tuning
    MAX_CONCURRENT_PREDICTIONS: int = Field(default=50, env="MAX_CONCURRENT_PREDICTIONS")
    PREDICTION_TIMEOUT_SECONDS: int = Field(default=300, env="PREDICTION_TIMEOUT_SECONDS")
    CACHE_TTL_SECONDS: int = Field(default=300, env="CACHE_TTL_SECONDS")
    
    # Feature flags
    ENABLE_REAL_TIME_PREDICTION: bool = Field(default=True, env="ENABLE_REAL_TIME_PREDICTION")
    ENABLE_BATCH_PREDICTION: bool = Field(default=True, env="ENABLE_BATCH_PREDICTION")
    ENABLE_MODEL_RETRAINING: bool = Field(default=True, env="ENABLE_MODEL_RETRAINING")
    ENABLE_ANOMALY_DETECTION: bool = Field(default=True, env="ENABLE_ANOMALY_DETECTION")
    
    @validator("DATABASE_URL", pre=True)
    def build_database_url(cls, v: Optional[str], values: Dict[str, Any]) -> str:
        """Build database URL from components if not provided."""
        if isinstance(v, str):
            return v
        return (
            f"postgresql+asyncpg://{values.get('POSTGRES_USER')}:"
            f"{values.get('POSTGRES_PASSWORD')}@{values.get('POSTGRES_HOST')}:"
            f"{values.get('POSTGRES_PORT')}/{values.get('POSTGRES_DB')}"
        )
    
    @validator("REDIS_URL", pre=True)
    def build_redis_url(cls, v: Optional[str], values: Dict[str, Any]) -> str:
        """Build Redis URL from components if not provided."""
        if isinstance(v, str):
            return v
        
        password_part = f":{values.get('REDIS_PASSWORD')}@" if values.get('REDIS_PASSWORD') else ""
        return (
            f"redis://{password_part}{values.get('REDIS_HOST')}:"
            f"{values.get('REDIS_PORT')}/{values.get('REDIS_DB')}"
        )
    
    @validator("ALLOWED_HOSTS", pre=True)
    def parse_allowed_hosts(cls, v):
        """Parse allowed hosts from string or list."""
        if isinstance(v, str):
            return [host.strip() for host in v.split(",")]
        return v
    
    @validator("CORS_ORIGINS", pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return v.upper()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        

class DevelopmentSettings(Settings):
    """Development environment settings."""
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    

class ProductionSettings(Settings):
    """Production environment settings."""
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    

class TestingSettings(Settings):
    """Testing environment settings."""
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    POSTGRES_DB: str = "aiops_prediction_test"
    REDIS_DB: int = 15  # Use separate Redis DB for testing


@lru_cache()
def get_settings() -> Settings:
    """Get application settings based on environment."""
    environment = os.getenv("ENVIRONMENT", "development").lower()
    
    if environment == "production":
        return ProductionSettings()
    elif environment == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()


# Global settings instance
settings = get_settings() 