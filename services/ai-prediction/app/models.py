"""
Pydantic models for AIOps AI Prediction Service
Defines request/response schemas, data models, and validation
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
import uuid


# Enums
class PredictionType(str, Enum):
    """Types of predictions supported"""
    FAILURE_PREDICTION = "failure_prediction"
    CAPACITY_PLANNING = "capacity_planning"
    PERFORMANCE_FORECAST = "performance_forecast"
    RESOURCE_OPTIMIZATION = "resource_optimization"
    ANOMALY_DETECTION = "anomaly_detection"
    TREND_ANALYSIS = "trend_analysis"


class ModelType(str, Enum):
    """Types of ML models"""
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    RANDOM_FOREST = "random_forest"
    ISOLATION_FOREST = "isolation_forest"
    AUTOENCODER = "autoencoder"
    SVM = "svm"
    XGBOOST = "xgboost"
    PROPHET = "prophet"


class AnomalySeverity(str, Enum):
    """Anomaly severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ModelStatus(str, Enum):
    """Model status enumeration"""
    TRAINING = "training"
    ACTIVE = "active"
    INACTIVE = "inactive"
    FAILED = "failed"
    DEPRECATED = "deprecated"


class TrainingStatus(str, Enum):
    """Training job status"""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Base Models
class BaseTimestampModel(BaseModel):
    """Base model with timestamp fields"""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    
    class Config:
        orm_mode = True


class MetricPoint(BaseModel):
    """Individual metric data point"""
    name: str = Field(..., description="Metric name")
    value: float = Field(..., description="Metric value")
    timestamp: datetime = Field(..., description="Timestamp of the metric")
    labels: Dict[str, str] = Field(default_factory=dict, description="Metric labels")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        if v > datetime.utcnow():
            raise ValueError('Timestamp cannot be in the future')
        return v


class SystemContext(BaseModel):
    """System context information"""
    system_id: str = Field(..., description="Unique system identifier")
    system_name: Optional[str] = Field(None, description="Human-readable system name")
    environment: str = Field(default="production", description="Environment (dev/staging/prod)")
    region: Optional[str] = Field(None, description="Geographic region")
    tags: Dict[str, str] = Field(default_factory=dict, description="System tags")


# Request Models
class PredictionRequest(BaseModel):
    """Request for making predictions"""
    system_id: str = Field(..., description="System identifier")
    metrics: List[MetricPoint] = Field(..., description="Input metrics for prediction")
    prediction_type: PredictionType = Field(..., description="Type of prediction to make")
    time_horizon: int = Field(default=12, description="Prediction horizon in time units")
    confidence_threshold: float = Field(default=0.85, ge=0.0, le=1.0, description="Minimum confidence threshold")
    context: Optional[SystemContext] = Field(None, description="Additional system context")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")
    
    @validator('metrics')
    def validate_metrics(cls, v):
        if not v:
            raise ValueError('At least one metric must be provided')
        return v
    
    @validator('time_horizon')
    def validate_time_horizon(cls, v):
        if v < 1 or v > 168:  # Max 1 week
            raise ValueError('Time horizon must be between 1 and 168 hours')
        return v


class AnomalyDetectionRequest(BaseModel):
    """Request for anomaly detection"""
    system_id: str = Field(..., description="System identifier")
    metrics: List[MetricPoint] = Field(..., description="Metrics to analyze for anomalies")
    sensitivity: float = Field(default=0.8, ge=0.1, le=1.0, description="Detection sensitivity")
    time_window: int = Field(default=24, description="Time window in hours to analyze")
    context: Optional[SystemContext] = Field(None, description="System context")
    
    @validator('metrics')
    def validate_metrics(cls, v):
        if not v:
            raise ValueError('At least one metric must be provided')
        return v


class TrainingRequest(BaseModel):
    """Request for model training"""
    model_type: ModelType = Field(..., description="Type of model to train")
    system_id: Optional[str] = Field(None, description="System-specific model training")
    training_data: Dict[str, Any] = Field(..., description="Training data configuration")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Training parameters")
    validation_split: float = Field(default=0.2, ge=0.1, le=0.5, description="Validation data split")
    priority: int = Field(default=1, ge=1, le=10, description="Training priority (1-10)")


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions"""
    requests: List[PredictionRequest] = Field(..., description="List of prediction requests")
    batch_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), description="Batch identifier")
    priority: int = Field(default=1, ge=1, le=10, description="Batch priority")
    
    @validator('requests')
    def validate_requests(cls, v):
        if not v:
            raise ValueError('At least one prediction request must be provided')
        if len(v) > 100:
            raise ValueError('Maximum 100 requests per batch')
        return v


# Response Models
class PredictionResult(BaseModel):
    """Individual prediction result"""
    predicted_value: float = Field(..., description="Predicted value")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    timestamp: datetime = Field(..., description="Timestamp for the prediction")
    lower_bound: Optional[float] = Field(None, description="Lower confidence bound")
    upper_bound: Optional[float] = Field(None, description="Upper confidence bound")


class PredictionResponse(BaseModel):
    """Response from prediction request"""
    system_id: str = Field(..., description="System identifier")
    prediction_type: PredictionType = Field(..., description="Type of prediction")
    model_used: str = Field(..., description="Model used for prediction")
    predictions: List[PredictionResult] = Field(..., description="Prediction results")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="Response generation time")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class Anomaly(BaseModel):
    """Detected anomaly"""
    metric_name: str = Field(..., description="Name of the anomalous metric")
    severity: AnomalySeverity = Field(..., description="Anomaly severity")
    score: float = Field(..., ge=0.0, le=1.0, description="Anomaly score")
    timestamp: datetime = Field(..., description="When the anomaly occurred")
    expected_value: Optional[float] = Field(None, description="Expected value")
    actual_value: float = Field(..., description="Actual anomalous value")
    deviation: float = Field(..., description="Deviation from normal")
    description: str = Field(..., description="Human-readable description")
    recommendations: List[str] = Field(default_factory=list, description="Remediation recommendations")


class AnomalyDetectionResponse(BaseModel):
    """Response from anomaly detection"""
    system_id: str = Field(..., description="System identifier")
    anomalies: List[Anomaly] = Field(..., description="Detected anomalies")
    detection_time: datetime = Field(default_factory=datetime.utcnow)
    total_anomalies: int = Field(..., description="Total number of anomalies detected")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class TrainingJob(BaseModel):
    """Training job information"""
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Job identifier")
    model_type: ModelType = Field(..., description="Type of model being trained")
    status: TrainingStatus = Field(default=TrainingStatus.QUEUED, description="Training status")
    progress: float = Field(default=0.0, ge=0.0, le=100.0, description="Training progress percentage")
    estimated_duration: Optional[int] = Field(None, description="Estimated duration in seconds")
    system_id: Optional[str] = Field(None, description="System-specific training")
    started_at: Optional[datetime] = Field(None, description="Training start time")
    completed_at: Optional[datetime] = Field(None, description="Training completion time")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Training metrics")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class ModelInfo(BaseModel):
    """ML model information"""
    model_id: str = Field(..., description="Model identifier")
    model_type: ModelType = Field(..., description="Model type")
    status: ModelStatus = Field(..., description="Model status")
    version: str = Field(..., description="Model version")
    accuracy: Optional[float] = Field(None, ge=0.0, le=1.0, description="Model accuracy")
    training_date: Optional[datetime] = Field(None, description="When model was trained")
    system_id: Optional[str] = Field(None, description="System-specific model")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Model parameters")
    performance_metrics: Dict[str, float] = Field(default_factory=dict, description="Performance metrics")
    size_bytes: Optional[int] = Field(None, description="Model size in bytes")


class BatchPredictionResponse(BaseModel):
    """Response from batch prediction"""
    batch_id: str = Field(..., description="Batch identifier")
    total_requests: int = Field(..., description="Total number of requests")
    successful_predictions: int = Field(..., description="Number of successful predictions")
    failed_predictions: int = Field(..., description="Number of failed predictions")
    results: List[PredictionResponse] = Field(..., description="Individual prediction results")
    processing_time_ms: float = Field(..., description="Total processing time")
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# Health and Status Models
class HealthCheck(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Overall health status")
    uptime: float = Field(..., description="Service uptime in seconds")
    version: str = Field(..., description="Service version")
    checks: Dict[str, str] = Field(..., description="Individual component health checks")
    active_models: int = Field(..., description="Number of active models")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ServiceMetrics(BaseModel):
    """Service performance metrics"""
    total_predictions: int = Field(..., description="Total predictions made")
    average_response_time: float = Field(..., description="Average response time in ms")
    success_rate: float = Field(..., ge=0.0, le=1.0, description="Prediction success rate")
    active_models: int = Field(..., description="Number of active models")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")
    cpu_usage_percent: float = Field(..., description="CPU usage percentage")
    queue_size: int = Field(..., description="Current queue size")


# Configuration Models
class ModelConfiguration(BaseModel):
    """Model configuration"""
    model_type: ModelType = Field(..., description="Model type")
    parameters: Dict[str, Any] = Field(..., description="Model parameters")
    training_config: Dict[str, Any] = Field(..., description="Training configuration")
    preprocessing_steps: List[str] = Field(default_factory=list, description="Preprocessing steps")
    feature_columns: List[str] = Field(..., description="Feature columns")
    target_column: str = Field(..., description="Target column")


class SystemProfile(BaseModel):
    """System profile for personalized models"""
    system_id: str = Field(..., description="System identifier")
    characteristics: Dict[str, Any] = Field(..., description="System characteristics")
    historical_patterns: Dict[str, Any] = Field(default_factory=dict, description="Historical patterns")
    preferred_models: List[ModelType] = Field(default_factory=list, description="Preferred model types")
    custom_thresholds: Dict[str, float] = Field(default_factory=dict, description="Custom alert thresholds")


# Error Models
class PredictionError(BaseModel):
    """Prediction error details"""
    error_code: str = Field(..., description="Error code")
    error_message: str = Field(..., description="Error message")
    system_id: Optional[str] = Field(None, description="System identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    context: Dict[str, Any] = Field(default_factory=dict, description="Error context")


class ValidationError(BaseModel):
    """Data validation error"""
    field: str = Field(..., description="Field with validation error")
    message: str = Field(..., description="Validation error message")
    value: Any = Field(None, description="Invalid value")


# Utility Models
class TimeSeriesData(BaseModel):
    """Time series data structure"""
    timestamps: List[datetime] = Field(..., description="Timestamps")
    values: List[float] = Field(..., description="Values")
    metric_name: str = Field(..., description="Metric name")
    
    @validator('values')
    def validate_values_length(cls, v, values):
        if 'timestamps' in values and len(v) != len(values['timestamps']):
            raise ValueError('Values and timestamps must have the same length')
        return v


class FeatureImportance(BaseModel):
    """Feature importance for model explainability"""
    feature_name: str = Field(..., description="Feature name")
    importance_score: float = Field(..., description="Importance score")
    rank: int = Field(..., description="Importance rank")


class ModelExplanation(BaseModel):
    """Model prediction explanation"""
    feature_importances: List[FeatureImportance] = Field(..., description="Feature importance scores")
    prediction_confidence: float = Field(..., description="Prediction confidence")
    key_factors: List[str] = Field(..., description="Key factors influencing the prediction")
    explanation_text: str = Field(..., description="Human-readable explanation") 