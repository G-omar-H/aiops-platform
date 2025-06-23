#!/usr/bin/env python3
"""
AIOps AI Prediction Service
Enterprise-grade ML service for predictive analytics and anomaly detection
"""

import asyncio
import logging
import signal
import sys
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import start_http_server, Counter, Histogram, Gauge
import structlog

from app.config import Settings, get_settings
from app.models import (
    PredictionRequest, PredictionResponse, AnomalyDetectionRequest,
    TrainingRequest, ModelStatus, HealthCheck
)
from app.services.prediction_engine import PredictionEngine
from app.services.anomaly_detector import AnomalyDetector
from app.services.model_manager import ModelManager
from app.services.data_processor import DataProcessor
from app.middleware import (
    MetricsMiddleware, LoggingMiddleware, AuthenticationMiddleware
)
from app.utils.logger import setup_logging
from app.utils.metrics import setup_metrics

# Global variables
prediction_engine: Optional[PredictionEngine] = None
anomaly_detector: Optional[AnomalyDetector] = None
model_manager: Optional[ModelManager] = None
data_processor: Optional[DataProcessor] = None
startup_time = time.time()

# Prometheus metrics
PREDICTIONS_TOTAL = Counter('ai_predictions_total', 'Total predictions made', ['model_type', 'status'])
PREDICTION_DURATION = Histogram('ai_prediction_duration_seconds', 'Prediction duration')
ACTIVE_MODELS = Gauge('ai_active_models', 'Number of active models')
ANOMALIES_DETECTED = Counter('ai_anomalies_detected_total', 'Total anomalies detected', ['severity'])

logger = structlog.get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    # Startup
    logger.info("🚀 Starting AI Prediction Service...")
    
    settings = get_settings()
    
    # Initialize services
    global prediction_engine, anomaly_detector, model_manager, data_processor
    
    try:
        # Setup logging and metrics
        setup_logging(settings)
        setup_metrics()
        
        # Start Prometheus metrics server
        start_http_server(settings.METRICS_PORT)
        logger.info(f"📊 Metrics server started on port {settings.METRICS_PORT}")
        
        # Initialize core services
        model_manager = ModelManager(settings)
        await model_manager.initialize()
        
        data_processor = DataProcessor(settings)
        await data_processor.initialize()
        
        prediction_engine = PredictionEngine(settings, model_manager, data_processor)
        await prediction_engine.initialize()
        
        anomaly_detector = AnomalyDetector(settings, model_manager, data_processor)
        await anomaly_detector.initialize()
        
        # Load pre-trained models
        await model_manager.load_default_models()
        
        # Start background tasks
        asyncio.create_task(model_retraining_scheduler())
        asyncio.create_task(health_monitoring_task())
        
        logger.info("✅ AI Prediction Service started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to start AI Prediction Service: {e}")
        raise
    
    # Shutdown
    logger.info("🛑 Shutting down AI Prediction Service...")
    
    # Cleanup services
    if prediction_engine:
        await prediction_engine.cleanup()
    if anomaly_detector:
        await anomaly_detector.cleanup()
    if model_manager:
        await model_manager.cleanup()
    if data_processor:
        await data_processor.cleanup()
    
    logger.info("✅ AI Prediction Service shutdown completed")

# Create FastAPI app
app = FastAPI(
    title="AIOps AI Prediction Service",
    description="Enterprise AI/ML service for predictive analytics and anomaly detection",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on environment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(MetricsMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(AuthenticationMiddleware)

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    uptime = time.time() - startup_time
    
    # Check service health
    checks = {
        "prediction_engine": "healthy" if prediction_engine and prediction_engine.is_healthy() else "unhealthy",
        "anomaly_detector": "healthy" if anomaly_detector and anomaly_detector.is_healthy() else "unhealthy",
        "model_manager": "healthy" if model_manager and model_manager.is_healthy() else "unhealthy",
        "data_processor": "healthy" if data_processor and data_processor.is_healthy() else "unhealthy",
    }
    
    overall_status = "healthy" if all(status == "healthy" for status in checks.values()) else "unhealthy"
    
    return HealthCheck(
        status=overall_status,
        uptime=uptime,
        version="1.0.0",
        checks=checks,
        active_models=len(model_manager.get_active_models()) if model_manager else 0
    )

@app.get("/health/live")
async def liveness_check():
    """Kubernetes liveness probe"""
    return {"status": "alive"}

@app.get("/health/ready")
async def readiness_check():
    """Kubernetes readiness probe"""
    if not all([prediction_engine, anomaly_detector, model_manager, data_processor]):
        raise HTTPException(status_code=503, detail="Service not ready")
    return {"status": "ready"}

@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings)
):
    """Make predictions based on input metrics"""
    start_time = time.time()
    
    try:
        if not prediction_engine:
            raise HTTPException(status_code=503, detail="Prediction engine not available")
        
        # Validate request
        if not request.metrics or len(request.metrics) == 0:
            raise HTTPException(status_code=400, detail="No metrics provided")
        
        # Make prediction
        result = await prediction_engine.predict(
            metrics=request.metrics,
            system_id=request.system_id,
            prediction_type=request.prediction_type,
            time_horizon=request.time_horizon,
            confidence_threshold=request.confidence_threshold
        )
        
        # Record metrics
        duration = time.time() - start_time
        PREDICTIONS_TOTAL.labels(
            model_type=request.prediction_type,
            status="success"
        ).inc()
        PREDICTION_DURATION.observe(duration)
        
        # Schedule background tasks
        background_tasks.add_task(
            log_prediction_result,
            request.system_id,
            request.prediction_type,
            result,
            duration
        )
        
        logger.info(
            "Prediction completed",
            system_id=request.system_id,
            prediction_type=request.prediction_type,
            duration=duration,
            confidence=result.confidence
        )
        
        return result
        
    except Exception as e:
        PREDICTIONS_TOTAL.labels(
            model_type=request.prediction_type,
            status="error"
        ).inc()
        
        logger.error(
            "Prediction failed",
            system_id=request.system_id,
            error=str(e)
        )
        
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/v1/detect-anomalies")
async def detect_anomalies(
    request: AnomalyDetectionRequest,
    background_tasks: BackgroundTasks
):
    """Detect anomalies in system metrics"""
    try:
        if not anomaly_detector:
            raise HTTPException(status_code=503, detail="Anomaly detector not available")
        
        # Detect anomalies
        anomalies = await anomaly_detector.detect(
            metrics=request.metrics,
            system_id=request.system_id,
            sensitivity=request.sensitivity,
            time_window=request.time_window
        )
        
        # Record metrics
        for anomaly in anomalies:
            ANOMALIES_DETECTED.labels(severity=anomaly.severity).inc()
        
        # Schedule background processing
        background_tasks.add_task(
            process_anomalies,
            request.system_id,
            anomalies
        )
        
        logger.info(
            "Anomaly detection completed",
            system_id=request.system_id,
            anomalies_found=len(anomalies)
        )
        
        return {
            "system_id": request.system_id,
            "anomalies": anomalies,
            "detection_time": time.time(),
            "total_anomalies": len(anomalies)
        }
        
    except Exception as e:
        logger.error(
            "Anomaly detection failed",
            system_id=request.system_id,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {str(e)}")

@app.post("/api/v1/train")
async def train_model(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
):
    """Train or retrain ML models"""
    try:
        if not model_manager:
            raise HTTPException(status_code=503, detail="Model manager not available")
        
        # Start training process
        training_job = await model_manager.start_training(
            model_type=request.model_type,
            training_data=request.training_data,
            parameters=request.parameters,
            system_id=request.system_id
        )
        
        # Schedule background monitoring
        background_tasks.add_task(
            monitor_training_job,
            training_job.job_id
        )
        
        logger.info(
            "Model training started",
            job_id=training_job.job_id,
            model_type=request.model_type,
            system_id=request.system_id
        )
        
        return {
            "job_id": training_job.job_id,
            "status": training_job.status,
            "model_type": request.model_type,
            "estimated_duration": training_job.estimated_duration
        }
        
    except Exception as e:
        logger.error(
            "Model training failed to start",
            model_type=request.model_type,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.get("/api/v1/models")
async def list_models():
    """List all available models"""
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not available")
    
    models = await model_manager.list_models()
    ACTIVE_MODELS.set(len([m for m in models if m.status == "active"]))
    
    return {
        "models": models,
        "total": len(models),
        "active": len([m for m in models if m.status == "active"])
    }

@app.get("/api/v1/models/{model_id}/status")
async def get_model_status(model_id: str) -> ModelStatus:
    """Get status of a specific model"""
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not available")
    
    status = await model_manager.get_model_status(model_id)
    if not status:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return status

# Background tasks
async def model_retraining_scheduler():
    """Background task to schedule model retraining"""
    while True:
        try:
            if model_manager:
                await model_manager.check_retraining_schedule()
            await asyncio.sleep(3600)  # Check every hour
        except Exception as e:
            logger.error("Model retraining scheduler error", error=str(e))
            await asyncio.sleep(60)

async def health_monitoring_task():
    """Background task to monitor service health"""
    while True:
        try:
            # Update metrics
            if model_manager:
                active_models = len(model_manager.get_active_models())
                ACTIVE_MODELS.set(active_models)
            
            await asyncio.sleep(30)  # Update every 30 seconds
        except Exception as e:
            logger.error("Health monitoring error", error=str(e))
            await asyncio.sleep(10)

async def log_prediction_result(system_id: str, prediction_type: str, result, duration: float):
    """Background task to log prediction results"""
    try:
        # Log to structured storage for analysis
        logger.info(
            "Prediction result logged",
            system_id=system_id,
            prediction_type=prediction_type,
            confidence=result.confidence,
            duration=duration
        )
    except Exception as e:
        logger.error("Failed to log prediction result", error=str(e))

async def process_anomalies(system_id: str, anomalies: List):
    """Background task to process detected anomalies"""
    try:
        # Process each anomaly
        for anomaly in anomalies:
            # Could trigger alerts, auto-remediation, etc.
            logger.info(
                "Anomaly processed",
                system_id=system_id,
                anomaly_type=anomaly.type,
                severity=anomaly.severity
            )
    except Exception as e:
        logger.error("Failed to process anomalies", error=str(e))

async def monitor_training_job(job_id: str):
    """Background task to monitor training job progress"""
    try:
        if model_manager:
            await model_manager.monitor_training_job(job_id)
    except Exception as e:
        logger.error("Training job monitoring failed", job_id=job_id, error=str(e))

def handle_shutdown(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

if __name__ == "__main__":
    settings = get_settings()
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1,  # Use 1 worker for ML models to avoid memory issues
        access_log=True,
        log_config=None  # Use our custom logging
    ) 