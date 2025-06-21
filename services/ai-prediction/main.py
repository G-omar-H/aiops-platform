#!/usr/bin/env python3
"""
AI-Powered Prediction Service - Main Application Entry Point

This service provides real-time system failure prediction using machine learning models.
Supports multiple algorithms including LSTM, Random Forest, and Transformer models.

Features:
- Real-time anomaly detection
- Predictive failure analysis  
- Multi-metric correlation analysis
- Auto-scaling prediction recommendations
- Integration with monitoring systems
"""

import asyncio
import logging
import os
import signal
import sys
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from prometheus_client import make_asgi_app, Counter, Histogram, Gauge

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.api.routes import predictions, health, metrics, models
from app.core.events import startup_event, shutdown_event
from app.db.session import DatabaseManager
from app.services.ml_service import MLService
from app.services.metrics_service import MetricsService

# Prometheus metrics
REQUEST_COUNT = Counter('ai_prediction_requests_total', 'Total prediction requests', ['endpoint', 'method', 'status'])
REQUEST_DURATION = Histogram('ai_prediction_request_duration_seconds', 'Request duration', ['endpoint'])
ACTIVE_PREDICTIONS = Gauge('ai_prediction_active_jobs', 'Active prediction jobs')
MODEL_ACCURACY = Gauge('ai_prediction_model_accuracy', 'Current model accuracy', ['model_type'])

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    logger = logging.getLogger(__name__)
    
    try:
        # Startup
        logger.info("🚀 Starting AI Prediction Service...")
        await startup_event()
        
        # Initialize core services
        app.state.db_manager = DatabaseManager()
        app.state.ml_service = MLService()
        app.state.metrics_service = MetricsService()
        
        # Load ML models
        await app.state.ml_service.load_models()
        logger.info("✅ AI models loaded successfully")
        
        # Start background tasks
        asyncio.create_task(app.state.metrics_service.start_collection())
        logger.info("✅ Metrics collection started")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    finally:
        # Shutdown
        logger.info("🛑 Shutting down AI Prediction Service...")
        await shutdown_event()
        
        if hasattr(app.state, 'ml_service'):
            await app.state.ml_service.cleanup()
        if hasattr(app.state, 'metrics_service'):
            await app.state.metrics_service.stop_collection()
        if hasattr(app.state, 'db_manager'):
            await app.state.db_manager.close()
        
        logger.info("✅ Shutdown complete")

def create_application() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="AIOps AI Prediction Service",
        description="Enterprise-grade AI-powered system failure prediction and anomaly detection",
        version="1.0.0",
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        lifespan=lifespan
    )
    
    # Security middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.ALLOWED_HOSTS
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )
    
    # Compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Request middleware for metrics
    @app.middleware("http")
    async def metrics_middleware(request, call_next):
        start_time = time.time()
        
        response = await call_next(request)
        
        duration = time.time() - start_time
        endpoint = request.url.path
        method = request.method
        status = response.status_code
        
        REQUEST_COUNT.labels(endpoint=endpoint, method=method, status=status).inc()
        REQUEST_DURATION.labels(endpoint=endpoint).observe(duration)
        
        return response
    
    # Include API routes
    app.include_router(health.router, prefix="/health", tags=["Health"])
    app.include_router(predictions.router, prefix="/api/v1/predictions", tags=["Predictions"])
    app.include_router(metrics.router, prefix="/api/v1/metrics", tags=["Metrics"])
    app.include_router(models.router, prefix="/api/v1/models", tags=["Models"])
    
    # Prometheus metrics endpoint
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)
    
    return app

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logging.info(f"Received signal {signum}. Initiating graceful shutdown...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

app = create_application()

if __name__ == "__main__":
    setup_logging()
    setup_signal_handlers()
    
    logger = logging.getLogger(__name__)
    logger.info(f"🚀 Starting AI Prediction Service on {settings.HOST}:{settings.PORT}")
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1 if settings.DEBUG else settings.WORKERS,
        log_level="debug" if settings.DEBUG else "info",
        access_log=True,
        loop="uvloop"
    ) 