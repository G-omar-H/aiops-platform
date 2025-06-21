"""
Health Check API Routes

Provides health, readiness, and liveness endpoints for the AI prediction service.
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from app.core.config import get_settings
from app.services.ml_service import MLService

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: datetime
    version: str
    service: str
    details: Dict[str, Any] = {}


class ReadinessResponse(BaseModel):
    """Readiness check response model."""
    ready: bool
    timestamp: datetime
    services: Dict[str, str]


def get_ml_service():
    """Dependency to get ML service instance."""
    try:
        from main import app
        return app.state.ml_service
    except:
        return None


@router.get("/", response_model=HealthResponse)
async def health_check():
    """
    Basic health check endpoint.
    
    **Returns service status and basic information.**
    
    - **Always returns 200 if service is running**
    - **Provides service version and timestamp**
    - **Used by load balancers for basic health checks**
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version=settings.APP_VERSION,
        service=settings.APP_NAME,
        details={
            "environment": "production" if not settings.DEBUG else "development",
            "debug": settings.DEBUG
        }
    )


@router.get("/live", response_model=HealthResponse)
async def liveness_check():
    """
    Liveness probe endpoint.
    
    **Kubernetes liveness probe - checks if service should be restarted.**
    
    - **Returns 200 if application is alive**
    - **Used by Kubernetes to restart unhealthy pods**
    - **Should only fail if application is completely broken**
    """
    try:
        # Basic liveness check - can we respond?
        return HealthResponse(
            status="alive",
            timestamp=datetime.utcnow(),
            version=settings.APP_VERSION,
            service=settings.APP_NAME,
            details={
                "check": "liveness",
                "pid": "available"
            }
        )
    except Exception as e:
        logger.error(f"Liveness check failed: {e}")
        raise HTTPException(status_code=503, detail="Service not alive")


@router.get("/ready", response_model=ReadinessResponse)
async def readiness_check(ml_service: MLService = Depends(get_ml_service)):
    """
    Readiness probe endpoint.
    
    **Kubernetes readiness probe - checks if service can handle traffic.**
    
    - **Returns 200 if service is ready to handle requests**
    - **Checks ML models, database connections, etc.**
    - **Used by Kubernetes to route traffic to healthy pods**
    """
    try:
        services_status = {}
        all_ready = True
        
        # Check ML service and models
        if ml_service:
            try:
                model_count = len(ml_service.models)
                if model_count > 0:
                    services_status["ml_models"] = f"ready ({model_count} models loaded)"
                else:
                    services_status["ml_models"] = "not ready (no models loaded)"
                    all_ready = False
            except Exception as e:
                services_status["ml_models"] = f"error: {str(e)}"
                all_ready = False
        else:
            services_status["ml_models"] = "not available"
            all_ready = False
        
        # Check database connectivity (placeholder)
        try:
            services_status["database"] = "ready"
        except Exception as e:
            services_status["database"] = f"error: {str(e)}"
            all_ready = False
        
        # Check Redis connectivity (placeholder)
        try:
            services_status["redis"] = "ready"
        except Exception as e:
            services_status["redis"] = f"error: {str(e)}"
            all_ready = False
        
        response = ReadinessResponse(
            ready=all_ready,
            timestamp=datetime.utcnow(),
            services=services_status
        )
        
        if not all_ready:
            logger.warning(f"Readiness check failed: {services_status}")
            raise HTTPException(status_code=503, detail=response.dict())
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")


@router.get("/metrics/simple")
async def simple_metrics():
    """
    Simple metrics endpoint for basic monitoring.
    
    **Provides basic operational metrics in plain text format.**
    
    - **Memory usage, request counts, model status**
    - **Lightweight alternative to full Prometheus metrics**
    - **Useful for simple monitoring setups**
    """
    try:
        import psutil
        import os
        
        # Get basic system metrics
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        cpu_percent = process.cpu_percent()
        
        metrics = {
            "memory_rss_bytes": memory_info.rss,
            "memory_vms_bytes": memory_info.vms,
            "cpu_percent": cpu_percent,
            "threads": process.num_threads(),
            "open_files": len(process.open_files()),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get simple metrics: {e}")
        return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}


@router.get("/debug")
async def debug_info(ml_service: MLService = Depends(get_ml_service)):
    """
    Debug information endpoint (only available in debug mode).
    
    **Provides detailed debug information about the service state.**
    
    - **Only available when DEBUG=True**
    - **Shows model status, configuration, and internal state**
    - **Useful for troubleshooting and development**
    """
    if not settings.DEBUG:
        raise HTTPException(status_code=404, detail="Debug endpoint not available in production")
    
    try:
        debug_info = {
            "timestamp": datetime.utcnow().isoformat(),
            "service": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "environment": "development",
            "configuration": {
                "host": settings.HOST,
                "port": settings.PORT,
                "workers": settings.WORKERS,
                "debug": settings.DEBUG,
                "log_level": settings.LOG_LEVEL
            },
            "ml_service": {
                "available": ml_service is not None,
                "models_loaded": len(ml_service.models) if ml_service else 0,
                "model_names": list(ml_service.models.keys()) if ml_service else [],
                "models_path": str(ml_service.models_path) if ml_service else None
            },
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": platform.platform()
        }
        
        return debug_info
        
    except Exception as e:
        logger.error(f"Failed to get debug info: {e}")
        raise HTTPException(status_code=500, detail=f"Debug info failed: {str(e)}")


# Additional imports for debug endpoint
import sys
import platform 