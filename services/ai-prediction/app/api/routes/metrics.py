"""
Metrics API Routes

Provides endpoints for system metrics collection and analysis.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query

from app.core.config import get_settings
from app.services.metrics_service import MetricsService
from app.services.auth_service import verify_token
from app.api.routes.predictions import get_current_user

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()


def get_metrics_service():
    """Dependency to get metrics service instance."""
    from main import app
    return app.state.metrics_service


@router.get("/systems/{system_id}/latest")
async def get_latest_metrics(
    system_id: str,
    metrics_service: MetricsService = Depends(get_metrics_service),
    user: dict = Depends(get_current_user)
):
    """
    Get latest metrics for a specific system.
    
    **Retrieve the most recent metrics data for system monitoring.**
    
    - **Returns current system performance indicators**
    - **Includes CPU, memory, disk, network metrics**
    - **Used for real-time monitoring dashboards**
    """
    try:
        metrics = await metrics_service.get_latest_metrics(system_id)
        
        if not metrics:
            raise HTTPException(status_code=404, detail=f"No metrics found for system {system_id}")
        
        return {
            'system_id': system_id,
            'metrics': metrics,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get latest metrics for {system_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@router.get("/systems/{system_id}/historical")
async def get_historical_metrics(
    system_id: str,
    start_time: datetime = Query(..., description="Start time for metrics range"),
    end_time: datetime = Query(..., description="End time for metrics range"),
    metrics_service: MetricsService = Depends(get_metrics_service),
    user: dict = Depends(get_current_user)
):
    """
    Get historical metrics for a specific system.
    
    **Retrieve historical metrics data for trend analysis.**
    
    - **Supports custom time ranges**
    - **Useful for performance analysis and reporting**
    - **Returns time-series data for charting**
    """
    try:
        # Validate time range
        if end_time <= start_time:
            raise HTTPException(status_code=400, detail="End time must be after start time")
        
        if (end_time - start_time).days > 30:
            raise HTTPException(status_code=400, detail="Time range cannot exceed 30 days")
        
        metrics = await metrics_service.get_historical_metrics(system_id, start_time, end_time)
        
        return {
            'system_id': system_id,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'metrics': metrics,
            'count': len(metrics)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get historical metrics for {system_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get historical metrics: {str(e)}")


@router.get("/aggregated")
async def get_aggregated_metrics(
    time_range: str = Query(default="1h", description="Time range (1h, 6h, 24h, 7d)"),
    metric_names: Optional[List[str]] = Query(default=None, description="Specific metrics to include"),
    user: dict = Depends(get_current_user)
):
    """
    Get aggregated metrics across all systems.
    
    **Retrieve aggregated metrics for fleet-wide monitoring.**
    
    - **Provides system-wide performance overview**
    - **Supports multiple time ranges**
    - **Includes averages, percentiles, and trends**
    """
    try:
        # Parse time range
        time_ranges = {
            '1h': timedelta(hours=1),
            '6h': timedelta(hours=6),
            '24h': timedelta(hours=24),
            '7d': timedelta(days=7)
        }
        
        if time_range not in time_ranges:
            raise HTTPException(status_code=400, detail=f"Invalid time range. Must be one of: {list(time_ranges.keys())}")
        
        delta = time_ranges[time_range]
        end_time = datetime.utcnow()
        start_time = end_time - delta
        
        # This would aggregate metrics from multiple systems
        # For now, return placeholder data
        aggregated_data = {
            'time_range': time_range,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'metrics': {
                'avg_cpu_usage': 0.65,
                'avg_memory_usage': 0.72,
                'avg_response_time': 245.0,
                'total_requests': 15000,
                'error_rate': 0.015,
                'systems_count': 10
            },
            'trends': {
                'cpu_usage': 'stable',
                'memory_usage': 'increasing',
                'response_time': 'stable',
                'error_rate': 'decreasing'
            }
        }
        
        return aggregated_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get aggregated metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get aggregated metrics: {str(e)}")


@router.post("/systems/{system_id}/custom-query")
async def execute_custom_metrics_query(
    system_id: str,
    query: Dict[str, Any],
    user: dict = Depends(get_current_user)
):
    """
    Execute custom metrics query.
    
    **Run custom queries against metrics data.**
    
    - **Supports complex filtering and aggregation**
    - **Useful for custom dashboards and reports**
    - **Returns query results in structured format**
    """
    try:
        # Validate user has permissions for custom queries
        if user.get('role') not in ['admin', 'analyst']:
            raise HTTPException(status_code=403, detail="Insufficient permissions for custom queries")
        
        # This would execute the custom query against the metrics database
        # For now, return placeholder response
        result = {
            'system_id': system_id,
            'query': query,
            'results': [],
            'execution_time_ms': 125,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to execute custom query: {e}")
        raise HTTPException(status_code=500, detail=f"Custom query failed: {str(e)}")


@router.get("/alerts")
async def get_metrics_alerts(
    severity: Optional[str] = Query(default=None, description="Filter by severity (low, medium, high, critical)"),
    system_id: Optional[str] = Query(default=None, description="Filter by system ID"),
    active_only: bool = Query(default=True, description="Show only active alerts"),
    user: dict = Depends(get_current_user)
):
    """
    Get metrics-based alerts.
    
    **Retrieve active alerts based on metrics thresholds.**
    
    - **Supports filtering by severity and system**
    - **Shows threshold violations and anomalies**
    - **Integrates with alerting and notification systems**
    """
    try:
        # This would query the alerting system
        # For now, return placeholder alerts
        alerts = [
            {
                'alert_id': 'alert_001',
                'system_id': 'web-server-01',
                'metric': 'cpu_usage',
                'current_value': 0.95,
                'threshold': 0.8,
                'severity': 'high',
                'status': 'active',
                'created_at': datetime.utcnow().isoformat(),
                'description': 'CPU usage above threshold'
            },
            {
                'alert_id': 'alert_002',
                'system_id': 'db-server-01',
                'metric': 'response_time',
                'current_value': 2500.0,
                'threshold': 1000.0,
                'severity': 'medium',
                'status': 'active',
                'created_at': (datetime.utcnow() - timedelta(minutes=30)).isoformat(),
                'description': 'Response time degradation detected'
            }
        ]
        
        # Apply filters
        if severity:
            alerts = [a for a in alerts if a['severity'] == severity]
        
        if system_id:
            alerts = [a for a in alerts if a['system_id'] == system_id]
        
        if active_only:
            alerts = [a for a in alerts if a['status'] == 'active']
        
        return {
            'alerts': alerts,
            'total_count': len(alerts),
            'filters': {
                'severity': severity,
                'system_id': system_id,
                'active_only': active_only
            },
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}") 