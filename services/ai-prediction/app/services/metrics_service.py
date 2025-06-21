"""
Metrics Collection Service

Handles collection and processing of system metrics for ML models.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import aiohttp
import json

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class MetricsService:
    """Service for collecting and processing system metrics."""
    
    def __init__(self):
        self.collection_active = False
        self.collection_task: Optional[asyncio.Task] = None
        
    async def start_collection(self) -> None:
        """Start metrics collection background task."""
        if not self.collection_active:
            self.collection_active = True
            self.collection_task = asyncio.create_task(self._collection_loop())
            logger.info("📊 Metrics collection started")
    
    async def stop_collection(self) -> None:
        """Stop metrics collection background task."""
        self.collection_active = False
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        logger.info("📊 Metrics collection stopped")
    
    async def _collection_loop(self) -> None:
        """Background metrics collection loop."""
        while self.collection_active:
            try:
                await self._collect_metrics()
                await asyncio.sleep(settings.METRICS_COLLECTION_INTERVAL)
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _collect_metrics(self) -> None:
        """Collect metrics from various sources."""
        try:
            # Collect from Prometheus
            prometheus_metrics = await self._collect_from_prometheus()
            
            # Process and store metrics
            if prometheus_metrics:
                await self._process_metrics(prometheus_metrics)
                
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
    
    async def _collect_from_prometheus(self) -> Optional[Dict[str, Any]]:
        """Collect metrics from Prometheus."""
        try:
            async with aiohttp.ClientSession() as session:
                # Query Prometheus for system metrics
                queries = {
                    'cpu_usage': 'avg(cpu_usage_percent)',
                    'memory_usage': 'avg(memory_usage_percent)',
                    'disk_usage': 'avg(disk_usage_percent)',
                    'response_time': 'avg(http_request_duration_seconds)',
                    'error_rate': 'rate(http_requests_total{status=~"5.."}[5m])'
                }
                
                metrics = {}
                for metric_name, query in queries.items():
                    try:
                        url = f"{settings.PROMETHEUS_URL}/api/v1/query"
                        params = {'query': query}
                        
                        async with session.get(url, params=params) as response:
                            if response.status == 200:
                                data = await response.json()
                                if data.get('status') == 'success':
                                    result = data.get('data', {}).get('result', [])
                                    if result:
                                        metrics[metric_name] = float(result[0]['value'][1])
                    except Exception as e:
                        logger.warning(f"Failed to collect {metric_name}: {e}")
                
                return metrics if metrics else None
                
        except Exception as e:
            logger.error(f"Prometheus collection failed: {e}")
            return None
    
    async def _process_metrics(self, metrics: Dict[str, Any]) -> None:
        """Process and store collected metrics."""
        try:
            # Add timestamp
            metrics['timestamp'] = datetime.utcnow().isoformat()
            
            # Log metrics for debugging
            logger.debug(f"Collected metrics: {metrics}")
            
            # In a real implementation, you would:
            # 1. Store metrics in time-series database
            # 2. Trigger alerts if thresholds exceeded
            # 3. Update ML model training data
            # 4. Send to analytics pipeline
            
        except Exception as e:
            logger.error(f"Failed to process metrics: {e}")
    
    async def get_latest_metrics(self, system_id: str) -> Optional[Dict[str, float]]:
        """Get latest metrics for a system."""
        # Placeholder implementation
        # In production, this would query a time-series database
        return {
            'cpu_usage': 0.45,
            'memory_usage': 0.67,
            'disk_usage': 0.23,
            'network_io': 1024.0,
            'request_count': 1500,
            'response_time': 250.0,
            'error_rate': 0.02,
            'gc_frequency': 30,
            'thread_count': 150,
            'connection_count': 500
        }
    
    async def get_historical_metrics(
        self, 
        system_id: str, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Get historical metrics for a system."""
        # Placeholder implementation
        return [] 