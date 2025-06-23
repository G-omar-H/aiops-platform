"""
Metrics utility for AIOps AI Prediction Service
Provides Prometheus metrics integration and ML model performance tracking
"""

import time
from typing import Dict, Any, Optional, List
from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry
import structlog


logger = structlog.get_logger()


class MetricsCollector:
    """Central metrics collector for the AI prediction service"""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Setup all Prometheus metrics"""
        
        # Prediction metrics
        self.predictions_total = Counter(
            'ai_predictions_total',
            'Total number of predictions made',
            ['model_type', 'prediction_type', 'status'],
            registry=self.registry
        )
        
        self.prediction_duration = Histogram(
            'ai_prediction_duration_seconds',
            'Time spent on predictions',
            ['model_type', 'prediction_type'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry
        )
        
        self.prediction_confidence = Histogram(
            'ai_prediction_confidence',
            'Confidence scores of predictions',
            ['model_type', 'prediction_type'],
            buckets=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0],
            registry=self.registry
        )
        
        self.batch_prediction_size = Histogram(
            'ai_batch_prediction_size',
            'Size of batch predictions',
            buckets=[1, 5, 10, 20, 50, 100, 200, 500],
            registry=self.registry
        )
        
        # Model metrics
        self.active_models = Gauge(
            'ai_active_models_total',
            'Number of active models',
            ['model_type'],
            registry=self.registry
        )
        
        self.model_accuracy = Gauge(
            'ai_model_accuracy',
            'Current model accuracy',
            ['model_id', 'model_type'],
            registry=self.registry
        )
        
        self.model_training_duration = Histogram(
            'ai_model_training_duration_seconds',
            'Time spent training models',
            ['model_type'],
            buckets=[60, 300, 600, 1800, 3600, 7200, 14400, 28800],
            registry=self.registry
        )
        
        self.model_size_bytes = Gauge(
            'ai_model_size_bytes',
            'Model size in bytes',
            ['model_id', 'model_type'],
            registry=self.registry
        )
        
        # Anomaly detection metrics
        self.anomalies_detected = Counter(
            'ai_anomalies_detected_total',
            'Total anomalies detected',
            ['severity', 'metric_name'],
            registry=self.registry
        )
        
        self.anomaly_detection_duration = Histogram(
            'ai_anomaly_detection_duration_seconds',
            'Time spent on anomaly detection',
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=self.registry
        )
        
        # Data quality metrics
        self.data_quality_score = Histogram(
            'ai_data_quality_score',
            'Data quality scores',
            ['system_id'],
            buckets=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0],
            registry=self.registry
        )
        
        self.missing_data_points = Counter(
            'ai_missing_data_points_total',
            'Number of missing data points',
            ['system_id', 'metric_name'],
            registry=self.registry
        )
        
        # Feature engineering metrics
        self.feature_engineering_duration = Histogram(
            'ai_feature_engineering_duration_seconds',
            'Time spent on feature engineering',
            ['feature_type'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
            registry=self.registry
        )
        
        self.features_generated = Counter(
            'ai_features_generated_total',
            'Total features generated',
            ['feature_type', 'status'],
            registry=self.registry
        )
        
        # Cache metrics
        self.cache_hits = Counter(
            'ai_cache_hits_total',
            'Cache hits',
            ['cache_type'],
            registry=self.registry
        )
        
        self.cache_misses = Counter(
            'ai_cache_misses_total',
            'Cache misses',
            ['cache_type'],
            registry=self.registry
        )
        
        self.cache_size = Gauge(
            'ai_cache_size_items',
            'Number of items in cache',
            ['cache_type'],
            registry=self.registry
        )
        
        # Memory and resource metrics
        self.memory_usage_mb = Gauge(
            'ai_memory_usage_mb',
            'Memory usage in MB',
            ['component'],
            registry=self.registry
        )
        
        self.cpu_usage_percent = Gauge(
            'ai_cpu_usage_percent',
            'CPU usage percentage',
            ['component'],
            registry=self.registry
        )
        
        self.gpu_usage_percent = Gauge(
            'ai_gpu_usage_percent',
            'GPU usage percentage',
            ['gpu_id'],
            registry=self.registry
        )
        
        # Queue metrics
        self.queue_size = Gauge(
            'ai_queue_size',
            'Current queue size',
            ['queue_type'],
            registry=self.registry
        )
        
        self.queue_processing_time = Histogram(
            'ai_queue_processing_time_seconds',
            'Time items spend in queue',
            ['queue_type'],
            buckets=[0.1, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0],
            registry=self.registry
        )
        
        # Error metrics
        self.errors_total = Counter(
            'ai_errors_total',
            'Total errors',
            ['error_type', 'component'],
            registry=self.registry
        )
        
        self.error_rate = Gauge(
            'ai_error_rate',
            'Error rate percentage',
            ['component'],
            registry=self.registry
        )
    
    # Prediction metrics methods
    def record_prediction(self, model_type: str, prediction_type: str, 
                         duration: float, confidence: float, status: str = "success"):
        """Record a prediction event"""
        self.predictions_total.labels(
            model_type=model_type, 
            prediction_type=prediction_type, 
            status=status
        ).inc()
        
        self.prediction_duration.labels(
            model_type=model_type, 
            prediction_type=prediction_type
        ).observe(duration)
        
        if confidence > 0:
            self.prediction_confidence.labels(
                model_type=model_type, 
                prediction_type=prediction_type
            ).observe(confidence)
    
    def record_batch_prediction(self, batch_size: int):
        """Record batch prediction metrics"""
        self.batch_prediction_size.observe(batch_size)
    
    # Model metrics methods
    def update_active_models(self, model_type: str, count: int):
        """Update active model count"""
        self.active_models.labels(model_type=model_type).set(count)
    
    def update_model_accuracy(self, model_id: str, model_type: str, accuracy: float):
        """Update model accuracy"""
        self.model_accuracy.labels(model_id=model_id, model_type=model_type).set(accuracy)
    
    def record_model_training(self, model_type: str, duration: float):
        """Record model training duration"""
        self.model_training_duration.labels(model_type=model_type).observe(duration)
    
    def update_model_size(self, model_id: str, model_type: str, size_bytes: int):
        """Update model size"""
        self.model_size_bytes.labels(model_id=model_id, model_type=model_type).set(size_bytes)
    
    # Anomaly detection methods
    def record_anomaly(self, severity: str, metric_name: str):
        """Record an anomaly detection"""
        self.anomalies_detected.labels(severity=severity, metric_name=metric_name).inc()
    
    def record_anomaly_detection_duration(self, duration: float):
        """Record anomaly detection duration"""
        self.anomaly_detection_duration.observe(duration)
    
    # Data quality methods
    def record_data_quality(self, system_id: str, quality_score: float):
        """Record data quality score"""
        self.data_quality_score.labels(system_id=system_id).observe(quality_score)
    
    def record_missing_data(self, system_id: str, metric_name: str, count: int = 1):
        """Record missing data points"""
        self.missing_data_points.labels(system_id=system_id, metric_name=metric_name).inc(count)
    
    # Feature engineering methods
    def record_feature_engineering(self, feature_type: str, duration: float, 
                                 feature_count: int, status: str = "success"):
        """Record feature engineering metrics"""
        self.feature_engineering_duration.labels(feature_type=feature_type).observe(duration)
        self.features_generated.labels(feature_type=feature_type, status=status).inc(feature_count)
    
    # Cache methods
    def record_cache_hit(self, cache_type: str):
        """Record cache hit"""
        self.cache_hits.labels(cache_type=cache_type).inc()
    
    def record_cache_miss(self, cache_type: str):
        """Record cache miss"""
        self.cache_misses.labels(cache_type=cache_type).inc()
    
    def update_cache_size(self, cache_type: str, size: int):
        """Update cache size"""
        self.cache_size.labels(cache_type=cache_type).set(size)
    
    # Resource monitoring methods
    def update_memory_usage(self, component: str, memory_mb: float):
        """Update memory usage"""
        self.memory_usage_mb.labels(component=component).set(memory_mb)
    
    def update_cpu_usage(self, component: str, cpu_percent: float):
        """Update CPU usage"""
        self.cpu_usage_percent.labels(component=component).set(cpu_percent)
    
    def update_gpu_usage(self, gpu_id: str, gpu_percent: float):
        """Update GPU usage"""
        self.gpu_usage_percent.labels(gpu_id=gpu_id).set(gpu_percent)
    
    # Queue methods
    def update_queue_size(self, queue_type: str, size: int):
        """Update queue size"""
        self.queue_size.labels(queue_type=queue_type).set(size)
    
    def record_queue_processing_time(self, queue_type: str, duration: float):
        """Record queue processing time"""
        self.queue_processing_time.labels(queue_type=queue_type).observe(duration)
    
    # Error methods
    def record_error(self, error_type: str, component: str):
        """Record an error"""
        self.errors_total.labels(error_type=error_type, component=component).inc()
    
    def update_error_rate(self, component: str, error_rate: float):
        """Update error rate"""
        self.error_rate.labels(component=component).set(error_rate)


class PerformanceTimer:
    """Context manager for measuring operation duration"""
    
    def __init__(self, metrics_collector: MetricsCollector, metric_func, *args, **kwargs):
        self.metrics_collector = metrics_collector
        self.metric_func = metric_func
        self.args = args
        self.kwargs = kwargs
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.metric_func(duration, *self.args, **self.kwargs)


class ModelPerformanceTracker:
    """Track ML model performance over time"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.model_stats = {}
    
    def track_prediction(self, model_id: str, model_type: str, actual: float, 
                        predicted: float, confidence: float):
        """Track a single prediction for model performance"""
        if model_id not in self.model_stats:
            self.model_stats[model_id] = {
                'predictions': [],
                'errors': [],
                'confidences': []
            }
        
        error = abs(actual - predicted)
        self.model_stats[model_id]['predictions'].append((actual, predicted))
        self.model_stats[model_id]['errors'].append(error)
        self.model_stats[model_id]['confidences'].append(confidence)
        
        # Calculate and update rolling metrics
        self._update_rolling_metrics(model_id, model_type)
    
    def _update_rolling_metrics(self, model_id: str, model_type: str, window_size: int = 100):
        """Update rolling performance metrics"""
        stats = self.model_stats[model_id]
        
        if len(stats['errors']) >= window_size:
            # Keep only recent predictions
            stats['predictions'] = stats['predictions'][-window_size:]
            stats['errors'] = stats['errors'][-window_size:]
            stats['confidences'] = stats['confidences'][-window_size:]
        
        # Calculate metrics
        if stats['errors']:
            mae = sum(stats['errors']) / len(stats['errors'])
            avg_confidence = sum(stats['confidences']) / len(stats['confidences'])
            
            # Update Prometheus metrics
            self.metrics_collector.update_model_accuracy(model_id, model_type, 1.0 - mae)
            
            logger.info(
                "Model performance updated",
                model_id=model_id,
                model_type=model_type,
                mae=mae,
                avg_confidence=avg_confidence,
                prediction_count=len(stats['errors'])
            )


# Global metrics collector instance
_metrics_collector = None


def setup_metrics() -> MetricsCollector:
    """Setup and return global metrics collector"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = setup_metrics()
    return _metrics_collector


# Utility functions
def time_prediction(model_type: str, prediction_type: str):
    """Decorator to time prediction functions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            collector = get_metrics_collector()
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Extract confidence from result if available
                confidence = 0.0
                if hasattr(result, 'confidence'):
                    confidence = result.confidence
                elif isinstance(result, dict) and 'confidence' in result:
                    confidence = result['confidence']
                
                collector.record_prediction(model_type, prediction_type, duration, confidence)
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                collector.record_prediction(model_type, prediction_type, duration, 0.0, "error")
                collector.record_error(type(e).__name__, "prediction")
                raise
        
        return wrapper
    return decorator


def time_async_prediction(model_type: str, prediction_type: str):
    """Decorator to time async prediction functions"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            collector = get_metrics_collector()
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Extract confidence from result if available
                confidence = 0.0
                if hasattr(result, 'confidence'):
                    confidence = result.confidence
                elif isinstance(result, dict) and 'confidence' in result:
                    confidence = result['confidence']
                
                collector.record_prediction(model_type, prediction_type, duration, confidence)
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                collector.record_prediction(model_type, prediction_type, duration, 0.0, "error")
                collector.record_error(type(e).__name__, "prediction")
                raise
        
        return wrapper
    return decorator


def monitor_resource_usage():
    """Monitor system resource usage"""
    try:
        import psutil
        collector = get_metrics_collector()
        
        # Memory usage
        memory = psutil.virtual_memory()
        collector.update_memory_usage("system", memory.used / 1024 / 1024)
        
        # CPU usage
        cpu_percent = psutil.cpu_percent()
        collector.update_cpu_usage("system", cpu_percent)
        
        # GPU usage (if available)
        try:
            import nvidia_ml_py3 as nvml
            nvml.nvmlInit()
            device_count = nvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = nvml.nvmlDeviceGetHandleByIndex(i)
                info = nvml.nvmlDeviceGetUtilizationRates(handle)
                collector.update_gpu_usage(f"gpu_{i}", info.gpu)
                
        except ImportError:
            pass  # GPU monitoring not available
            
    except Exception as e:
        logger.warning("Failed to monitor resource usage", error=str(e)) 