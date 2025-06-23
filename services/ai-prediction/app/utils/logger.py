"""
Structured logging utility for AIOps AI Prediction Service
Provides enterprise-grade logging with correlation IDs, structured output, and monitoring integration
"""

import logging
import sys
from typing import Any, Dict, Optional
import structlog
from structlog.types import EventDict, WrappedLogger
import json
import traceback
from datetime import datetime


class CorrelationIDFilter(logging.Filter):
    """Filter to add correlation ID to log records"""
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Add correlation ID if available
        if not hasattr(record, 'correlation_id'):
            record.correlation_id = getattr(record, 'request_id', 'unknown')
        return True


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add correlation ID if available
        if hasattr(record, 'correlation_id'):
            log_entry['correlation_id'] = record.correlation_id
        
        # Add request ID if available
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        
        # Add user ID if available
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ('name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'message',
                          'correlation_id', 'request_id', 'user_id'):
                if not key.startswith('_'):
                    log_entry[key] = value
        
        return json.dumps(log_entry, default=str, ensure_ascii=False)


def add_correlation_id(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    """Add correlation ID to log events"""
    import contextvars
    
    # Try to get correlation ID from context
    try:
        correlation_id = contextvars.copy_context().get('correlation_id', 'unknown')
        event_dict['correlation_id'] = correlation_id
    except:
        pass
    
    return event_dict


def add_service_info(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    """Add service information to log events"""
    event_dict.update({
        'service': 'ai-prediction',
        'component': 'ml-engine',
        'version': '1.0.0'
    })
    return event_dict


def add_performance_metrics(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    """Add performance metrics to log events"""
    import psutil
    import time
    
    try:
        # Add basic performance metrics
        process = psutil.Process()
        event_dict.update({
            'memory_mb': round(process.memory_info().rss / 1024 / 1024, 2),
            'cpu_percent': process.cpu_percent(),
            'timestamp_ms': int(time.time() * 1000)
        })
    except:
        pass
    
    return event_dict


def filter_sensitive_data(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    """Filter sensitive data from log events"""
    sensitive_keys = {'password', 'token', 'api_key', 'secret', 'auth', 'credential'}
    
    def mask_value(key: str, value: Any) -> Any:
        if isinstance(key, str) and any(sensitive in key.lower() for sensitive in sensitive_keys):
            return '***MASKED***'
        elif isinstance(value, dict):
            return {k: mask_value(k, v) for k, v in value.items()}
        elif isinstance(value, list):
            return [mask_value(f"item_{i}", item) for i, item in enumerate(value)]
        return value
    
    # Recursively mask sensitive data
    for key, value in list(event_dict.items()):
        event_dict[key] = mask_value(key, value)
    
    return event_dict


def setup_logging(settings):
    """Setup structured logging configuration"""
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            add_correlation_id,
            add_service_info,
            add_performance_metrics,
            filter_sensitive_data,
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.monitoring.log_level.upper())
    )
    
    # Configure specific loggers
    configure_logger('uvicorn', settings.monitoring.log_level)
    configure_logger('uvicorn.access', 'WARNING')  # Reduce access log noise
    configure_logger('fastapi', settings.monitoring.log_level)
    configure_logger('sqlalchemy', 'WARNING')  # Reduce SQL noise
    
    # Add custom filter for correlation IDs
    correlation_filter = CorrelationIDFilter()
    
    # Get root logger and add filter
    root_logger = logging.getLogger()
    root_logger.addFilter(correlation_filter)
    
    # Setup file logging if specified
    if settings.monitoring.log_file:
        setup_file_logging(settings.monitoring.log_file, settings.monitoring.log_level)


def configure_logger(name: str, level: str):
    """Configure a specific logger"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create handler with JSON formatter
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    logger.propagate = False


def setup_file_logging(log_file: str, log_level: str):
    """Setup file-based logging"""
    import os
    from logging.handlers import RotatingFileHandler
    
    # Ensure log directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Create rotating file handler
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=100 * 1024 * 1024,  # 100MB
        backupCount=5
    )
    file_handler.setFormatter(JSONFormatter())
    file_handler.setLevel(getattr(logging, log_level.upper()))
    
    # Add to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)


class LoggerMixin:
    """Mixin class to add structured logging to any class"""
    
    @property
    def logger(self):
        """Get a logger for this class"""
        if not hasattr(self, '_logger'):
            self._logger = structlog.get_logger(self.__class__.__name__)
        return self._logger
    
    def log_method_entry(self, method_name: str, **kwargs):
        """Log method entry with parameters"""
        self.logger.debug(
            "Method entry",
            method=method_name,
            parameters=kwargs
        )
    
    def log_method_exit(self, method_name: str, result=None, duration: Optional[float] = None):
        """Log method exit with result and duration"""
        log_data = {"method": method_name}
        
        if result is not None:
            log_data["result_type"] = type(result).__name__
            if hasattr(result, '__len__'):
                log_data["result_length"] = len(result)
        
        if duration is not None:
            log_data["duration_ms"] = round(duration * 1000, 2)
        
        self.logger.debug("Method exit", **log_data)
    
    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Log an error with context"""
        log_data = {
            "error_type": type(error).__name__,
            "error_message": str(error)
        }
        
        if context:
            log_data.update(context)
        
        self.logger.error("Error occurred", **log_data)


class PerformanceLogger:
    """Context manager for performance logging"""
    
    def __init__(self, logger, operation_name: str, **context):
        self.logger = logger
        self.operation_name = operation_name
        self.context = context
        self.start_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        self.logger.info(
            "Operation started",
            operation=self.operation_name,
            **self.context
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        duration = time.time() - self.start_time
        
        if exc_type is None:
            self.logger.info(
                "Operation completed",
                operation=self.operation_name,
                duration_ms=round(duration * 1000, 2),
                **self.context
            )
        else:
            self.logger.error(
                "Operation failed",
                operation=self.operation_name,
                duration_ms=round(duration * 1000, 2),
                error_type=exc_type.__name__,
                error_message=str(exc_val),
                **self.context
            )


def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
    """Get a structured logger instance"""
    return structlog.get_logger(name)


def log_prediction_request(system_id: str, prediction_type: str, metrics_count: int):
    """Log a prediction request"""
    logger = get_logger("prediction")
    logger.info(
        "Prediction request received",
        system_id=system_id,
        prediction_type=prediction_type,
        metrics_count=metrics_count,
        event_type="prediction_request"
    )


def log_prediction_response(system_id: str, prediction_type: str, confidence: float, 
                          processing_time_ms: float, model_used: str):
    """Log a prediction response"""
    logger = get_logger("prediction")
    logger.info(
        "Prediction completed",
        system_id=system_id,
        prediction_type=prediction_type,
        confidence=confidence,
        processing_time_ms=processing_time_ms,
        model_used=model_used,
        event_type="prediction_response"
    )


def log_model_training(model_type: str, job_id: str, status: str, **kwargs):
    """Log model training events"""
    logger = get_logger("training")
    logger.info(
        "Model training event",
        model_type=model_type,
        job_id=job_id,
        status=status,
        event_type="model_training",
        **kwargs
    )


def log_anomaly_detection(system_id: str, anomalies_count: int, processing_time_ms: float):
    """Log anomaly detection results"""
    logger = get_logger("anomaly")
    logger.info(
        "Anomaly detection completed",
        system_id=system_id,
        anomalies_count=anomalies_count,
        processing_time_ms=processing_time_ms,
        event_type="anomaly_detection"
    )


def log_health_check(component: str, status: str, **metrics):
    """Log health check results"""
    logger = get_logger("health")
    logger.info(
        "Health check performed",
        component=component,
        status=status,
        event_type="health_check",
        **metrics
    )


# Decorators for automatic logging
def log_execution_time(operation_name: Optional[str] = None):
    """Decorator to log execution time of functions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            import functools
            
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            logger = get_logger("performance")
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                logger.info(
                    "Function executed",
                    operation=op_name,
                    duration_ms=round(duration * 1000, 2),
                    status="success"
                )
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    "Function failed",
                    operation=op_name,
                    duration_ms=round(duration * 1000, 2),
                    error_type=type(e).__name__,
                    error_message=str(e),
                    status="error"
                )
                raise
        
        return wrapper
    return decorator


def log_async_execution_time(operation_name: Optional[str] = None):
    """Decorator to log execution time of async functions"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            import time
            import functools
            
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            logger = get_logger("performance")
            
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                logger.info(
                    "Async function executed",
                    operation=op_name,
                    duration_ms=round(duration * 1000, 2),
                    status="success"
                )
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    "Async function failed",
                    operation=op_name,
                    duration_ms=round(duration * 1000, 2),
                    error_type=type(e).__name__,
                    error_message=str(e),
                    status="error"
                )
                raise
        
        return wrapper
    return decorator 