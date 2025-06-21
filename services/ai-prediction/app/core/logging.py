"""
Logging Configuration for AI Prediction Service

Provides structured logging with JSON formatting and integration with observability tools.
"""

import logging
import logging.config
import sys
from datetime import datetime
from typing import Dict, Any

import structlog
from pythonjsonlogger import jsonlogger

from app.core.config import get_settings

settings = get_settings()


class CustomJSONFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional fields."""
    
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        super().add_fields(log_record, record, message_dict)
        
        # Add timestamp
        log_record['timestamp'] = datetime.utcnow().isoformat()
        
        # Add service information
        log_record['service'] = 'ai-prediction'
        log_record['version'] = settings.APP_VERSION
        
        # Add level name
        log_record['level'] = record.levelname
        
        # Add correlation ID if available
        if hasattr(record, 'correlation_id'):
            log_record['correlation_id'] = record.correlation_id
        
        # Add user ID if available
        if hasattr(record, 'user_id'):
            log_record['user_id'] = record.user_id


def setup_logging() -> None:
    """Configure structured logging for the application."""
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Logging configuration
    log_level = getattr(logging, settings.LOG_LEVEL.upper())
    
    if settings.LOG_FORMAT.lower() == 'json':
        formatter = CustomJSONFormatter(
            '%(timestamp)s %(level)s %(name)s %(message)s'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    handlers = [console_handler]
    
    # File handler (if specified)
    if settings.LOG_FILE:
        file_handler = logging.FileHandler(settings.LOG_FILE)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Root logger configuration
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        format='%(message)s'
    )
    
    # Set specific logger levels
    logger_levels = {
        'uvicorn': logging.INFO,
        'uvicorn.access': logging.INFO if settings.DEBUG else logging.WARNING,
        'sqlalchemy.engine': logging.INFO if settings.DEBUG else logging.WARNING,
        'aioredis': logging.INFO if settings.DEBUG else logging.WARNING,
        'torch': logging.WARNING,
        'tensorflow': logging.WARNING,
        'matplotlib': logging.WARNING
    }
    
    for logger_name, level in logger_levels.items():
        logging.getLogger(logger_name).setLevel(level)
    
    # Get logger for this module
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured - Level: {settings.LOG_LEVEL}, Format: {settings.LOG_FORMAT}")


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance."""
    return logging.getLogger(name)


class LoggingMiddleware:
    """Middleware for request/response logging."""
    
    def __init__(self, app):
        self.app = app
        self.logger = get_logger(__name__)
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            request_id = scope.get("headers", {}).get("x-request-id", "unknown")
            
            # Log request
            self.logger.info(
                "Request started",
                extra={
                    "request_id": request_id,
                    "method": scope["method"],
                    "path": scope["path"],
                    "query_string": scope.get("query_string", b"").decode()
                }
            )
        
        await self.app(scope, receive, send) 