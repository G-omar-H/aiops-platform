"""
Middleware for AIOps AI Prediction Service
Provides authentication, logging, metrics, and security features
"""

import time
import uuid
from typing import Callable
import structlog
from fastapi import FastAPI, Request, Response, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
import jwt
from prometheus_client import Counter, Histogram, Gauge
import asyncio
from collections import defaultdict


logger = structlog.get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status_code'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'])
ACTIVE_REQUESTS = Gauge('http_requests_active', 'Active HTTP requests')
AUTH_FAILURES = Counter('auth_failures_total', 'Authentication failures', ['reason'])
RATE_LIMIT_HITS = Counter('rate_limit_hits_total', 'Rate limit hits', ['endpoint'])

# Security
security = HTTPBearer(auto_error=False)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting HTTP metrics"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Track active requests
        ACTIVE_REQUESTS.inc()
        
        try:
            # Extract endpoint and method
            method = request.method
            endpoint = request.url.path
            
            # Add request ID
            request_id = str(uuid.uuid4())
            request.state.request_id = request_id
            
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Record metrics
            status_code = str(response.status_code)
            REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
            REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
            
            # Add headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Processing-Time"] = f"{duration:.3f}s"
            
            return response
            
        except Exception as e:
            # Record error metrics
            REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code="500").inc()
            logger.error("Request processing failed", error=str(e), request_id=request_id)
            raise
            
        finally:
            ACTIVE_REQUESTS.dec()


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for structured logging"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Get request details
        method = request.method
        url = str(request.url)
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
        
        # Log request start
        logger.info(
            "Request started",
            method=method,
            url=url,
            client_ip=client_ip,
            user_agent=user_agent,
            request_id=request_id
        )
        
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log request completion
            logger.info(
                "Request completed",
                method=method,
                url=url,
                status_code=response.status_code,
                duration=duration,
                request_id=request_id
            )
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Log error
            logger.error(
                "Request failed",
                method=method,
                url=url,
                error=str(e),
                duration=duration,
                request_id=request_id
            )
            
            # Return error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "request_id": request_id,
                    "timestamp": time.time()
                }
            )


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Middleware for JWT authentication"""
    
    def __init__(self, app: FastAPI, secret_key: str, algorithm: str = "HS256"):
        super().__init__(app)
        self.secret_key = secret_key
        self.algorithm = algorithm
        
        # Public endpoints that don't require authentication
        self.public_endpoints = {
            "/health", "/health/live", "/health/ready",
            "/docs", "/redoc", "/openapi.json", "/metrics"
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip authentication for public endpoints
        if self._is_public_endpoint(request.url.path):
            return await call_next(request)
        
        # Extract token
        authorization = request.headers.get("Authorization")
        if not authorization:
            AUTH_FAILURES.labels(reason="missing_token").inc()
            return self._create_auth_error("Missing authorization header")
        
        try:
            # Parse Bearer token
            scheme, token = authorization.split(" ", 1)
            if scheme.lower() != "bearer":
                AUTH_FAILURES.labels(reason="invalid_scheme").inc()
                return self._create_auth_error("Invalid authorization scheme")
            
            # Verify JWT token
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Add user info to request state
            request.state.user = payload
            request.state.user_id = payload.get("sub")
            
            return await call_next(request)
            
        except jwt.ExpiredSignatureError:
            AUTH_FAILURES.labels(reason="expired_token").inc()
            return self._create_auth_error("Token has expired")
            
        except jwt.InvalidTokenError:
            AUTH_FAILURES.labels(reason="invalid_token").inc()
            return self._create_auth_error("Invalid token")
            
        except ValueError:
            AUTH_FAILURES.labels(reason="malformed_header").inc()
            return self._create_auth_error("Malformed authorization header")
    
    def _is_public_endpoint(self, path: str) -> bool:
        """Check if endpoint is public"""
        return any(path.startswith(endpoint) for endpoint in self.public_endpoints)
    
    def _create_auth_error(self, message: str) -> Response:
        """Create authentication error response"""
        return JSONResponse(
            status_code=401,
            content={
                "error": "Authentication failed",
                "message": message,
                "timestamp": time.time()
            }
        )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting"""
    
    def __init__(self, app: FastAPI, requests_per_minute: int = 100):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.window_size = 60  # 1 minute
        self.request_counts = defaultdict(list)
        self.cleanup_interval = 60  # Cleanup every minute
        self.last_cleanup = time.time()
        
        # Endpoints with different rate limits
        self.endpoint_limits = {
            "/api/v1/predict": 50,
            "/api/v1/detect-anomalies": 30,
            "/api/v1/train": 5,
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get client identifier
        client_id = self._get_client_id(request)
        endpoint = request.url.path
        
        # Get rate limit for this endpoint
        rate_limit = self.endpoint_limits.get(endpoint, self.requests_per_minute)
        
        # Check rate limit
        if self._is_rate_limited(client_id, endpoint, rate_limit):
            RATE_LIMIT_HITS.labels(endpoint=endpoint).inc()
            
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Limit: {rate_limit} per minute",
                    "retry_after": 60,
                    "timestamp": time.time()
                },
                headers={"Retry-After": "60"}
            )
        
        # Record request
        self._record_request(client_id, endpoint)
        
        # Cleanup old entries periodically
        await self._cleanup_if_needed()
        
        return await call_next(request)
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting"""
        # Try to get user ID from authentication
        if hasattr(request.state, 'user_id') and request.state.user_id:
            return f"user:{request.state.user_id}"
        
        # Fall back to IP address
        client_ip = request.client.host if request.client else "unknown"
        return f"ip:{client_ip}"
    
    def _is_rate_limited(self, client_id: str, endpoint: str, limit: int) -> bool:
        """Check if client is rate limited"""
        key = f"{client_id}:{endpoint}"
        current_time = time.time()
        
        # Get request timestamps for this client/endpoint
        timestamps = self.request_counts[key]
        
        # Remove old requests outside the time window
        cutoff_time = current_time - self.window_size
        timestamps[:] = [ts for ts in timestamps if ts > cutoff_time]
        
        # Check if limit exceeded
        return len(timestamps) >= limit
    
    def _record_request(self, client_id: str, endpoint: str):
        """Record a request timestamp"""
        key = f"{client_id}:{endpoint}"
        self.request_counts[key].append(time.time())
    
    async def _cleanup_if_needed(self):
        """Cleanup old request records"""
        current_time = time.time()
        
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        cutoff_time = current_time - self.window_size
        
        # Remove old entries
        for key in list(self.request_counts.keys()):
            timestamps = self.request_counts[key]
            timestamps[:] = [ts for ts in timestamps if ts > cutoff_time]
            
            # Remove empty entries
            if not timestamps:
                del self.request_counts[key]
        
        self.last_cleanup = current_time


class SecurityMiddleware(BaseHTTPMiddleware):
    """Middleware for security headers and protection"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        
        # Remove server header
        if "server" in response.headers:
            del response.headers["server"]
        
        return response


class CompressionMiddleware(BaseHTTPMiddleware):
    """Middleware for response compression"""
    
    def __init__(self, app: FastAPI, minimum_size: int = 1000):
        super().__init__(app)
        self.minimum_size = minimum_size
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Check if compression is supported
        accept_encoding = request.headers.get("accept-encoding", "")
        if "gzip" not in accept_encoding.lower():
            return response
        
        # Check content type
        content_type = response.headers.get("content-type", "")
        if not self._should_compress(content_type):
            return response
        
        # Check content length
        content_length = response.headers.get("content-length")
        if content_length and int(content_length) < self.minimum_size:
            return response
        
        # Add compression headers
        response.headers["Content-Encoding"] = "gzip"
        response.headers["Vary"] = "Accept-Encoding"
        
        return response
    
    def _should_compress(self, content_type: str) -> bool:
        """Check if content type should be compressed"""
        compressible_types = [
            "application/json",
            "application/javascript",
            "text/css",
            "text/html",
            "text/plain",
            "text/xml"
        ]
        
        return any(content_type.startswith(ct) for ct in compressible_types)


class RequestSizeMiddleware(BaseHTTPMiddleware):
    """Middleware to limit request size"""
    
    def __init__(self, app: FastAPI, max_size: int = 10 * 1024 * 1024):  # 10MB default
        super().__init__(app)
        self.max_size = max_size
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check content length
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_size:
            return JSONResponse(
                status_code=413,
                content={
                    "error": "Request too large",
                    "message": f"Request size exceeds maximum of {self.max_size} bytes",
                    "max_size": self.max_size,
                    "timestamp": time.time()
                }
            )
        
        return await call_next(request)


class HealthCheckMiddleware(BaseHTTPMiddleware):
    """Middleware for health check optimization"""
    
    def __init__(self, app: FastAPI):
        super().__init__(app)
        self.health_endpoints = {"/health", "/health/live", "/health/ready"}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip heavy processing for health checks
        if request.url.path in self.health_endpoints:
            # Set minimal logging for health checks
            request.state.skip_detailed_logging = True
        
        return await call_next(request)


# Utility functions for middleware configuration
def setup_middleware(app: FastAPI, settings):
    """Setup all middleware for the application"""
    
    # Health check optimization (should be first)
    app.add_middleware(HealthCheckMiddleware)
    
    # Security middleware
    app.add_middleware(SecurityMiddleware)
    
    # Request size limiting
    app.add_middleware(RequestSizeMiddleware, max_size=settings.max_batch_size * 1024)
    
    # Rate limiting
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=settings.security.rate_limit_requests
    )
    
    # Authentication (if enabled)
    if hasattr(settings.security, 'secret_key') and settings.security.secret_key:
        app.add_middleware(
            AuthenticationMiddleware,
            secret_key=settings.security.secret_key,
            algorithm=settings.security.algorithm
        )
    
    # Compression
    app.add_middleware(CompressionMiddleware)
    
    # Metrics collection
    app.add_middleware(MetricsMiddleware)
    
    # Logging (should be last to capture all processing)
    app.add_middleware(LoggingMiddleware)
    
    # CORS (FastAPI built-in)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.security.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )


# Exception handlers
def setup_exception_handlers(app: FastAPI):
    """Setup custom exception handlers"""
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "status_code": exc.status_code,
                "request_id": request_id,
                "timestamp": time.time()
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
        
        logger.error("Unhandled exception", error=str(exc), request_id=request_id)
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": "An unexpected error occurred",
                "request_id": request_id,
                "timestamp": time.time()
            }
        ) 