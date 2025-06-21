package middleware

import (
	"fmt"
	"net/http"
	"runtime/debug"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"golang.org/x/time/rate"
	"aiops-platform/services/api-gateway/internal/utils"
)

// RequestID middleware adds a unique request ID to each request
func RequestID() gin.HandlerFunc {
	return func(c *gin.Context) {
		// Check if request ID already exists in headers
		requestID := c.GetHeader("X-Request-ID")
		if requestID == "" {
			// Generate new UUID if not provided
			requestID = uuid.New().String()
		}

		// Set request ID in context and response header
		c.Set("request_id", requestID)
		c.Header("X-Request-ID", requestID)

		c.Next()
	}
}

// Logger middleware logs HTTP requests
func Logger(logger utils.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		path := c.Request.URL.Path
		raw := c.Request.URL.RawQuery

		// Process request
		c.Next()

		// Calculate latency
		latency := time.Since(start)

		// Get request ID
		requestID, _ := c.Get("request_id")

		// Get user ID if available
		userID, _ := c.Get("user_id")

		// Log request details
		logger.Info("HTTP Request",
			"request_id", requestID,
			"user_id", userID,
			"method", c.Request.Method,
			"path", path,
			"query", raw,
			"status", c.Writer.Status(),
			"latency", latency,
			"ip", c.ClientIP(),
			"user_agent", c.Request.UserAgent(),
			"size", c.Writer.Size(),
		)

		// Log errors if status >= 400
		if c.Writer.Status() >= 400 {
			errors := c.Errors.String()
			logger.Error("HTTP Error",
				"request_id", requestID,
				"user_id", userID,
				"method", c.Request.Method,
				"path", path,
				"status", c.Writer.Status(),
				"errors", errors,
				"ip", c.ClientIP(),
			)
		}
	}
}

// Recovery middleware handles panics and returns appropriate error responses
func Recovery(logger utils.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		defer func() {
			if err := recover(); err != nil {
				// Get request ID
				requestID, _ := c.Get("request_id")

				// Log the panic
				logger.Error("Panic recovered",
					"request_id", requestID,
					"error", err,
					"stack", string(debug.Stack()),
					"method", c.Request.Method,
					"path", c.Request.URL.Path,
					"ip", c.ClientIP(),
				)

				// Return error response
				c.JSON(http.StatusInternalServerError, gin.H{
					"error":      "Internal Server Error",
					"message":    "An unexpected error occurred",
					"request_id": requestID,
					"code":       "INTERNAL_ERROR",
				})

				c.Abort()
			}
		}()

		c.Next()
	}
}

// SecurityHeaders middleware adds security headers to responses
func SecurityHeaders() gin.HandlerFunc {
	return func(c *gin.Context) {
		// X-Content-Type-Options
		c.Header("X-Content-Type-Options", "nosniff")

		// X-Frame-Options
		c.Header("X-Frame-Options", "DENY")

		// X-XSS-Protection
		c.Header("X-XSS-Protection", "1; mode=block")

		// Strict-Transport-Security (HSTS)
		c.Header("Strict-Transport-Security", "max-age=31536000; includeSubDomains")

		// Content-Security-Policy
		c.Header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'")

		// Referrer-Policy
		c.Header("Referrer-Policy", "strict-origin-when-cross-origin")

		// Permissions-Policy
		c.Header("Permissions-Policy", "geolocation=(), microphone=(), camera=()")

		// Remove server header
		c.Header("Server", "")

		c.Next()
	}
}

// RateLimitConfig holds rate limiting configuration
type RateLimitConfig struct {
	RequestsPerMinute int
	Burst             int
}

// RateLimiter holds rate limiting state
type RateLimiter struct {
	limiters map[string]*rate.Limiter
	mu       sync.RWMutex
	config   RateLimitConfig
}

// NewRateLimiter creates a new rate limiter
func NewRateLimiter(config RateLimitConfig) *RateLimiter {
	return &RateLimiter{
		limiters: make(map[string]*rate.Limiter),
		config:   config,
	}
}

// GetLimiter returns a rate limiter for the given key
func (rl *RateLimiter) GetLimiter(key string) *rate.Limiter {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	limiter, exists := rl.limiters[key]
	if !exists {
		limiter = rate.NewLimiter(rate.Every(time.Minute/time.Duration(rl.config.RequestsPerMinute)), rl.config.Burst)
		rl.limiters[key] = limiter

		// Clean up old limiters (simple cleanup strategy)
		if len(rl.limiters) > 1000 {
			// Remove half of the limiters (simple cleanup)
			count := 0
			for k := range rl.limiters {
				if count > 500 {
					break
				}
				delete(rl.limiters, k)
				count++
			}
		}
	}

	return limiter
}

// RateLimit middleware implements rate limiting per IP address
func RateLimit(config RateLimitConfig) gin.HandlerFunc {
	rateLimiter := NewRateLimiter(config)

	return func(c *gin.Context) {
		// Use IP address as the key for rate limiting
		key := c.ClientIP()

		// Get user ID if authenticated for per-user rate limiting
		if userID, exists := c.Get("user_id"); exists {
			key = fmt.Sprintf("user_%s", userID)
		}

		limiter := rateLimiter.GetLimiter(key)

		if !limiter.Allow() {
			c.JSON(http.StatusTooManyRequests, gin.H{
				"error":   "Too Many Requests",
				"message": "Rate limit exceeded. Please try again later.",
				"code":    "RATE_LIMIT_EXCEEDED",
				"retry_after": 60, // seconds
			})
			c.Abort()
			return
		}

		c.Next()
	}
}

// CORS middleware handles Cross-Origin Resource Sharing
func CORS(allowedOrigins []string, allowedMethods []string, allowedHeaders []string) gin.HandlerFunc {
	return func(c *gin.Context) {
		origin := c.Request.Header.Get("Origin")

		// Check if origin is allowed
		allowed := false
		for _, allowedOrigin := range allowedOrigins {
			if allowedOrigin == "*" || allowedOrigin == origin {
				allowed = true
				break
			}
		}

		if allowed {
			c.Header("Access-Control-Allow-Origin", origin)
		}

		c.Header("Access-Control-Allow-Credentials", "true")
		c.Header("Access-Control-Allow-Headers", "Content-Type, Content-Length, Accept-Encoding, X-CSRF-Token, Authorization, accept, origin, Cache-Control, X-Requested-With, X-Request-ID")
		c.Header("Access-Control-Allow-Methods", "POST, OPTIONS, GET, PUT, DELETE, PATCH")

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}

		c.Next()
	}
}

// Timeout middleware adds request timeout
func Timeout(timeout time.Duration) gin.HandlerFunc {
	return func(c *gin.Context) {
		// Create a context with timeout
		ctx, cancel := c.Request.Context(), func() {}
		if timeout > 0 {
			ctx, cancel = c.Request.Context().WithTimeout(c.Request.Context(), timeout)
		}
		defer cancel()

		// Replace request context
		c.Request = c.Request.WithContext(ctx)

		// Channel to signal completion
		finished := make(chan bool, 1)

		go func() {
			c.Next()
			finished <- true
		}()

		select {
		case <-finished:
			// Request completed normally
		case <-ctx.Done():
			// Request timed out
			c.JSON(http.StatusRequestTimeout, gin.H{
				"error":   "Request Timeout",
				"message": "Request took too long to process",
				"code":    "REQUEST_TIMEOUT",
			})
			c.Abort()
		}
	}
}

// APIVersion middleware adds API version to response headers
func APIVersion(version string) gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Header("X-API-Version", version)
		c.Next()
	}
}

// ContentType middleware validates content type for POST/PUT requests
func ContentType(allowedTypes ...string) gin.HandlerFunc {
	return func(c *gin.Context) {
		if c.Request.Method == "POST" || c.Request.Method == "PUT" || c.Request.Method == "PATCH" {
			contentType := c.GetHeader("Content-Type")

			allowed := false
			for _, allowedType := range allowedTypes {
				if contentType == allowedType {
					allowed = true
					break
				}
			}

			if !allowed {
				c.JSON(http.StatusUnsupportedMediaType, gin.H{
					"error":   "Unsupported Media Type",
					"message": fmt.Sprintf("Content-Type must be one of: %v", allowedTypes),
					"code":    "UNSUPPORTED_MEDIA_TYPE",
				})
				c.Abort()
				return
			}
		}

		c.Next()
	}
}

// RequestSize middleware limits request body size
func RequestSize(maxSize int64) gin.HandlerFunc {
	return func(c *gin.Context) {
		if c.Request.ContentLength > maxSize {
			c.JSON(http.StatusRequestEntityTooLarge, gin.H{
				"error":   "Request Entity Too Large",
				"message": fmt.Sprintf("Request body too large. Maximum size: %d bytes", maxSize),
				"code":    "REQUEST_TOO_LARGE",
			})
			c.Abort()
			return
		}

		c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, maxSize)
		c.Next()
	}
}

// Health check middleware for load balancers
func HealthCheck(healthPath string) gin.HandlerFunc {
	return func(c *gin.Context) {
		if c.Request.URL.Path == healthPath {
			c.JSON(http.StatusOK, gin.H{
				"status":    "healthy",
				"timestamp": time.Now().UTC().Format(time.RFC3339),
				"version":   "1.0.0",
			})
			c.Abort()
			return
		}

		c.Next()
	}
}

// Metrics middleware for Prometheus
func Metrics() gin.HandlerFunc {
	// This would integrate with Prometheus metrics
	// For now, it's a placeholder
	return func(c *gin.Context) {
		start := time.Now()

		c.Next()

		// Record metrics here
		duration := time.Since(start)
		status := c.Writer.Status()
		method := c.Request.Method
		route := c.FullPath()

		// TODO: Implement Prometheus metrics recording
		_ = duration
		_ = status
		_ = method
		_ = route
	}
} 