package middleware

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/golang-jwt/jwt/v5"
	"github.com/prometheus/client_golang/prometheus"
	"go.uber.org/zap"
	"golang.org/x/time/rate"
)

var (
	// Metrics for middleware
	httpRequestsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "http_requests_total",
			Help: "Total number of HTTP requests",
		},
		[]string{"method", "endpoint", "status_code"},
	)

	httpRequestDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "http_request_duration_seconds",
			Help:    "HTTP request latency",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"method", "endpoint"},
	)

	activeConnections = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Name: "http_active_connections",
			Help: "Number of active HTTP connections",
		},
	)
)

func init() {
	prometheus.MustRegister(httpRequestsTotal)
	prometheus.MustRegister(httpRequestDuration)
	prometheus.MustRegister(activeConnections)
}

// Logger middleware for structured logging
func Logger(logger *zap.Logger) gin.HandlerFunc {
	return gin.LoggerWithFormatter(func(param gin.LogFormatterParams) string {
		logger.Info("HTTP Request",
			zap.String("method", param.Method),
			zap.String("path", param.Path),
			zap.Int("status", param.StatusCode),
			zap.Duration("latency", param.Latency),
			zap.String("client_ip", param.ClientIP),
			zap.String("user_agent", param.Request.UserAgent()),
			zap.String("request_id", param.Request.Header.Get("X-Request-ID")),
		)
		return ""
	})
}

// Recovery middleware with structured error logging
func Recovery(logger *zap.Logger) gin.HandlerFunc {
	return gin.CustomRecovery(func(c *gin.Context, recovered interface{}) {
		logger.Error("Panic recovered",
			zap.Any("error", recovered),
			zap.String("method", c.Request.Method),
			zap.String("path", c.Request.URL.Path),
			zap.String("client_ip", c.ClientIP()),
			zap.String("request_id", c.Request.Header.Get("X-Request-ID")),
		)
		c.AbortWithStatus(http.StatusInternalServerError)
	})
}

// CORS middleware for cross-origin requests
func CORS() gin.HandlerFunc {
	return func(c *gin.Context) {
		origin := c.Request.Header.Get("Origin")
		
		// Allow specific origins or all origins in development
		if origin == "" {
			origin = "*"
		}

		c.Header("Access-Control-Allow-Origin", origin)
		c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, PATCH, OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Origin, Content-Type, Content-Length, Accept-Encoding, X-CSRF-Token, Authorization, X-Request-ID, X-API-Key")
		c.Header("Access-Control-Expose-Headers", "Content-Length, X-Request-ID, X-Rate-Limit-Remaining")
		c.Header("Access-Control-Allow-Credentials", "true")
		c.Header("Access-Control-Max-Age", "86400")

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(http.StatusNoContent)
			return
		}

		c.Next()
	}
}

// Security middleware for HTTP security headers
func Security() gin.HandlerFunc {
	return func(c *gin.Context) {
		// Security headers
		c.Header("X-Content-Type-Options", "nosniff")
		c.Header("X-Frame-Options", "DENY")
		c.Header("X-XSS-Protection", "1; mode=block")
		c.Header("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
		c.Header("Content-Security-Policy", "default-src 'self'")
		c.Header("Referrer-Policy", "strict-origin-when-cross-origin")
		c.Header("Permissions-Policy", "geolocation=(), microphone=(), camera=()")

		// Remove server information
		c.Header("Server", "")

		c.Next()
	}
}

// Metrics middleware for Prometheus metrics collection
func Metrics() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		activeConnections.Inc()
		defer activeConnections.Dec()

		c.Next()

		duration := time.Since(start)
		status := strconv.Itoa(c.Writer.Status())

		httpRequestsTotal.WithLabelValues(
			c.Request.Method,
			c.FullPath(),
			status,
		).Inc()

		httpRequestDuration.WithLabelValues(
			c.Request.Method,
			c.FullPath(),
		).Observe(duration.Seconds())
	}
}

// RequestID middleware adds unique request ID to each request
func RequestID() gin.HandlerFunc {
	return func(c *gin.Context) {
		requestID := c.Request.Header.Get("X-Request-ID")
		if requestID == "" {
			// Generate request ID based on timestamp and client IP
			hash := sha256.Sum256([]byte(fmt.Sprintf("%d-%s", time.Now().UnixNano(), c.ClientIP())))
			requestID = hex.EncodeToString(hash[:])[:16]
		}

		c.Header("X-Request-ID", requestID)
		c.Set("request_id", requestID)
		c.Next()
	}
}

// RateLimit middleware for API rate limiting
func RateLimit(rps rate.Limit, burst int) gin.HandlerFunc {
	limiter := rate.NewLimiter(rps, burst)

	return func(c *gin.Context) {
		if !limiter.Allow() {
			c.Header("X-Rate-Limit-Remaining", "0")
			c.Header("Retry-After", "1")
			c.JSON(http.StatusTooManyRequests, gin.H{
				"error":   "Rate limit exceeded",
				"message": "Too many requests, please try again later",
			})
			c.Abort()
			return
		}

		// Set rate limit headers
		c.Header("X-Rate-Limit-Remaining", fmt.Sprintf("%d", burst-1))
		c.Next()
	}
}

// Authentication middleware for JWT token validation
func Authentication(jwtSecret string) gin.HandlerFunc {
	return func(c *gin.Context) {
		// Skip authentication for public endpoints
		if isPublicEndpoint(c.Request.URL.Path) {
			c.Next()
			return
		}

		token := extractToken(c.Request)
		if token == "" {
			c.JSON(http.StatusUnauthorized, gin.H{
				"error":   "Missing authentication token",
				"message": "Authorization header with Bearer token is required",
			})
			c.Abort()
			return
		}

		claims, err := validateJWT(token, jwtSecret)
		if err != nil {
			c.JSON(http.StatusUnauthorized, gin.H{
				"error":   "Invalid authentication token",
				"message": err.Error(),
			})
			c.Abort()
			return
		}

		// Set user context
		c.Set("user_id", claims["user_id"])
		c.Set("user_role", claims["role"])
		c.Set("user_permissions", claims["permissions"])
		c.Next()
	}
}

// RequireRole middleware for role-based access control
func RequireRole(requiredRole string) gin.HandlerFunc {
	return func(c *gin.Context) {
		userRole, exists := c.Get("user_role")
		if !exists {
			c.JSON(http.StatusForbidden, gin.H{
				"error":   "Access denied",
				"message": "User role not found in context",
			})
			c.Abort()
			return
		}

		if userRole != requiredRole && userRole != "admin" {
			c.JSON(http.StatusForbidden, gin.H{
				"error":   "Access denied",
				"message": fmt.Sprintf("Required role: %s", requiredRole),
			})
			c.Abort()
			return
		}

		c.Next()
	}
}

// RequirePermission middleware for permission-based access control
func RequirePermission(requiredPermission string) gin.HandlerFunc {
	return func(c *gin.Context) {
		permissions, exists := c.Get("user_permissions")
		if !exists {
			c.JSON(http.StatusForbidden, gin.H{
				"error":   "Access denied",
				"message": "User permissions not found in context",
			})
			c.Abort()
			return
		}

		permissionsList, ok := permissions.([]interface{})
		if !ok {
			c.JSON(http.StatusForbidden, gin.H{
				"error":   "Access denied",
				"message": "Invalid permissions format",
			})
			c.Abort()
			return
		}

		hasPermission := false
		for _, perm := range permissionsList {
			if perm == requiredPermission || perm == "*" {
				hasPermission = true
				break
			}
		}

		if !hasPermission {
			c.JSON(http.StatusForbidden, gin.H{
				"error":   "Access denied",
				"message": fmt.Sprintf("Required permission: %s", requiredPermission),
			})
			c.Abort()
			return
		}

		c.Next()
	}
}

// APIKeyAuth middleware for API key authentication
func APIKeyAuth(validAPIKeys map[string]string) gin.HandlerFunc {
	return func(c *gin.Context) {
		apiKey := c.Request.Header.Get("X-API-Key")
		if apiKey == "" {
			c.JSON(http.StatusUnauthorized, gin.H{
				"error":   "Missing API key",
				"message": "X-API-Key header is required",
			})
			c.Abort()
			return
		}

		// Validate API key
		if _, exists := validAPIKeys[apiKey]; !exists {
			c.JSON(http.StatusUnauthorized, gin.H{
				"error":   "Invalid API key",
				"message": "The provided API key is not valid",
			})
			c.Abort()
			return
		}

		// Set API key context
		c.Set("api_key", apiKey)
		c.Set("api_client", validAPIKeys[apiKey])
		c.Next()
	}
}

// Timeout middleware for request timeout handling
func Timeout(timeout time.Duration) gin.HandlerFunc {
	return func(c *gin.Context) {
		ctx := c.Request.Context()
		timeoutCtx, cancel := context.WithTimeout(ctx, timeout)
		defer cancel()

		c.Request = c.Request.WithContext(timeoutCtx)
		c.Next()
	}
}

// Compression middleware for response compression
func Compression() gin.HandlerFunc {
	return func(c *gin.Context) {
		// Check if client accepts gzip
		if !strings.Contains(c.Request.Header.Get("Accept-Encoding"), "gzip") {
			c.Next()
			return
		}

		// Check if response should be compressed
		if shouldCompress(c.Request.URL.Path) {
			c.Header("Content-Encoding", "gzip")
			c.Header("Vary", "Accept-Encoding")
		}

		c.Next()
	}
}

// BodySizeLimit middleware for limiting request body size
func BodySizeLimit(maxBytes int64) gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, maxBytes)
		c.Next()
	}
}

// Helper functions

func isPublicEndpoint(path string) bool {
	publicPaths := []string{
		"/health",
		"/health/live",
		"/health/ready",
		"/metrics",
		"/docs",
		"/ping",
	}

	for _, publicPath := range publicPaths {
		if strings.HasPrefix(path, publicPath) {
			return true
		}
	}
	return false
}

func extractToken(r *http.Request) string {
	// Check Authorization header
	authHeader := r.Header.Get("Authorization")
	if strings.HasPrefix(authHeader, "Bearer ") {
		return strings.TrimPrefix(authHeader, "Bearer ")
	}

	// Check query parameter
	return r.URL.Query().Get("token")
}

func validateJWT(tokenString, jwtSecret string) (jwt.MapClaims, error) {
	token, err := jwt.Parse(tokenString, func(token *jwt.Token) (interface{}, error) {
		if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
		}
		return []byte(jwtSecret), nil
	})

	if err != nil {
		return nil, err
	}

	if claims, ok := token.Claims.(jwt.MapClaims); ok && token.Valid {
		// Check token expiration
		if exp, ok := claims["exp"].(float64); ok {
			if time.Now().Unix() > int64(exp) {
				return nil, fmt.Errorf("token has expired")
			}
		}
		return claims, nil
	}

	return nil, fmt.Errorf("invalid token")
}

func shouldCompress(path string) bool {
	compressiblePaths := []string{
		"/api/",
		"/docs/",
	}

	for _, compressiblePath := range compressiblePaths {
		if strings.HasPrefix(path, compressiblePath) {
			return true
		}
	}
	return false
} 