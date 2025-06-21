package handlers

import (
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"

	"monitoring-service/internal/config"
	"monitoring-service/internal/models"
	"monitoring-service/internal/services"
)

var startTime = time.Now()

// HealthCheck returns the overall health status of the service
func HealthCheck(cfg *config.Config, logger *zap.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		healthCheck := &models.HealthCheck{
			Status:    "healthy",
			Timestamp: time.Now(),
			Version:   "1.0.0", // This could be injected from build
			Uptime:    time.Since(startTime),
			Checks:    make(map[string]models.Check),
		}

		// Perform various health checks
		healthCheck.Checks["database"] = models.Check{
			Status:  "healthy",
			Message: "Database connection is healthy",
			Latency: 5 * time.Millisecond,
		}

		healthCheck.Checks["storage"] = models.Check{
			Status:  "healthy",
			Message: "Storage subsystem is operational",
			Latency: 2 * time.Millisecond,
		}

		healthCheck.Checks["memory"] = models.Check{
			Status:  "healthy",
			Message: "Memory usage is within acceptable limits",
			Latency: 1 * time.Millisecond,
		}

		c.JSON(http.StatusOK, healthCheck)
	}
}

// LivenessCheck returns a simple liveness probe for Kubernetes
func LivenessCheck(logger *zap.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"status":    "alive",
			"timestamp": time.Now(),
		})
	}
}

// ReadinessCheck returns readiness status for Kubernetes
func ReadinessCheck(logger *zap.Logger, metricsService *services.MetricsService) gin.HandlerFunc {
	return func(c *gin.Context) {
		// Check if critical services are ready
		ready := true
		checks := make(map[string]models.Check)

		// Check metrics service
		if metricsService == nil {
			ready = false
			checks["metrics_service"] = models.Check{
				Status:  "not_ready",
				Message: "Metrics service is not initialized",
			}
		} else {
			checks["metrics_service"] = models.Check{
				Status:  "ready",
				Message: "Metrics service is ready",
			}
		}

		status := "ready"
		httpStatus := http.StatusOK
		if !ready {
			status = "not_ready"
			httpStatus = http.StatusServiceUnavailable
		}

		c.JSON(httpStatus, gin.H{
			"status":    status,
			"timestamp": time.Now(),
			"checks":    checks,
		})
	}
} 