package handlers

import (
	"net/http"
	"strconv"
	"time"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"

	"monitoring-service/internal/models"
	"monitoring-service/internal/services"
)

// CollectMetrics handles metrics collection endpoint
func CollectMetrics(metricsService *services.MetricsService) gin.HandlerFunc {
	return func(c *gin.Context) {
		var collection models.MetricsCollection
		if err := c.ShouldBindJSON(&collection); err != nil {
			c.JSON(http.StatusBadRequest, models.ErrorResponse{
				Error:   "Invalid request",
				Message: err.Error(),
				Code:    http.StatusBadRequest,
			})
			return
		}

		if err := metricsService.StoreMetrics(collection.SystemID, collection.Metrics); err != nil {
			c.JSON(http.StatusInternalServerError, models.ErrorResponse{
				Error:   "Failed to store metrics",
				Message: err.Error(),
				Code:    http.StatusInternalServerError,
			})
			return
		}

		c.JSON(http.StatusOK, models.SuccessResponse{
			Success: true,
			Message: "Metrics collected successfully",
		})
	}
}

// GetSystemMetrics retrieves metrics for a specific system
func GetSystemMetrics(metricsService *services.MetricsService) gin.HandlerFunc {
	return func(c *gin.Context) {
		systemID := c.Param("systemId")

		metrics, err := metricsService.GetSystemMetrics(systemID)
		if err != nil {
			c.JSON(http.StatusNotFound, models.ErrorResponse{
				Error:   "System not found",
				Message: err.Error(),
				Code:    http.StatusNotFound,
			})
			return
		}

		c.JSON(http.StatusOK, models.SuccessResponse{
			Success: true,
			Data:    metrics,
		})
	}
}

// GetHistoricalMetrics retrieves historical metrics for a system
func GetHistoricalMetrics(metricsService *services.MetricsService) gin.HandlerFunc {
	return func(c *gin.Context) {
		systemID := c.Param("systemId")
		
		// Parse time range parameters
		startTimeStr := c.Query("start_time")
		endTimeStr := c.Query("end_time")
		
		var startTime, endTime time.Time
		var err error
		
		if startTimeStr != "" {
			startTime, err = time.Parse(time.RFC3339, startTimeStr)
			if err != nil {
				c.JSON(http.StatusBadRequest, models.ErrorResponse{
					Error:   "Invalid start_time format",
					Message: "Use RFC3339 format",
					Code:    http.StatusBadRequest,
				})
				return
			}
		} else {
			startTime = time.Now().Add(-24 * time.Hour)
		}
		
		if endTimeStr != "" {
			endTime, err = time.Parse(time.RFC3339, endTimeStr)
			if err != nil {
				c.JSON(http.StatusBadRequest, models.ErrorResponse{
					Error:   "Invalid end_time format",
					Message: "Use RFC3339 format",
					Code:    http.StatusBadRequest,
				})
				return
			}
		} else {
			endTime = time.Now()
		}

		metrics, err := metricsService.GetHistoricalMetrics(systemID, startTime, endTime)
		if err != nil {
			c.JSON(http.StatusInternalServerError, models.ErrorResponse{
				Error:   "Failed to retrieve historical metrics",
				Message: err.Error(),
				Code:    http.StatusInternalServerError,
			})
			return
		}

		c.JSON(http.StatusOK, models.SuccessResponse{
			Success: true,
			Data:    metrics,
		})
	}
}

// GetAggregatedMetrics retrieves aggregated metrics across systems
func GetAggregatedMetrics(metricsService *services.MetricsService) gin.HandlerFunc {
	return func(c *gin.Context) {
		timeRangeStr := c.DefaultQuery("time_range", "1h")
		timeRange, err := time.ParseDuration(timeRangeStr)
		if err != nil {
			c.JSON(http.StatusBadRequest, models.ErrorResponse{
				Error:   "Invalid time_range format",
				Message: "Use duration format like '1h', '30m', '24h'",
				Code:    http.StatusBadRequest,
			})
			return
		}

		metrics, err := metricsService.GetAggregatedMetrics(timeRange)
		if err != nil {
			c.JSON(http.StatusInternalServerError, models.ErrorResponse{
				Error:   "Failed to retrieve aggregated metrics",
				Message: err.Error(),
				Code:    http.StatusInternalServerError,
			})
			return
		}

		c.JSON(http.StatusOK, models.SuccessResponse{
			Success: true,
			Data:    metrics,
		})
	}
}

// QueryMetrics executes a custom metrics query
func QueryMetrics(metricsService *services.MetricsService) gin.HandlerFunc {
	return func(c *gin.Context) {
		var query models.MetricsQuery
		if err := c.ShouldBindJSON(&query); err != nil {
			c.JSON(http.StatusBadRequest, models.ErrorResponse{
				Error:   "Invalid query request",
				Message: err.Error(),
				Code:    http.StatusBadRequest,
			})
			return
		}

		result, err := metricsService.QueryMetrics(&query)
		if err != nil {
			c.JSON(http.StatusInternalServerError, models.ErrorResponse{
				Error:   "Query execution failed",
				Message: err.Error(),
				Code:    http.StatusInternalServerError,
			})
			return
		}

		c.JSON(http.StatusOK, models.SuccessResponse{
			Success: true,
			Data:    result,
		})
	}
}

// ListSystems returns all registered systems
func ListSystems(metricsService *services.MetricsService) gin.HandlerFunc {
	return func(c *gin.Context) {
		systems, err := metricsService.ListSystems()
		if err != nil {
			c.JSON(http.StatusInternalServerError, models.ErrorResponse{
				Error:   "Failed to list systems",
				Message: err.Error(),
				Code:    http.StatusInternalServerError,
			})
			return
		}

		c.JSON(http.StatusOK, models.SuccessResponse{
			Success: true,
			Data:    systems,
		})
	}
}

// GetSystemDetails retrieves detailed information about a system
func GetSystemDetails(metricsService *services.MetricsService) gin.HandlerFunc {
	return func(c *gin.Context) {
		systemID := c.Param("systemId")

		details, err := metricsService.GetSystemDetails(systemID)
		if err != nil {
			c.JSON(http.StatusNotFound, models.ErrorResponse{
				Error:   "System not found",
				Message: err.Error(),
				Code:    http.StatusNotFound,
			})
			return
		}

		c.JSON(http.StatusOK, models.SuccessResponse{
			Success: true,
			Data:    details,
		})
	}
}

// UpdateSystem updates system information
func UpdateSystem(metricsService *services.MetricsService) gin.HandlerFunc {
	return func(c *gin.Context) {
		systemID := c.Param("systemId")

		var updates models.SystemUpdate
		if err := c.ShouldBindJSON(&updates); err != nil {
			c.JSON(http.StatusBadRequest, models.ErrorResponse{
				Error:   "Invalid update request",
				Message: err.Error(),
				Code:    http.StatusBadRequest,
			})
			return
		}

		if err := metricsService.UpdateSystem(systemID, &updates); err != nil {
			c.JSON(http.StatusInternalServerError, models.ErrorResponse{
				Error:   "Failed to update system",
				Message: err.Error(),
				Code:    http.StatusInternalServerError,
			})
			return
		}

		c.JSON(http.StatusOK, models.SuccessResponse{
			Success: true,
			Message: "System updated successfully",
		})
	}
}

// DeleteSystem removes a system from monitoring
func DeleteSystem(metricsService *services.MetricsService) gin.HandlerFunc {
	return func(c *gin.Context) {
		systemID := c.Param("systemId")

		if err := metricsService.RemoveSystem(systemID); err != nil {
			c.JSON(http.StatusInternalServerError, models.ErrorResponse{
				Error:   "Failed to delete system",
				Message: err.Error(),
				Code:    http.StatusInternalServerError,
			})
			return
		}

		c.JSON(http.StatusOK, models.SuccessResponse{
			Success: true,
			Message: "System deleted successfully",
		})
	}
}

// Alert handlers
func GetAlerts(alertService *services.AlertService) gin.HandlerFunc {
	return func(c *gin.Context) {
		systemID := c.Query("system_id")
		limitStr := c.DefaultQuery("limit", "50")
		
		limit, err := strconv.Atoi(limitStr)
		if err != nil {
			limit = 50
		}

		alerts, err := alertService.ListAlerts(systemID, limit)
		if err != nil {
			c.JSON(http.StatusInternalServerError, models.ErrorResponse{
				Error:   "Failed to retrieve alerts",
				Message: err.Error(),
				Code:    http.StatusInternalServerError,
			})
			return
		}

		c.JSON(http.StatusOK, models.SuccessResponse{
			Success: true,
			Data:    alerts,
		})
	}
}

func CreateAlert(alertService *services.AlertService) gin.HandlerFunc {
	return func(c *gin.Context) {
		var alert models.Alert
		if err := c.ShouldBindJSON(&alert); err != nil {
			c.JSON(http.StatusBadRequest, models.ErrorResponse{
				Error:   "Invalid alert request",
				Message: err.Error(),
				Code:    http.StatusBadRequest,
			})
			return
		}

		if err := alertService.CreateAlert(&alert); err != nil {
			c.JSON(http.StatusInternalServerError, models.ErrorResponse{
				Error:   "Failed to create alert",
				Message: err.Error(),
				Code:    http.StatusInternalServerError,
			})
			return
		}

		c.JSON(http.StatusCreated, models.SuccessResponse{
			Success: true,
			Data:    &alert,
			Message: "Alert created successfully",
		})
	}
}

func UpdateAlert(alertService *services.AlertService) gin.HandlerFunc {
	return func(c *gin.Context) {
		alertID := c.Param("alertId")

		var updates models.AlertUpdate
		if err := c.ShouldBindJSON(&updates); err != nil {
			c.JSON(http.StatusBadRequest, models.ErrorResponse{
				Error:   "Invalid update request",
				Message: err.Error(),
				Code:    http.StatusBadRequest,
			})
			return
		}

		if err := alertService.UpdateAlert(alertID, &updates); err != nil {
			c.JSON(http.StatusInternalServerError, models.ErrorResponse{
				Error:   "Failed to update alert",
				Message: err.Error(),
				Code:    http.StatusInternalServerError,
			})
			return
		}

		c.JSON(http.StatusOK, models.SuccessResponse{
			Success: true,
			Message: "Alert updated successfully",
		})
	}
}

func DeleteAlert(alertService *services.AlertService) gin.HandlerFunc {
	return func(c *gin.Context) {
		alertID := c.Param("alertId")

		if err := alertService.DeleteAlert(alertID); err != nil {
			c.JSON(http.StatusInternalServerError, models.ErrorResponse{
				Error:   "Failed to delete alert",
				Message: err.Error(),
				Code:    http.StatusInternalServerError,
			})
			return
		}

		c.JSON(http.StatusOK, models.SuccessResponse{
			Success: true,
			Message: "Alert deleted successfully",
		})
	}
}

func AcknowledgeAlert(alertService *services.AlertService) gin.HandlerFunc {
	return func(c *gin.Context) {
		alertID := c.Param("alertId")

		if err := alertService.AcknowledgeAlert(alertID); err != nil {
			c.JSON(http.StatusInternalServerError, models.ErrorResponse{
				Error:   "Failed to acknowledge alert",
				Message: err.Error(),
				Code:    http.StatusInternalServerError,
			})
			return
		}

		c.JSON(http.StatusOK, models.SuccessResponse{
			Success: true,
			Message: "Alert acknowledged successfully",
		})
	}
}

// Dashboard handlers (placeholder implementations)
func ListDashboards(metricsService *services.MetricsService) gin.HandlerFunc {
	return func(c *gin.Context) {
		c.JSON(http.StatusOK, models.SuccessResponse{
			Success: true,
			Data:    []models.Dashboard{},
			Message: "Dashboard listing endpoint",
		})
	}
}

func CreateDashboard(metricsService *services.MetricsService) gin.HandlerFunc {
	return func(c *gin.Context) {
		c.JSON(http.StatusCreated, models.SuccessResponse{
			Success: true,
			Message: "Dashboard creation endpoint",
		})
	}
}

func GetDashboard(metricsService *services.MetricsService) gin.HandlerFunc {
	return func(c *gin.Context) {
		c.JSON(http.StatusOK, models.SuccessResponse{
			Success: true,
			Message: "Dashboard retrieval endpoint",
		})
	}
}

func UpdateDashboard(metricsService *services.MetricsService) gin.HandlerFunc {
	return func(c *gin.Context) {
		c.JSON(http.StatusOK, models.SuccessResponse{
			Success: true,
			Message: "Dashboard update endpoint",
		})
	}
}

func DeleteDashboard(metricsService *services.MetricsService) gin.HandlerFunc {
	return func(c *gin.Context) {
		c.JSON(http.StatusOK, models.SuccessResponse{
			Success: true,
			Message: "Dashboard deletion endpoint",
		})
	}
} 