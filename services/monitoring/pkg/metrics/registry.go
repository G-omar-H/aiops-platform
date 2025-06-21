package metrics

import (
	"sync"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// Registry holds all application metrics
type Registry struct {
	metricsCollected *prometheus.CounterVec
	alertsTriggered  *prometheus.CounterVec
	systemsMonitored prometheus.Gauge
	httpRequests     *prometheus.CounterVec
	httpDuration     *prometheus.HistogramVec
	mu               sync.RWMutex
}

// NewRegistry creates a new metrics registry
func NewRegistry() *Registry {
	return &Registry{
		metricsCollected: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "monitoring_metrics_collected_total",
				Help: "Total number of metrics collected",
			},
			[]string{"system_id"},
		),
		alertsTriggered: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "monitoring_alerts_triggered_total",
				Help: "Total number of alerts triggered",
			},
			[]string{"system_id", "severity"},
		),
		systemsMonitored: promauto.NewGauge(
			prometheus.GaugeOpts{
				Name: "monitoring_systems_monitored",
				Help: "Number of systems currently being monitored",
			},
		),
		httpRequests: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "monitoring_http_requests_total",
				Help: "Total number of HTTP requests",
			},
			[]string{"method", "path", "status"},
		),
		httpDuration: promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "monitoring_http_request_duration_seconds",
				Help:    "HTTP request duration in seconds",
				Buckets: prometheus.DefBuckets,
			},
			[]string{"method", "path"},
		),
	}
}

// IncrementMetricsCollected increments the metrics collected counter
func (r *Registry) IncrementMetricsCollected(systemID string, count int) {
	r.metricsCollected.WithLabelValues(systemID).Add(float64(count))
}

// IncrementAlertsTriggered increments the alerts triggered counter
func (r *Registry) IncrementAlertsTriggered(systemID, severity string) {
	r.alertsTriggered.WithLabelValues(systemID, severity).Inc()
}

// SetSystemsMonitored sets the number of systems being monitored
func (r *Registry) SetSystemsMonitored(count int) {
	r.systemsMonitored.Set(float64(count))
}

// IncrementHTTPRequests increments the HTTP requests counter
func (r *Registry) IncrementHTTPRequests(method, path, status string) {
	r.httpRequests.WithLabelValues(method, path, status).Inc()
}

// ObserveHTTPDuration observes HTTP request duration
func (r *Registry) ObserveHTTPDuration(method, path string, duration float64) {
	r.httpDuration.WithLabelValues(method, path).Observe(duration)
} 