# Build stage
FROM golang:1.21-alpine AS builder

# Install build dependencies
RUN apk add --no-cache git ca-certificates tzdata make

# Set working directory
WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download && go mod verify

# Copy source code
COPY . .

# Build the application with optimizations
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \
    -ldflags='-w -s -extldflags "-static"' \
    -tags netgo \
    -installsuffix netgo \
    -o data-pipeline \
    ./main.go

# Development stage (for local development)
FROM golang:1.21-alpine AS development

# Install development tools
RUN apk add --no-cache git make curl bash vim postgresql-client redis

# Install air for hot reload
RUN go install github.com/cosmtrek/air@latest

# Install swag for API documentation
RUN go install github.com/swaggo/swag/cmd/swag@latest

# Set working directory
WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Expose port
EXPOSE 8080

# Default command for development
CMD ["air", "-c", ".air.toml"]

# Production stage
FROM alpine:3.18 AS production

# Install runtime dependencies
RUN apk add --no-cache \
    ca-certificates \
    tzdata \
    curl \
    bash \
    && rm -rf /var/cache/apk/*

# Create non-root user for security
RUN addgroup -g 1001 -S appgroup && \
    adduser -u 1001 -S appuser -G appgroup

# Set timezone
ENV TZ=UTC

# Create necessary directories
RUN mkdir -p /app/logs /app/config /app/data && \
    chown -R appuser:appgroup /app

# Copy binary from builder stage
COPY --from=builder /app/data-pipeline /app/data-pipeline
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

# Copy configuration files
COPY --chown=appuser:appgroup ./config/ /app/config/

# Set working directory
WORKDIR /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Set environment variables
ENV ENVIRONMENT=production
ENV LOG_LEVEL=info
ENV SERVER_PORT=8080

# Labels for better container management
LABEL maintainer="AIOps Platform Team" \
      version="1.0.0" \
      description="Enterprise Data Pipeline Service for AIOps Platform" \
      org.opencontainers.image.title="data-pipeline" \
      org.opencontainers.image.description="Real-time data processing pipeline" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.vendor="AIOps Platform" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/aiops-platform/data-pipeline"

# Run the application
CMD ["./data-pipeline"]

# Testing stage
FROM production AS testing

# Switch back to root for installing test dependencies
USER root

# Install testing tools
RUN apk add --no-cache git make

# Copy test files
COPY --chown=appuser:appgroup ./tests/ /app/tests/
COPY --chown=appuser:appgroup ./scripts/ /app/scripts/

# Install gotestsum for better test output
RUN go install gotest.tools/gotestsum@latest

# Switch back to non-root user
USER appuser

# Override command for testing
CMD ["make", "test"] 