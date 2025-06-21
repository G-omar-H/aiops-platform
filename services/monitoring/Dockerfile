# AIOps Monitoring Service Dockerfile
# Multi-stage build for optimized production image

# Stage 1: Build environment
FROM golang:1.21-alpine AS builder

# Set build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Add metadata
LABEL org.opencontainers.image.title="AIOps Monitoring Service" \
      org.opencontainers.image.description="Enterprise monitoring and metrics collection service" \
      org.opencontainers.image.created=$BUILD_DATE \
      org.opencontainers.image.revision=$VCS_REF \
      org.opencontainers.image.version=$VERSION \
      org.opencontainers.image.vendor="AIOps Platform" \
      org.opencontainers.image.licenses="MIT"

# Install dependencies
RUN apk add --no-cache git ca-certificates tzdata

# Set working directory
WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download && go mod verify

# Copy source code
COPY . .

# Build the application
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \
    -ldflags="-w -s -X main.Version=${VERSION} -X main.BuildTime=${BUILD_DATE} -X main.GitCommit=${VCS_REF}" \
    -a -installsuffix cgo \
    -o monitoring-service \
    ./cmd/monitoring-service

# Stage 2: Runtime environment
FROM alpine:3.18 AS runtime

# Install runtime dependencies
RUN apk add --no-cache ca-certificates tzdata curl \
    && addgroup -g 1001 -S aiops \
    && adduser -u 1001 -S aiops -G aiops

# Copy binary and assets from builder
COPY --from=builder /app/monitoring-service /usr/local/bin/monitoring-service
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
COPY --from=builder /usr/share/zoneinfo /usr/share/zoneinfo

# Create directories
RUN mkdir -p /app/config /app/logs /app/data \
    && chown -R aiops:aiops /app

# Switch to non-root user
USER aiops

# Set working directory
WORKDIR /app

# Environment variables
ENV TZ=UTC \
    GIN_MODE=release \
    LOG_LEVEL=info \
    SERVER_PORT=8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${SERVER_PORT}/health || exit 1

# Expose port
EXPOSE 8080

# Start the application
ENTRYPOINT ["/usr/local/bin/monitoring-service"] 