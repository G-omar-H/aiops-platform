# AIOps Auto-Remediation Service Dockerfile
# Multi-stage build for optimized production deployment

# Stage 1: Build stage
FROM golang:1.21-alpine AS builder

# Set build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Add metadata
LABEL org.opencontainers.image.title="AIOps Auto-Remediation Service" \
      org.opencontainers.image.description="Enterprise auto-remediation service for autonomous incident response" \
      org.opencontainers.image.created=$BUILD_DATE \
      org.opencontainers.image.revision=$VCS_REF \
      org.opencontainers.image.version=$VERSION \
      org.opencontainers.image.vendor="AIOps Platform" \
      org.opencontainers.image.licenses="MIT"

# Install build dependencies
RUN apk add --no-cache git ca-certificates tzdata

# Set working directory
WORKDIR /app

# Copy go mod and sum files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build the application
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \
    -ldflags="-w -s -X main.version=${VERSION} -X main.buildDate=${BUILD_DATE} -X main.gitCommit=${VCS_REF}" \
    -a -installsuffix cgo \
    -o main .

# Stage 2: Production runtime
FROM alpine:3.18 as runtime

# Install runtime dependencies
RUN apk --no-cache add ca-certificates tzdata curl

# Create app directory and user
RUN addgroup -g 1001 -S aiops && \
    adduser -u 1001 -S aiops -G aiops && \
    mkdir -p /app /app/config /app/logs /app/workflows /app/templates && \
    chown -R aiops:aiops /app

# Set working directory
WORKDIR /app

# Copy binary from builder stage
COPY --from=builder /app/main .
COPY --from=builder /app/config ./config
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

# Switch to non-root user
USER aiops

# Set environment variables
ENV GIN_MODE=release \
    ENVIRONMENT=production \
    PORT=8080 \
    METRICS_PORT=9090 \
    LOG_LEVEL=info

# Expose ports
EXPOSE 8080 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health/live || exit 1

# Start the application
CMD ["./main"]

# Development stage (optional)
FROM builder as development

# Install development tools
RUN go install github.com/cosmtrek/air@latest && \
    go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest

# Set development environment
ENV GIN_MODE=debug \
    ENVIRONMENT=development \
    LOG_LEVEL=debug

# Development command with hot reload
CMD ["air", "-c", ".air.toml"] 