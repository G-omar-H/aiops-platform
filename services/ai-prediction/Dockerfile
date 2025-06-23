# AIOps AI Prediction Service Dockerfile
# Multi-stage build for optimized production deployment

# Stage 1: Base Python image with system dependencies
FROM python:3.11-slim as base

# Set build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Add metadata
LABEL org.opencontainers.image.title="AIOps AI Prediction Service" \
      org.opencontainers.image.description="Enterprise ML service for predictive analytics and anomaly detection" \
      org.opencontainers.image.created=$BUILD_DATE \
      org.opencontainers.image.revision=$VCS_REF \
      org.opencontainers.image.version=$VERSION \
      org.opencontainers.image.vendor="AIOps Platform" \
      org.opencontainers.image.licenses="MIT"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set environment variables for Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Stage 2: Dependencies installation
FROM base as dependencies

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Stage 3: Application build
FROM dependencies as builder

# Create app directory
WORKDIR /app

# Copy application code
COPY . .

# Create non-root user for security
RUN groupadd -r aiops && useradd -r -g aiops -u 1001 aiops

# Create necessary directories
RUN mkdir -p /app/models /app/logs /app/data /app/cache \
    && chown -R aiops:aiops /app

# Stage 4: Production runtime
FROM python:3.11-slim as runtime

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    libpq5 \
    libhdf5-103 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python environment from dependencies stage
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Copy application from builder stage
COPY --from=builder /app /app
COPY --from=builder /etc/passwd /etc/passwd
COPY --from=builder /etc/group /etc/group

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app \
    PATH="/usr/local/bin:$PATH" \
    ENVIRONMENT=production \
    LOG_LEVEL=info \
    PORT=8000 \
    WORKERS=1 \
    MAX_WORKERS=4 \
    METRICS_PORT=9090

# Switch to non-root user
USER aiops

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose ports
EXPOSE 8000 9090

# Start the application
CMD ["python", "main.py"]

# Development stage (optional)
FROM runtime as development

USER root

# Install development dependencies
RUN pip install --no-cache-dir \
    debugpy==1.8.0 \
    ipython==8.17.2 \
    jupyter==1.0.0

# Install additional development tools
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

USER aiops

# Override environment for development
ENV ENVIRONMENT=development \
    LOG_LEVEL=debug \
    DEBUG=true

# Development command with hot reload
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"] 