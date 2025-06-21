# AI-Powered Prediction Service Dockerfile
# Multi-stage build for optimized production image

# Stage 1: Build environment
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Add metadata
LABEL org.opencontainers.image.title="AIOps AI Prediction Service" \
      org.opencontainers.image.description="Enterprise AI-powered system failure prediction service" \
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
    pkg-config \
    libhdf5-dev \
    liblapack-dev \
    libblas-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime environment
FROM python:3.11-slim as runtime

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    libhdf5-103 \
    liblapack3 \
    libblas3 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r aiops && useradd -r -g aiops aiops

# Copy Python packages from builder
COPY --from=builder /root/.local /home/aiops/.local

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Set ownership and permissions
RUN chown -R aiops:aiops /app && \
    chmod +x scripts/*.sh 2>/dev/null || true

# Switch to non-root user
USER aiops

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/home/aiops/.local/bin:$PATH

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "main.py"] 