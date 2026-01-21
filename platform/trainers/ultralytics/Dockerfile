# Ultralytics YOLO Trainer
# Build from platform/ directory:
#   docker build -f trainers/ultralytics/Dockerfile -t trainer-ultralytics:latest .
# Run: docker run --env-file .env trainer-ultralytics:latest python train.py --job-id 123 ...
#
# For GPU support, use NVIDIA base image:
#   docker build -f trainers/ultralytics/Dockerfile.gpu -t trainer-ultralytics:gpu .

ARG REGISTRY_MIRROR=

# ============================================
# Builder Stage
# ============================================
FROM ${REGISTRY_MIRROR}python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first (for better caching)
COPY trainers/ultralytics/pyproject.toml .

# Install all dependencies including export extras using uv
RUN uv pip install --system --no-cache ".[export]"

# ============================================
# Production Stage
# ============================================
FROM ${REGISTRY_MIRROR}python:3.11-slim AS production

WORKDIR /app

# Install runtime dependencies for OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy all trainer code
COPY trainers/ultralytics/train.py .
COPY trainers/ultralytics/export.py .
COPY trainers/ultralytics/predict.py .
COPY trainers/ultralytics/evaluate.py .
COPY trainers/ultralytics/utils.py .
COPY trainers/ultralytics/config_schema.py .
COPY trainers/ultralytics/validate_capabilities.py .

# Copy unified SDK from shared location
COPY trainer-sdk/trainer_sdk.py .

# Copy configuration and runtime files
COPY trainers/ultralytics/capabilities.json .
COPY trainers/ultralytics/runtimes/ ./runtimes/

# Create directories for data and outputs
RUN mkdir -p /app/data /app/outputs /app/checkpoints

# Create non-root user for security
RUN groupadd --gid 1000 trainer && \
    useradd --uid 1000 --gid trainer --shell /bin/bash --create-home trainer && \
    chown -R trainer:trainer /app

USER trainer

# Environment settings
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check (for Kubernetes readiness probe)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import ultralytics; print('OK')" || exit 1

# Default command (can be overridden)
CMD ["python", "train.py", "--help"]
