# MMYOLO Trainer (YOLO Series)
# Build: docker build -t trainer-mmyolo:latest .
# Run: docker run --gpus all --env-file .env trainer-mmyolo:latest python train.py ...

ARG REGISTRY_MIRROR=
ARG PYTORCH_VERSION=2.1.0
ARG CUDA_VERSION=12.1

# ============================================
# Builder Stage
# ============================================
FROM ${REGISTRY_MIRROR}pytorch/pytorch:${PYTORCH_VERSION}-cuda${CUDA_VERSION}-cudnn8-runtime AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    git \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install OpenMMLab packages (mmyolo depends on mmdet)
RUN pip install --no-cache-dir openmim && \
    mim install mmengine && \
    mim install "mmcv>=2.0.0,<2.2.0" && \
    mim install "mmdet>=3.0.0" && \
    mim install "mmyolo>=0.6.0"

# Copy dependency files
COPY pyproject.toml .

# Install remaining dependencies
RUN uv pip install --system --no-cache ".[export]"

# ============================================
# Production Stage
# ============================================
FROM ${REGISTRY_MIRROR}pytorch/pytorch:${PYTORCH_VERSION}-cuda${CUDA_VERSION}-cudnn8-runtime AS production

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /opt/conda/lib/python3.10/site-packages /opt/conda/lib/python3.10/site-packages
COPY --from=builder /opt/conda/bin /opt/conda/bin

# Copy trainer code
COPY train.py .
COPY trainer_sdk.py .
COPY capabilities.json .

# Create directories
RUN mkdir -p /app/data /app/outputs /app/checkpoints /app/configs

# Create non-root user
RUN groupadd --gid 1000 trainer && \
    useradd --uid 1000 --gid trainer --shell /bin/bash --create-home trainer && \
    chown -R trainer:trainer /app

USER trainer

# Environment settings
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import mmyolo; print('OK')" || exit 1

# Default command
CMD ["python", "train.py", "--help"]
