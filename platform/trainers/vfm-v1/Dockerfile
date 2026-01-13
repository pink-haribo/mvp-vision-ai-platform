# VFM-v1 Trainer (Vision Foundation Model based on YOLO World)
# Build: docker build -t trainer-vfm-v1:latest .
# Run: docker run --gpus all --env-file .env trainer-vfm-v1:latest python train.py --job-id 123 ...

# ============================================
# Builder Stage
# ============================================
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV FORCE_CUDA="1"
ENV MMCV_WITH_OPS=1
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0"

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    python3-wheel \
    git \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2

# Install MMEngine ecosystem
RUN pip3 install --no-cache-dir \
    mmengine>=0.10.0 \
    openmim \
    && mim install mmcv==2.1.0 \
    && pip3 install --no-cache-dir \
    mmdet==3.3.0 \
    mmyolo==0.6.0

# Install VFM-specific dependencies
RUN pip3 install --no-cache-dir \
    transformers>=4.28.0 \
    tokenizers>=0.13.0 \
    open-clip-torch>=2.20.0

# Install TrainerSDK dependencies
RUN pip3 install --no-cache-dir \
    boto3>=1.34.0 \
    httpx>=0.26.0 \
    tenacity>=8.2.0 \
    python-dotenv>=1.0.0 \
    pyyaml>=6.0

# Copy VFM source code and install
COPY . /app/vfm-v1
WORKDIR /app/vfm-v1
RUN pip3 install -e .

# ============================================
# Production Stage
# ============================================
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 AS production

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy VFM source code
COPY --from=builder /app/vfm-v1 /app/vfm-v1

WORKDIR /app/vfm-v1

# Copy trainer scripts (overwrite with latest)
COPY train.py .
COPY trainer_sdk.py .
COPY capabilities.json .
COPY requirements.txt .

# Create directories for data and outputs
RUN mkdir -p /app/data /app/outputs /app/checkpoints /tmp/training

# Create pretrained_models directory
RUN mkdir -p /app/vfm-v1/pretrained_models

# Create non-root user for security
RUN groupadd --gid 1000 trainer && \
    useradd --uid 1000 --gid trainer --shell /bin/bash --create-home trainer && \
    chown -R trainer:trainer /app /tmp/training

USER trainer

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python3 -c "import torch; import mmengine; print('OK')" || exit 1

# Default command
CMD ["python3", "train.py", "--help"]
