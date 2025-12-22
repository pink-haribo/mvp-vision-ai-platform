# OpenMMLab Trainer Docker Image
FROM python:3.11-slim

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy trainer files
COPY trainer_sdk.py .
COPY train.py .
COPY predict.py .
COPY export.py .
COPY evaluate.py .
COPY utils.py .
COPY config_schema.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FRAMEWORK=openmm

# Default command
CMD ["python", "train.py"]
