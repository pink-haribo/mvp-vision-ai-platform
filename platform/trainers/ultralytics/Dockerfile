# Ultralytics YOLO Trainer
# Build: docker build -t trainer-ultralytics:latest .
# Run: docker run --env-file .env trainer-ultralytics:latest python train.py --job-id 123 ...

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy trainer code
COPY train.py .
COPY utils.py .

# Set Python path
ENV PYTHONUNBUFFERED=1

# Default command (can be overridden)
CMD ["python", "train.py", "--help"]
