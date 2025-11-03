FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
# Note: Railway build context is at repository root
COPY mvp/backend/requirements.txt .

# Install Python dependencies (backend only - lightweight)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY mvp/backend/ .

# Copy sample datasets to /app/datasets (shared via volume in production)
COPY mvp/backend/sample_datasets/ /app/datasets/

# Note: Training code is NOT copied to Backend (proper dependency isolation)
# Backend communicates with Training Services via HTTP API for all training-related operations:
# - Model registry: GET /models/list, GET /models/{name}
# - Config schema: GET /config-schema?framework=X&task_type=Y
# - Training execution: POST /training/start
# This maintains clean separation between Backend (API/DB) and Training (ML)

# Create necessary directories
RUN mkdir -p outputs uploads /app/data/outputs /app/data/logs /app/mlruns

# Make start script executable
RUN chmod +x start.sh

# Expose ports (Railway will set PORT env var for FastAPI)
EXPOSE ${PORT:-8000}
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:${PORT:-8000}/health')" || exit 1

# Start both MLflow server and FastAPI application
CMD ["./start.sh"]
