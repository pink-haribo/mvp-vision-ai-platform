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

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY mvp/backend/ .

# Copy training module to /mvp/training/ (code expects this path)
RUN mkdir -p /mvp/training
COPY mvp/training/ /mvp/training/

# Copy sample datasets to /app/datasets
# Note: sample_datasets should be in mvp/backend/sample_datasets/
COPY mvp/backend/sample_datasets/ /app/datasets/

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
