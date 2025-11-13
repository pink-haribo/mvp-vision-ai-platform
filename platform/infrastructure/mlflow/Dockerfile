FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install MLflow and PostgreSQL adapter
RUN pip install --no-cache-dir \
    mlflow==2.9.2 \
    psycopg2-binary==2.9.9 \
    boto3==1.34.10

# Create MLflow directory
RUN mkdir -p /mlflow

WORKDIR /mlflow

# Expose MLflow port
EXPOSE 5000

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Start MLflow server
# - Backend store: PostgreSQL for metadata
# - Artifact store: MinIO (S3-compatible)
# - Host: 0.0.0.0 to accept external connections
CMD mlflow server \
    --backend-store-uri ${BACKEND_STORE_URI} \
    --default-artifact-root ${ARTIFACT_ROOT} \
    --host 0.0.0.0 \
    --port 5000
