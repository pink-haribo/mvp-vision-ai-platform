# MLflow Tracking Server for Railway Deployment
FROM python:3.11-slim

LABEL maintainer="Vision AI Platform"
LABEL description="MLflow tracking server for experiment tracking and model registry"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install MLflow and dependencies
RUN pip install --no-cache-dir \
    mlflow==2.9.2 \
    psycopg2-binary \
    boto3 \
    gunicorn

# Create MLflow user and directories
RUN useradd -m -u 1000 mlflow && \
    mkdir -p /mlflow/data && \
    chown -R mlflow:mlflow /mlflow

# Set working directory
WORKDIR /mlflow

# Switch to mlflow user
USER mlflow

# Expose MLflow UI port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Environment variables (will be overridden by Railway)
ENV MLFLOW_BACKEND_STORE_URI="sqlite:///mlflow/data/mlflow.db"
ENV MLFLOW_DEFAULT_ARTIFACT_ROOT="./mlflow-artifacts"
ENV MLFLOW_HOST="0.0.0.0"
ENV MLFLOW_PORT="5000"
ENV GUNICORN_WORKERS="2"
ENV GUNICORN_THREADS="4"

# Entrypoint script
COPY --chown=mlflow:mlflow docker/mlflow/docker-entrypoint.sh /mlflow/
RUN chmod +x /mlflow/docker-entrypoint.sh

ENTRYPOINT ["/mlflow/docker-entrypoint.sh"]
