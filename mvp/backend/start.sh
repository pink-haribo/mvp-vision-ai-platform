#!/bin/bash

# Start MLflow server in background
echo "Starting MLflow server on port 5000..."
mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root /app/mlruns \
    &

# Wait for MLflow to be ready with health check
echo "Waiting for MLflow server to start..."
max_attempts=30
attempt=0
until curl -s http://localhost:5000/health > /dev/null 2>&1; do
    attempt=$((attempt + 1))
    if [ $attempt -ge $max_attempts ]; then
        echo "WARNING: MLflow did not start within 30 seconds, continuing anyway..."
        break
    fi
    echo "Attempt $attempt/$max_attempts: MLflow not ready yet..."
    sleep 1
done

echo "MLflow server is ready!"

# Start FastAPI application
echo "Starting FastAPI application on port ${PORT:-8000}..."
exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
