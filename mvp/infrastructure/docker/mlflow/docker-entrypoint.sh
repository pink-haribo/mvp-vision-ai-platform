#!/bin/bash
set -e

echo "========================================"
echo "Starting MLflow Tracking Server"
echo "========================================"

# Display configuration
echo "Backend Store URI: ${MLFLOW_BACKEND_STORE_URI}"
echo "Artifact Root: ${MLFLOW_DEFAULT_ARTIFACT_ROOT}"
echo "Host: ${MLFLOW_HOST}"
echo "Port: ${MLFLOW_PORT}"
echo "Workers: ${GUNICORN_WORKERS}"
echo "Threads: ${GUNICORN_THREADS}"

# Wait for PostgreSQL to be ready (if using PostgreSQL backend)
if [[ $MLFLOW_BACKEND_STORE_URI == postgresql://* ]]; then
    echo "Waiting for PostgreSQL to be ready..."

    # Extract host and port from connection string
    # Format: postgresql://user:pass@host:port/dbname
    DB_HOST=$(echo $MLFLOW_BACKEND_STORE_URI | sed -n 's/.*@\([^:]*\):.*/\1/p')
    DB_PORT=$(echo $MLFLOW_BACKEND_STORE_URI | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')

    if [ -z "$DB_PORT" ]; then
        DB_PORT=5432
    fi

    echo "Database host: $DB_HOST:$DB_PORT"

    # Wait for PostgreSQL
    for i in {1..30}; do
        if pg_isready -h "$DB_HOST" -p "$DB_PORT" > /dev/null 2>&1; then
            echo "PostgreSQL is ready!"
            break
        fi
        echo "Waiting for PostgreSQL... ($i/30)"
        sleep 2
    done
fi

# Check if S3 credentials are set
if [ -n "$AWS_ACCESS_KEY_ID" ] && [ -n "$AWS_SECRET_ACCESS_KEY" ]; then
    echo "AWS credentials detected"
    if [ -n "$AWS_S3_ENDPOINT_URL" ]; then
        echo "Using custom S3 endpoint: $AWS_S3_ENDPOINT_URL"
    fi
else
    echo "Warning: AWS credentials not set. Artifacts will be stored locally."
fi

# Start MLflow server with gunicorn for production
echo "Starting MLflow server..."
exec gunicorn \
    --bind "${MLFLOW_HOST}:${MLFLOW_PORT}" \
    --workers "${GUNICORN_WORKERS}" \
    --threads "${GUNICORN_THREADS}" \
    --timeout 120 \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
    "mlflow.server:app" \
    --env MLFLOW_BACKEND_STORE_URI="${MLFLOW_BACKEND_STORE_URI}" \
    --env MLFLOW_DEFAULT_ARTIFACT_ROOT="${MLFLOW_DEFAULT_ARTIFACT_ROOT}"
