FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
# Note: Railway build context is at repository root
COPY mvp/backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY mvp/backend/ .

# Copy training module (needed by model_registry)
COPY mvp/training/ /training/

# Create necessary directories
RUN mkdir -p outputs uploads datasets

# Expose port (Railway will set PORT env var)
EXPOSE ${PORT:-8000}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:${PORT:-8000}/health')" || exit 1

# Start server
# Railway will provide PORT environment variable
# Note: Run migrations manually after first deployment
# Railway CLI: railway run python -c "from app.db.database import init_db; init_db()"
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
