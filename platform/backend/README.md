# Backend Service

**FastAPI-based backend** for Vision AI Training Platform with Temporal Workflows integration.

## Structure

```
backend/
├── api/              # REST API endpoints
│   ├── v1/          # API version 1
│   ├── auth.py      # Authentication
│   └── deps.py      # Dependencies
├── core/            # Business logic
│   ├── workflows/   # Temporal activity implementations
│   └── services/    # Business services
├── db/              # Database layer
│   ├── models.py    # SQLAlchemy models
│   └── migrations/  # Alembic migrations
└── config/          # Configuration
    └── settings.py  # Pydantic settings

```

## Key Features

- **Temporal Workflows**: Training job orchestration
- **JWT Authentication**: Secure API and callback tokens
- **PostgreSQL**: Relational data storage
- **Redis**: Caching and event streaming
- **S3 Storage**: Object storage abstraction

## API Endpoints

### Training Jobs
- `POST /api/v1/jobs` - Create training job
- `GET /api/v1/jobs/{id}` - Get job status
- `DELETE /api/v1/jobs/{id}` - Cancel job

### Callbacks (from Trainers)
- `POST /api/v1/jobs/{id}/heartbeat` - Progress updates
- `POST /api/v1/jobs/{id}/event` - Training events
- `POST /api/v1/jobs/{id}/done` - Completion notification

### WebSocket
- `WS /ws/jobs/{id}` - Real-time job updates

## Development

```bash
# Install dependencies
poetry install

# Run migrations
poetry run alembic upgrade head

# Start server
poetry run uvicorn app.main:app --reload --port 8000

# Run tests
poetry run pytest
```

## Environment Variables

See `.env.example` for required configuration.
