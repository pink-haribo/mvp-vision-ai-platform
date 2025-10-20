# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Vision AI Training Platform** is a natural language-driven platform for training computer vision models. Users can configure and train models (ResNet, YOLO, etc.) by conversing with an LLM instead of writing configuration files.

**Current Stage:** Planning/Design - Architecture and API specifications are complete, but implementation has not started yet.

**Key Concept:** Natural language → LLM Intent Parser → Training Config → Temporal Workflow → Kubernetes Training Pod

## Architecture Overview

### High-Level Flow
```
User (자연어) → Frontend (Next.js) → API Gateway (Kong)
  → Intent Parser (LLM) → Orchestrator (Temporal)
  → VM Controller (K8s) → Training Runner (Pod with Sidecar)
```

### Core Components

**Frontend (Next.js 14)**
- Chat interface for natural language model configuration
- Real-time training monitoring via WebSocket
- Tech: React 18, TailwindCSS, Zustand, React Query, Socket.io

**Backend Services (All FastAPI + Python 3.11)**
1. **Intent Parser** - Parses natural language into `TrainingIntent` using LangChain + Claude/GPT-4
2. **Orchestrator** - Manages training workflows using Temporal
3. **Model Registry** - Adapter pattern for multiple frameworks (timm, HuggingFace, Ultralytics, MMDetection)
4. **Data Service** - Dataset upload, validation, format conversion (COCO, YOLO, Pascal VOC)
5. **VM Controller** - Kubernetes pod lifecycle management
6. **Telemetry Service** - Real-time metrics collection and WebSocket broadcast

**Training Runner (Kubernetes Pod)**
- Dual container pattern: User Container (runs actual training) + Sidecar Container (monitors, uploads checkpoints, sends telemetry)
- Sidecar parses stdout for metrics and auto-uploads checkpoints to S3

**Orchestration & Data**
- Temporal 1.22.x for workflow orchestration
- PostgreSQL 16 (users, projects, metadata)
- MongoDB 7 (configs, workflow definitions)
- Redis 7.2 (cache, Celery queue, real-time state)
- S3/MinIO (datasets, models, checkpoints)

## Key Patterns & Principles

### 1. Adapter Pattern for Model Frameworks
All model sources (timm, HuggingFace, Ultralytics, etc.) implement `ModelAdapter` interface with:
- `load_model()` - Load model metadata
- `prepare_training_script()` - Generate training code
- `normalize_output()` - Standardize training results

See ARCHITECTURE.md:276-338 for implementation details.

### 2. Temporal Workflow Orchestration
Training workflow steps:
1. Validate dataset
2. Prepare model
3. Allocate VM (Kubernetes pod)
4. Run training (up to 24h timeout, 5min heartbeat)
5. Cleanup resources

See ARCHITECTURE.md:200-253 for workflow definition.

### 3. Intent Parsing Flow
LLM parses user message → Returns either:
- `ParseResult.complete` with full `TrainingIntent`
- `ParseResult.needs_clarification` with list of questions

User fills gaps → LLM generates executable `TrainingConfig`

See API_SPECIFICATION.md:106-187 for request/response examples.

## Development Commands

**Note:** The codebase is in the planning stage. The commands below are from DEVELOPMENT.md and will be used once implementation begins.

### Infrastructure (Docker Compose)
```bash
# Start PostgreSQL, MongoDB, Redis, MinIO
docker-compose up -d

# Check status
docker-compose ps

# Shutdown
docker-compose down
```

### Frontend
```bash
cd frontend
pnpm install        # Install dependencies
pnpm dev            # Dev server (http://localhost:3000)
pnpm build          # Production build
pnpm lint           # ESLint
pnpm type-check     # TypeScript validation
pnpm test           # Jest + React Testing Library
pnpm test:e2e       # Playwright E2E tests
```

### Backend Services
Each service runs on a different port (8001-8006):

```bash
cd backend/intent-parser
poetry install
poetry run uvicorn app.main:app --reload --port 8001

# Similarly for other services:
# orchestrator:     8002
# model-registry:   8003
# data-service:     8004
# vm-controller:    8005
# telemetry:        8006
```

### Testing
```bash
# Frontend
cd frontend && pnpm test
cd frontend && pnpm test:e2e

# Backend (each service)
cd backend/intent-parser
poetry run pytest tests/unit -v           # Unit tests
poetry run pytest tests/integration -v    # Integration tests
poetry run pytest --cov=app tests/        # Coverage report

# Run specific test
poetry run pytest tests/unit/test_parser.py::test_parse_classification -v
```

### Database
```bash
# PostgreSQL migrations (Alembic)
cd backend/orchestrator
poetry run alembic upgrade head      # Apply migrations
poetry run alembic downgrade -1      # Rollback one migration
poetry run alembic revision -m "..."  # Create new migration

# MongoDB index setup
python scripts/init_mongodb.py
```

### Temporal
```bash
# Start Temporal dev server
temporal server start-dev

# Or via Docker
docker run -d -p 7233:7233 -p 8233:8233 temporalio/auto-setup:latest

# Temporal UI: http://localhost:8233
```

## Code Style & Conventions

### Python
- Formatter: Black (line-length 100)
- Import sorting: isort (profile="black")
- Linting: flake8, pylint
- Type checking: mypy (python_version="3.11")
- Docstrings: Google style

```bash
black .
isort .
flake8 .
mypy .
```

### TypeScript/JavaScript
- Formatter: Prettier (semi, singleQuote, printWidth 100)
- Linting: ESLint (next/core-web-vitals, @typescript-eslint)

```bash
pnpm lint
pnpm format
```

### Git Workflow
- Branch naming: `feature/<name>`, `bugfix/<name>`, `hotfix/<issue>`
- Commits: Conventional Commits format
  ```
  feat(scope): add new feature
  fix(scope): fix bug
  docs(scope): update documentation
  ```
- Main branch: `main` (protected)
- Development branch: `develop` (default)

See DEVELOPMENT.md:466-563 for detailed git workflow.

## API Structure

### Public REST API
Base URL: `/api/v1`
Authentication: JWT Bearer tokens

**Key Endpoints:**
- `POST /chat/message` - Send natural language message, get parsed intent or clarification questions
- `POST /workflows` - Create training workflow
- `GET /workflows/{id}` - Get workflow status and metrics
- `ws://api/v1/ws/workflows/{id}` - WebSocket for real-time training updates
- `POST /datasets/upload` - Upload dataset (multipart/form-data)
- `GET /models` - List available models (filter by task_type, framework)
- `POST /inference/{workflow_id}/predict` - Run inference on trained model

**WebSocket Message Types:**
- `training_progress` - Epoch, step, metrics, ETA
- `training_complete` - Final results
- `training_error` - Error details
- `checkpoint_saved` - Checkpoint info

See API_SPECIFICATION.md for complete API reference.

### Internal Service APIs
Prefix: `/internal/<service>`
Authentication: `X-Internal-Auth` header

Used for service-to-service communication only.

## Environment Variables

Required environment variables (see DEVELOPMENT.md:160-187):

```bash
# LLM APIs
ANTHROPIC_API_KEY=sk-ant-api03-...
OPENAI_API_KEY=sk-...

# Databases
DATABASE_URL=postgresql://admin:devpass@localhost:5432/vision_platform
MONGODB_URL=mongodb://localhost:27017/vision_platform
REDIS_URL=redis://localhost:6379

# Object Storage
S3_ENDPOINT=http://localhost:9000
S3_BUCKET=vision-platform-dev
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin

# Temporal
TEMPORAL_HOST=localhost:7233

# Auth
JWT_SECRET=your-super-secret-key-change-this-in-production
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60
```

Frontend needs separate `.env.local`:
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
NEXT_PUBLIC_ENVIRONMENT=development
```

## Important Implementation Details

### Training Runner Sidecar Pattern
The sidecar container monitors user training code without requiring instrumentation:
- Parses stdout for metrics using regex (e.g., "Epoch 5, Loss: 0.234")
- Watches `/workspace/*.pth` for auto-upload to S3
- Collects GPU/memory metrics via system APIs
- Sends heartbeats to Temporal workflow

See ARCHITECTURE.md:490-536 for sidecar implementation.

### Dataset Format Support
Supported formats: COCO, YOLO, Pascal VOC, ImageNet
Data Service can convert between formats using `convert_format()` method.

### Supported Model Frameworks
- timm (PyTorch Image Models)
- HuggingFace Transformers
- Ultralytics YOLO
- MMDetection
- MMSegmentation
- Detectron2
- Custom Docker Images

Each framework has its own adapter implementing the `ModelAdapter` interface.

## Design System

The platform uses a custom design system based on:
- Font: SUIT (Korean) + Inter (English fallback)
- Colors: Brand Primary (#4F46E5 Indigo), Success (#10B981), Warning (#F59E0B), Error (#EF4444)
- Shadows: Layered elevation system
- Spacing: 4px base unit

Components follow atomic design: Atoms → Molecules → Organisms

See DESIGN_SYSTEM.md and UI_COMPONENTS.md for complete specifications.

## Common Troubleshooting

### Port Conflicts
```bash
lsof -i :3000   # Check what's using port 3000
kill -9 <PID>   # Kill process
PORT=3001 pnpm dev  # Use different port
```

### Docker Disk Space
```bash
docker system df        # Check usage
docker system prune -a  # Clean up
```

### Database Reset
```bash
docker-compose down -v                    # Remove volumes
docker-compose up -d postgres             # Start fresh
cd backend/orchestrator
poetry run alembic upgrade head           # Re-run migrations
```

### Python Dependency Conflicts
```bash
cd backend/<service>
rm -rf .venv
poetry install
```

See DEVELOPMENT.md:872-952 for more troubleshooting scenarios.

## Custom Slash Commands

This repository includes custom slash commands to streamline development workflows.

### /commit - Smart Git Commit

Automatically analyzes code changes and creates a well-formatted git commit.

**Usage:**
```
/commit
```

**What it does:**
1. Runs `git status` and `git diff` to analyze changes
2. Categorizes changes by type (feat, fix, docs, etc.)
3. Generates a Conventional Commits message
4. Stages appropriate files (excluding secrets)
5. Creates the commit with proper formatting
6. Verifies the commit was created successfully

**Example output:**
```bash
# Analyzes changes to mvp/backend/app/main.py
# Creates commit:
feat(backend): add FastAPI application structure

Create main FastAPI app with health check endpoint and CORS middleware.
Add configuration management with pydantic-settings.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

**Safety features:**
- Never commits secrets or `.env` files
- Never uses `--amend` unless explicitly requested
- Never pushes unless explicitly requested
- Warns about suspicious files

See `.claude/commands/commit.md` for full implementation.

## Documentation References

- **ARCHITECTURE.md** - Detailed system design, component responsibilities, data flows
- **API_SPECIFICATION.md** - Complete REST API and WebSocket reference
- **DEVELOPMENT.md** - Setup instructions, coding conventions, testing
- **DESIGN_SYSTEM.md** - UI design tokens, colors, typography
- **UI_COMPONENTS.md** - Component specifications and usage
- **README.md** - Project overview, quick start, roadmap
- **MVP_PLAN.md** - 2-week MVP implementation plan
- **MVP_STRUCTURE.md** - Detailed MVP folder structure
- **DATABASE_SCHEMA.md** - Database schemas and migration guides

## Notes for Future Implementation

When implementing features, remember:
1. The project structure (frontend/, backend/, infrastructure/) doesn't exist yet - create as needed
2. Use the Adapter pattern for all model frameworks to maintain consistency
3. All training is orchestrated via Temporal workflows - never start training directly
4. Real-time updates MUST go through WebSocket, not polling
5. The sidecar pattern is critical for framework-agnostic monitoring
6. All natural language processing goes through the Intent Parser service
7. Use Poetry for Python dependencies, pnpm for Node.js
8. Follow the branching strategy: feature branches → develop → main

## Current Development Status (Phase 1 MVP - Q1 2025)

From README.md roadmap:
- [x] Basic UI/UX design
- [x] Natural language parsing (LLM) architecture
- [x] Model framework support (timm, HuggingFace) specification
- [x] Local training execution design
- [ ] Basic telemetry implementation
- [ ] Actual code implementation

**Next Steps:** Begin implementation starting with infrastructure setup (Docker Compose), then backend services, then frontend.
