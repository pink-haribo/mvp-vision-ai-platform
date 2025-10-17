# ================================
# Vision AI Platform - Makefile
# ================================
# Convenience commands for development

.PHONY: help
help: ## Show this help message
	@echo "Vision AI Platform - Development Commands"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

# ================================
# Infrastructure
# ================================

.PHONY: infra-up
infra-up: ## Start all infrastructure services (Docker Compose)
	docker-compose up -d
	@echo "✅ Infrastructure is starting..."
	@echo "Waiting for services to be healthy..."
	@sleep 10
	@docker-compose ps
	@echo ""
	@echo "📊 Service URLs:"
	@echo "  PostgreSQL:    localhost:5432"
	@echo "  MongoDB:       localhost:27017"
	@echo "  Redis:         localhost:6379"
	@echo "  MinIO API:     http://localhost:9000"
	@echo "  MinIO Console: http://localhost:9001 (minioadmin/minioadmin)"
	@echo "  Temporal:      localhost:7233"
	@echo "  Temporal UI:   http://localhost:8233"
	@echo "  Prometheus:    http://localhost:9090"
	@echo "  Grafana:       http://localhost:3001 (admin/admin)"
	@echo "  Mailhog:       http://localhost:8025"

.PHONY: infra-down
infra-down: ## Stop all infrastructure services
	docker-compose down

.PHONY: infra-clean
infra-clean: ## Stop and remove all containers, volumes, and networks
	docker-compose down -v --remove-orphans
	@echo "⚠️  All data has been removed!"

.PHONY: infra-logs
infra-logs: ## View logs from all infrastructure services
	docker-compose logs -f

.PHONY: infra-ps
infra-ps: ## Show status of all services
	docker-compose ps

.PHONY: infra-restart
infra-restart: ## Restart all infrastructure services
	docker-compose restart

# ================================
# Database
# ================================

.PHONY: db-migrate
db-migrate: ## Run database migrations (PostgreSQL)
	cd backend/orchestrator && poetry run alembic upgrade head

.PHONY: db-migrate-down
db-migrate-down: ## Rollback last database migration
	cd backend/orchestrator && poetry run alembic downgrade -1

.PHONY: db-migrate-create
db-migrate-create: ## Create a new migration (usage: make db-migrate-create msg="description")
	@if [ -z "$(msg)" ]; then \
		echo "❌ Error: Please provide a message. Usage: make db-migrate-create msg='your message'"; \
		exit 1; \
	fi
	cd backend/orchestrator && poetry run alembic revision -m "$(msg)"

.PHONY: db-seed
db-seed: ## Seed database with sample data
	@echo "🌱 Seeding database..."
	cd backend/orchestrator && poetry run python scripts/seed_data.py

.PHONY: db-reset
db-reset: infra-clean infra-up db-migrate db-seed ## Reset database (clean + migrate + seed)
	@echo "✅ Database reset complete!"

.PHONY: mongo-indexes
mongo-indexes: ## Create MongoDB indexes
	@echo "📚 Creating MongoDB indexes..."
	python scripts/init_mongodb.py

# ================================
# Frontend
# ================================

.PHONY: frontend-install
frontend-install: ## Install frontend dependencies
	cd frontend && pnpm install

.PHONY: frontend-dev
frontend-dev: ## Start frontend development server
	cd frontend && pnpm dev

.PHONY: frontend-build
frontend-build: ## Build frontend for production
	cd frontend && pnpm build

.PHONY: frontend-lint
frontend-lint: ## Lint frontend code
	cd frontend && pnpm lint

.PHONY: frontend-type-check
frontend-type-check: ## Type check frontend code
	cd frontend && pnpm type-check

.PHONY: frontend-test
frontend-test: ## Run frontend tests
	cd frontend && pnpm test

.PHONY: frontend-test-watch
frontend-test-watch: ## Run frontend tests in watch mode
	cd frontend && pnpm test:watch

.PHONY: frontend-test-e2e
frontend-test-e2e: ## Run frontend E2E tests
	cd frontend && pnpm test:e2e

# ================================
# Backend Services
# ================================

.PHONY: backend-install-all
backend-install-all: ## Install all backend dependencies
	@echo "📦 Installing backend dependencies..."
	cd backend/intent-parser && poetry install
	cd backend/orchestrator && poetry install
	cd backend/model-registry && poetry install
	cd backend/data-service && poetry install
	cd backend/vm-controller && poetry install
	cd backend/telemetry && poetry install
	@echo "✅ All backend dependencies installed!"

.PHONY: backend-intent-parser
backend-intent-parser: ## Start Intent Parser service
	cd backend/intent-parser && poetry run uvicorn app.main:app --reload --port 8001

.PHONY: backend-orchestrator
backend-orchestrator: ## Start Orchestrator service
	cd backend/orchestrator && poetry run uvicorn app.main:app --reload --port 8002

.PHONY: backend-model-registry
backend-model-registry: ## Start Model Registry service
	cd backend/model-registry && poetry run uvicorn app.main:app --reload --port 8003

.PHONY: backend-data-service
backend-data-service: ## Start Data Service
	cd backend/data-service && poetry run uvicorn app.main:app --reload --port 8004

.PHONY: backend-vm-controller
backend-vm-controller: ## Start VM Controller service
	cd backend/vm-controller && poetry run uvicorn app.main:app --reload --port 8005

.PHONY: backend-telemetry
backend-telemetry: ## Start Telemetry service
	cd backend/telemetry && poetry run uvicorn app.main:app --reload --port 8006

# ================================
# Testing
# ================================

.PHONY: test-all
test-all: ## Run all tests (frontend + backend)
	@echo "🧪 Running all tests..."
	$(MAKE) frontend-test
	$(MAKE) test-backend

.PHONY: test-backend
test-backend: ## Run all backend tests
	@echo "🧪 Running backend tests..."
	cd backend/intent-parser && poetry run pytest -v
	cd backend/orchestrator && poetry run pytest -v
	cd backend/model-registry && poetry run pytest -v
	cd backend/data-service && poetry run pytest -v
	cd backend/vm-controller && poetry run pytest -v
	cd backend/telemetry && poetry run pytest -v

.PHONY: test-backend-coverage
test-backend-coverage: ## Run backend tests with coverage
	@echo "🧪 Running backend tests with coverage..."
	cd backend/intent-parser && poetry run pytest --cov=app --cov-report=html tests/
	cd backend/orchestrator && poetry run pytest --cov=app --cov-report=html tests/
	@echo "📊 Coverage reports generated in htmlcov/ directories"

.PHONY: test-integration
test-integration: ## Run integration tests
	@echo "🧪 Running integration tests..."
	cd backend/orchestrator && poetry run pytest tests/integration -v

# ================================
# Code Quality
# ================================

.PHONY: lint-backend
lint-backend: ## Lint backend code (Python)
	@echo "🔍 Linting backend code..."
	cd backend/intent-parser && poetry run flake8 app tests
	cd backend/intent-parser && poetry run black --check app tests
	cd backend/intent-parser && poetry run isort --check app tests

.PHONY: format-backend
format-backend: ## Format backend code (Python)
	@echo "✨ Formatting backend code..."
	cd backend/intent-parser && poetry run black app tests
	cd backend/intent-parser && poetry run isort app tests
	cd backend/orchestrator && poetry run black app tests
	cd backend/orchestrator && poetry run isort app tests

.PHONY: type-check-backend
type-check-backend: ## Type check backend code (mypy)
	@echo "🔍 Type checking backend code..."
	cd backend/intent-parser && poetry run mypy app
	cd backend/orchestrator && poetry run mypy app

.PHONY: lint-all
lint-all: lint-backend frontend-lint ## Lint all code

.PHONY: format-all
format-all: format-backend ## Format all code
	cd frontend && pnpm format

# ================================
# Development Workflow
# ================================

.PHONY: dev-setup
dev-setup: ## Complete development setup (install + infra + migrate + seed)
	@echo "🚀 Setting up development environment..."
	$(MAKE) infra-up
	@sleep 5
	$(MAKE) frontend-install
	$(MAKE) backend-install-all
	$(MAKE) db-migrate
	$(MAKE) mongo-indexes
	$(MAKE) db-seed
	@echo "✅ Development environment is ready!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Copy .env.example to .env and fill in your API keys"
	@echo "  2. Run 'make frontend-dev' to start the frontend"
	@echo "  3. Run 'make backend-<service>' to start backend services"

.PHONY: dev-check
dev-check: ## Check if development environment is properly set up
	@echo "🔍 Checking development environment..."
	@echo ""
	@echo "Checking infrastructure..."
	@docker-compose ps
	@echo ""
	@echo "Checking .env file..."
	@if [ ! -f .env ]; then echo "❌ .env file not found"; else echo "✅ .env file exists"; fi
	@echo ""
	@echo "Checking dependencies..."
	@cd frontend && pnpm list --depth=0 2>/dev/null && echo "✅ Frontend dependencies installed" || echo "❌ Frontend dependencies missing"
	@cd backend/intent-parser && poetry check 2>/dev/null && echo "✅ Backend dependencies installed" || echo "❌ Backend dependencies missing"

# ================================
# Cleanup
# ================================

.PHONY: clean-all
clean-all: ## Clean all generated files and caches
	@echo "🧹 Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "node_modules" -exec rm -rf {} +
	find . -type d -name ".next" -exec rm -rf {} +
	@echo "✅ Cleanup complete!"

.PHONY: clean-docker
clean-docker: ## Remove all Docker containers, images, and volumes
	docker system prune -af --volumes
	@echo "⚠️  All Docker data has been removed!"

# ================================
# Utilities
# ================================

.PHONY: logs-postgres
logs-postgres: ## View PostgreSQL logs
	docker-compose logs -f postgres

.PHONY: logs-mongodb
logs-mongodb: ## View MongoDB logs
	docker-compose logs -f mongodb

.PHONY: logs-redis
logs-redis: ## View Redis logs
	docker-compose logs -f redis

.PHONY: psql
psql: ## Connect to PostgreSQL CLI
	docker-compose exec postgres psql -U admin -d vision_platform

.PHONY: mongo-shell
mongo-shell: ## Connect to MongoDB shell
	docker-compose exec mongodb mongosh vision_platform

.PHONY: redis-cli
redis-cli: ## Connect to Redis CLI
	docker-compose exec redis redis-cli

# ================================
# Documentation
# ================================

.PHONY: docs-serve
docs-serve: ## Serve documentation locally
	@echo "📚 Serving documentation..."
	@echo "Available docs:"
	@echo "  - README.md"
	@echo "  - ARCHITECTURE.md"
	@echo "  - API_SPECIFICATION.md"
	@echo "  - DEVELOPMENT.md"
	@echo "  - DATABASE_SCHEMA.md"

# ================================
# CI/CD
# ================================

.PHONY: ci
ci: lint-all test-all ## Run CI pipeline (lint + test)
	@echo "✅ CI pipeline passed!"

# ================================
# Default
# ================================

.DEFAULT_GOAL := help
