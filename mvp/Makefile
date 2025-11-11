# ================================
# MVP Makefile
# ================================

.PHONY: help mvp-setup mvp-backend mvp-frontend mvp-dev mvp-clean mvp-test

help: ## Show this help
	@echo "MVP Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ================================
# Setup
# ================================

mvp-setup: ## Setup MVP environment
	@echo "🚀 Setting up MVP..."
	@if [ ! -f .env.mvp ]; then \
		cp .env.mvp.example .env.mvp; \
		echo "✅ Created .env.mvp"; \
		echo "⚠️  Please edit .env.mvp and add your ANTHROPIC_API_KEY"; \
	fi
	@echo "📦 Installing backend dependencies..."
	cd mvp/backend && pip install -r requirements.txt
	@echo "📦 Installing training dependencies..."
	cd mvp/training && pip install -r requirements.txt
	@echo "✅ MVP setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Edit .env.mvp and add your API key"
	@echo "  2. Run 'make mvp-dev' to start backend"
	@echo "  3. Open http://localhost:8000/docs"

mvp-create-dirs: ## Create data directories
	mkdir -p mvp/data/uploads
	mkdir -p mvp/data/outputs
	mkdir -p mvp/data/models
	mkdir -p mvp/data/logs
	mkdir -p mvp/data/db
	@echo "✅ Data directories created"

# ================================
# Development
# ================================

mvp-backend: ## Run backend server
	@echo "🚀 Starting backend on http://localhost:8000"
	cd mvp/backend && uvicorn app.main:app --reload --port 8000 --env-file ../../.env.mvp

mvp-frontend: ## Run frontend (placeholder)
	@echo "Frontend not yet implemented"
	@echo "Coming soon!"

mvp-dev: mvp-backend ## Run backend (frontend not yet implemented)

# ================================
# Training
# ================================

mvp-train-sample: ## Run sample training
	@echo "🎓 Running sample training..."
	cd mvp/training && python train_classification.py \
		--data_dir ../data/uploads/sample \
		--output_dir ../data/outputs/sample_run \
		--model resnet50 \
		--epochs 2 \
		--batch_size 8

# ================================
# Testing
# ================================

mvp-test-backend: ## Run backend tests
	cd mvp/backend && pytest tests/ -v

mvp-test: mvp-test-backend ## Run all tests

# ================================
# Cleanup
# ================================

mvp-clean: ## Clean generated files
	@echo "🧹 Cleaning..."
	rm -f mvp/data/db/*.db
	find mvp -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find mvp -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find mvp -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Cleanup complete"

mvp-clean-data: ## Clean all data (WARNING: destructive)
	@echo "⚠️  WARNING: This will delete all uploaded data and training outputs!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf mvp/data/uploads/*; \
		rm -rf mvp/data/outputs/*; \
		rm -rf mvp/data/models/*; \
		rm -rf mvp/data/logs/*; \
		echo "✅ Data cleaned"; \
	fi

# ================================
# Utils
# ================================

mvp-logs: ## Show backend logs
	tail -f mvp/data/logs/backend.log

mvp-check: ## Check MVP environment
	@echo "Checking MVP environment..."
	@echo ""
	@echo "📁 Directory structure:"
	@ls -la mvp/ 2>/dev/null && echo "✅ mvp/ exists" || echo "❌ mvp/ missing"
	@echo ""
	@echo "🔑 Environment:"
	@[ -f .env.mvp ] && echo "✅ .env.mvp exists" || echo "❌ .env.mvp missing"
	@echo ""
	@echo "📦 Backend dependencies:"
	@cd mvp/backend && pip list | grep fastapi > /dev/null && echo "✅ FastAPI installed" || echo "❌ FastAPI not installed"
	@echo ""
	@echo "📦 Training dependencies:"
	@cd mvp/training && pip list | grep torch > /dev/null && echo "✅ PyTorch installed" || echo "❌ PyTorch not installed"

.DEFAULT_GOAL := help
