# Supply Chain Optimization Platform - Makefile
.PHONY: help setup build start stop restart logs clean test lint format docs deploy

# Default target
help: ## Show this help message
	@echo "Supply Chain Optimization Platform"
	@echo "=================================="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Environment setup
setup: ## Setup the development environment
	@echo "🚀 Setting up development environment..."
	@./scripts/setup.sh

# Docker operations
build: ## Build Docker images
	@echo "🐳 Building Docker images..."
	@docker-compose build --no-cache

start: ## Start all services
	@echo "▶️ Starting services..."
	@docker-compose up -d

stop: ## Stop all services
	@echo "⏹️ Stopping services..."
	@docker-compose down

restart: ## Restart all services
	@echo "🔄 Restarting services..."
	@docker-compose restart

logs: ## View service logs
	@echo "📋 Viewing logs..."
	@docker-compose logs -f

# Development
dev: ## Start development environment with hot reload
	@echo "🔧 Starting development environment..."
	@docker-compose -f docker-compose.dev.yml up

test: ## Run all tests
	@echo "🧪 Running tests..."
	@docker-compose exec streamlit-app python -m pytest tests/ -v

test-unit: ## Run unit tests only
	@echo "🧪 Running unit tests..."
	@docker-compose exec streamlit-app python -m pytest tests/unit/ -v

test-integration: ## Run integration tests only
	@echo "🧪 Running integration tests..."
	@docker-compose exec streamlit-app python -m pytest tests/integration/ -v

lint: ## Run linting checks
	@echo "🔍 Running linting checks..."
	@docker-compose exec streamlit-app flake8 app/ core/
	@docker-compose exec streamlit-app pylint app/ core/

format: ## Format code with black
	@echo "✨ Formatting code..."
	@docker-compose exec streamlit-app black app/ core/ tests/

# Database operations
db-migrate: ## Run database migrations
	@echo "🗄️ Running database migrations..."
	@docker-compose exec postgres psql -U $$POSTGRES_USER -d $$POSTGRES_DB -f /app/database/migrations/latest.sql

db-backup: ## Backup database
	@echo "💾 Backing up database..."
	@mkdir -p backups
	@docker-compose exec postgres pg_dump -U $$POSTGRES_USER $$POSTGRES_DB > backups/backup_$$(date +%Y%m%d_%H%M%S).sql

db-restore: ## Restore database from backup (use BACKUP_FILE=filename)
	@echo "🔄 Restoring database..."
	@docker-compose exec postgres psql -U $$POSTGRES_USER -d $$POSTGRES_DB < $(BACKUP_FILE)

# Cache operations
cache-clear: ## Clear Redis cache
	@echo "🧹 Clearing cache..."
	@docker-compose exec redis redis-cli FLUSHALL

cache-stats: ## Show cache statistics
	@echo "📊 Cache statistics..."
	@docker-compose exec redis redis-cli INFO memory

# AI operations
ai-model-update: ## Update AI model
	@echo "🤖 Updating AI model..."
	@docker-compose exec ollama ollama pull qwen2.5:0.5b

ai-model-list: ## List available AI models
	@echo "📋 Available AI models..."
	@docker-compose exec ollama ollama list

# Documentation
docs: ## Generate documentation
	@echo "📚 Generating documentation..."
	@docker-compose exec streamlit-app sphinx-build -b html docs/ docs/_build/html

docs-serve: ## Serve documentation locally
	@echo "🌐 Serving documentation..."
	@cd docs/_build/html && python -m http.server 8080

# Deployment
deploy-staging: ## Deploy to staging environment
	@echo "🚀 Deploying to staging..."
	@docker-compose -f docker-compose.staging.yml up -d

deploy-prod: ## Deploy to production environment
	@echo "🚀 Deploying to production..."
	@docker-compose -f docker-compose.prod.yml up -d

# Monitoring
health: ## Check service health
	@echo "🏥 Checking service health..."
	@curl -f http://localhost:8501/health || echo "❌ Application health check failed"
	@docker-compose exec postgres pg_isready -U $$POSTGRES_USER || echo "❌ Database health check failed"
	@docker-compose exec redis redis-cli ping || echo "❌ Redis health check failed"

status: ## Show service status
	@echo "📊 Service status..."
	@docker-compose ps

metrics: ## Show system metrics
	@echo "📈 System metrics..."
	@docker stats --no-stream

# Maintenance
clean: ## Clean up containers, images, and volumes
	@echo "🧹 Cleaning up..."
	@docker-compose down -v --remove-orphans
	@docker system prune -f
	@docker volume prune -f

clean-all: ## Clean up everything including images
	@echo "🧹 Deep cleaning..."
	@docker-compose down -v --remove-orphans
	@docker system prune -af
	@docker volume prune -f

reset: ## Reset the entire environment
	@echo "🔄 Resetting environment..."
	@make clean-all
	@make setup
	@make start

# Security
security-scan: ## Run security scans
	@echo "🔒 Running security scans..."
	@docker-compose exec streamlit-app bandit -r app/ core/
	@docker-compose exec streamlit-app safety check

# Performance
benchmark: ## Run performance benchmarks
	@echo "⚡ Running benchmarks..."
	@docker-compose exec streamlit-app python scripts/benchmark.py

profile: ## Profile application performance
	@echo "📊 Profiling application..."
	@docker-compose exec streamlit-app python -m cProfile -o profile.stats scripts/profile_app.py

# Data operations
data-validate: ## Validate input data format
	@echo "✅ Validating data..."
	@docker-compose exec streamlit-app python scripts/validate_data.py data_files/input/

data-sample: ## Generate sample data for testing
	@echo "📊 Generating sample data..."
	@docker-compose exec streamlit-app python scripts/generate_sample_data.py

# Backup and restore
backup: ## Full system backup
	@echo "💾 Creating full system backup..."
	@make db-backup
	@tar -czf backups/full_backup_$$(date +%Y%m%d_%H%M%S).tar.gz data/ cache/ logs/

restore: ## Restore from backup (use BACKUP_FILE=filename)
	@echo "🔄 Restoring from backup..."
	@tar -xzf $(BACKUP_FILE) -C ./

# Version and release
version: ## Show current version
	@echo "📋 Current version..."
	@cat VERSION || echo "Version file not found"

release: ## Create a new release
	@echo "🎉 Creating new release..."
	@./scripts/release.sh

# Quick commands
up: start ## Alias for start
down: stop ## Alias for stop
ps: status ## Alias for status