# AIOps Platform - Self-Healing Enterprise Application Monitoring
# Makefile for development, testing, and deployment

.PHONY: help install dev-up dev-down build test lint clean deploy-local deploy-prod health-check

# Variables
COMPOSE_FILE := docker-compose.yml
ENVIRONMENT := development
NAMESPACE := aiops-platform
REGISTRY := ghcr.io/your-org/aiops-platform

# Default target
help: ## Show this help message
	@echo "AIOps Platform - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""

# =============================================================================
# DEVELOPMENT COMMANDS
# =============================================================================

install: ## Install all dependencies
	@echo "🔧 Installing dependencies..."
	@$(MAKE) install-go
	@$(MAKE) install-python
	@$(MAKE) install-node
	@$(MAKE) setup-git-hooks
	@echo "✅ All dependencies installed!"

install-go: ## Install Go dependencies
	@echo "📦 Installing Go dependencies..."
	@cd services/api-gateway && go mod download
	@cd services/monitoring-service && go mod download
	@cd services/remediation-service && go mod download
	@cd services/incident-service && go mod download
	@cd services/compliance-service && go mod download

install-python: ## Install Python dependencies
	@echo "🐍 Installing Python dependencies..."
	@cd services/prediction-service && pip install -r requirements.txt
	@cd ai-ml && pip install -r requirements.txt
	@cd data-pipeline && pip install -r requirements.txt

install-node: ## Install Node.js dependencies
	@echo "📦 Installing Node.js dependencies..."
	@cd services/notification-service && npm install
	@cd web-dashboard/frontend && npm install
	@cd web-dashboard/backend && npm install

setup-git-hooks: ## Setup Git hooks for code quality
	@echo "🔗 Setting up Git hooks..."
	@cp scripts/development/pre-commit .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit

dev-up: ## Start development environment
	@echo "🚀 Starting development environment..."
	@docker-compose -f $(COMPOSE_FILE) up -d
	@echo "⏳ Waiting for services to be ready..."
	@sleep 30
	@$(MAKE) health-check
	@echo "✅ Development environment is ready!"
	@echo "🌐 Access points:"
	@echo "  - API Gateway: http://localhost:8080"
	@echo "  - Grafana: http://localhost:3000 (admin/admin)"
	@echo "  - Prometheus: http://localhost:9090"
	@echo "  - Kibana: http://localhost:5601"
	@echo "  - Web Dashboard: http://localhost:3001"

dev-down: ## Stop development environment
	@echo "🛑 Stopping development environment..."
	@docker-compose -f $(COMPOSE_FILE) down
	@echo "✅ Development environment stopped!"

dev-restart: ## Restart development environment
	@$(MAKE) dev-down
	@$(MAKE) dev-up

dev-logs: ## View logs from all services
	@docker-compose -f $(COMPOSE_FILE) logs -f

dev-logs-service: ## View logs from specific service (usage: make dev-logs-service SERVICE=api-gateway)
	@docker-compose -f $(COMPOSE_FILE) logs -f $(SERVICE)

# =============================================================================
# BUILD COMMANDS
# =============================================================================

build: ## Build all services
	@echo "🔨 Building all services..."
	@$(MAKE) build-go-services
	@$(MAKE) build-python-services
	@$(MAKE) build-node-services
	@echo "✅ All services built!"

build-go-services: ## Build Go services
	@echo "🔨 Building Go services..."
	@cd services/api-gateway && docker build -t aiops/api-gateway:latest .
	@cd services/monitoring-service && docker build -t aiops/monitoring-service:latest .
	@cd services/remediation-service && docker build -t aiops/remediation-service:latest .
	@cd services/incident-service && docker build -t aiops/incident-service:latest .
	@cd services/compliance-service && docker build -t aiops/compliance-service:latest .

build-python-services: ## Build Python services
	@echo "🔨 Building Python services..."
	@cd services/prediction-service && docker build -t aiops/prediction-service:latest .

build-node-services: ## Build Node.js services
	@echo "🔨 Building Node.js services..."
	@cd services/notification-service && docker build -t aiops/notification-service:latest .
	@cd web-dashboard/frontend && docker build -t aiops/web-dashboard:latest .

# =============================================================================
# TESTING COMMANDS
# =============================================================================

test: ## Run all tests
	@echo "🧪 Running all tests..."
	@$(MAKE) test-unit
	@$(MAKE) test-integration
	@$(MAKE) test-e2e
	@echo "✅ All tests completed!"

test-unit: ## Run unit tests
	@echo "🧪 Running unit tests..."
	@cd services/api-gateway && go test ./... -v
	@cd services/monitoring-service && go test ./... -v
	@cd services/prediction-service && python -m pytest tests/unit/ -v
	@cd services/notification-service && npm test

test-integration: ## Run integration tests
	@echo "🧪 Running integration tests..."
	@cd tests/integration && python -m pytest . -v

test-e2e: ## Run end-to-end tests
	@echo "🧪 Running end-to-end tests..."
	@cd tests/e2e && python -m pytest . -v

test-performance: ## Run performance tests
	@echo "🧪 Running performance tests..."
	@cd tests/performance && python -m pytest . -v

# =============================================================================
# CODE QUALITY COMMANDS
# =============================================================================

lint: ## Run linting for all services
	@echo "🔍 Running linting..."
	@$(MAKE) lint-go
	@$(MAKE) lint-python
	@$(MAKE) lint-node
	@echo "✅ Linting completed!"

lint-go: ## Lint Go code
	@echo "🔍 Linting Go code..."
	@cd services/api-gateway && golangci-lint run
	@cd services/monitoring-service && golangci-lint run
	@cd services/remediation-service && golangci-lint run
	@cd services/incident-service && golangci-lint run
	@cd services/compliance-service && golangci-lint run

lint-python: ## Lint Python code
	@echo "🔍 Linting Python code..."
	@cd services/prediction-service && black --check . && pylint --recursive=y .
	@cd ai-ml && black --check . && pylint --recursive=y .

lint-node: ## Lint Node.js code
	@echo "🔍 Linting Node.js code..."
	@cd services/notification-service && npm run lint
	@cd web-dashboard/frontend && npm run lint

format: ## Format all code
	@echo "✨ Formatting code..."
	@$(MAKE) format-go
	@$(MAKE) format-python
	@$(MAKE) format-node

format-go: ## Format Go code
	@cd services/api-gateway && go fmt ./...
	@cd services/monitoring-service && go fmt ./...
	@cd services/remediation-service && go fmt ./...
	@cd services/incident-service && go fmt ./...
	@cd services/compliance-service && go fmt ./...

format-python: ## Format Python code
	@cd services/prediction-service && black .
	@cd ai-ml && black .

format-node: ## Format Node.js code
	@cd services/notification-service && npm run format
	@cd web-dashboard/frontend && npm run format

# =============================================================================
# INFRASTRUCTURE COMMANDS
# =============================================================================

infra-plan: ## Plan infrastructure changes
	@echo "📋 Planning infrastructure changes..."
	@cd infrastructure/terraform && terraform plan -var-file="environments/$(ENVIRONMENT).tfvars"

infra-deploy: ## Deploy infrastructure
	@echo "🏗️ Deploying infrastructure for $(ENVIRONMENT)..."
	@cd infrastructure/terraform && terraform apply -var-file="environments/$(ENVIRONMENT).tfvars" -auto-approve

infra-destroy: ## Destroy infrastructure
	@echo "💥 Destroying infrastructure for $(ENVIRONMENT)..."
	@cd infrastructure/terraform && terraform destroy -var-file="environments/$(ENVIRONMENT).tfvars" -auto-approve

# =============================================================================
# KUBERNETES DEPLOYMENT COMMANDS
# =============================================================================

deploy-local: ## Deploy to local Kubernetes
	@echo "🚢 Deploying to local Kubernetes..."
	@kubectl create namespace $(NAMESPACE) --dry-run=client -o yaml | kubectl apply -f -
	@helm upgrade --install aiops-platform ./infrastructure/helm-charts/aiops-platform \
		--namespace $(NAMESPACE) \
		--values ./infrastructure/helm-charts/aiops-platform/values-local.yaml
	@echo "✅ Local deployment completed!"

deploy-staging: ## Deploy to staging environment
	@echo "🚢 Deploying to staging environment..."
	@kubectl create namespace $(NAMESPACE)-staging --dry-run=client -o yaml | kubectl apply -f -
	@helm upgrade --install aiops-platform-staging ./infrastructure/helm-charts/aiops-platform \
		--namespace $(NAMESPACE)-staging \
		--values ./infrastructure/helm-charts/aiops-platform/values-staging.yaml

deploy-prod: ## Deploy to production environment
	@echo "🚢 Deploying to production environment..."
	@kubectl create namespace $(NAMESPACE)-production --dry-run=client -o yaml | kubectl apply -f -
	@helm upgrade --install aiops-platform-production ./infrastructure/helm-charts/aiops-platform \
		--namespace $(NAMESPACE)-production \
		--values ./infrastructure/helm-charts/aiops-platform/values-production.yaml

# =============================================================================
# MONITORING & HEALTH COMMANDS
# =============================================================================

health-check: ## Check health of all services
	@echo "🏥 Performing health check..."
	@scripts/operations/health-check.sh

monitor: ## Open monitoring dashboard
	@echo "📊 Opening monitoring dashboard..."
	@open http://localhost:3000

logs: ## View aggregated logs
	@echo "📝 Viewing aggregated logs..."
	@kubectl logs -f -l app=aiops-platform -n $(NAMESPACE) --tail=100

metrics: ## View metrics dashboard
	@echo "📈 Opening metrics dashboard..."
	@open http://localhost:9090

# =============================================================================
# DATABASE COMMANDS
# =============================================================================

db-migrate: ## Run database migrations
	@echo "🗄️ Running database migrations..."
	@cd scripts/database && ./migrate.sh

db-seed: ## Seed database with sample data
	@echo "🌱 Seeding database with sample data..."
	@cd scripts/database && ./seed.sh

db-backup: ## Backup database
	@echo "💾 Backing up database..."
	@cd scripts/database && ./backup.sh $(ENVIRONMENT)

db-restore: ## Restore database from backup
	@echo "♻️ Restoring database from backup..."
	@cd scripts/database && ./restore.sh $(ENVIRONMENT) $(BACKUP_FILE)

# =============================================================================
# SECURITY COMMANDS
# =============================================================================

security-scan: ## Run security scans
	@echo "🔒 Running security scans..."
	@docker run --rm -v $(PWD):/src securecodewarrior/docker-image-scanner:latest /src

vulnerability-check: ## Check for vulnerabilities
	@echo "🛡️ Checking for vulnerabilities..."
	@cd services/api-gateway && go list -json -m all | nancy sleuth
	@cd services/prediction-service && safety check

# =============================================================================
# AI/ML COMMANDS
# =============================================================================

train-models: ## Train AI/ML models
	@echo "🤖 Training AI/ML models..."
	@cd ai-ml/training && python train_prediction_models.py
	@cd ai-ml/training && python train_rca_models.py

deploy-models: ## Deploy trained models
	@echo "🚀 Deploying trained models..."
	@cd ai-ml && python deploy_models.py

model-performance: ## Check model performance
	@echo "📊 Checking model performance..."
	@cd ai-ml && python evaluate_models.py

# =============================================================================
# UTILITY COMMANDS
# =============================================================================

clean: ## Clean up development environment
	@echo "🧹 Cleaning up..."
	@docker-compose -f $(COMPOSE_FILE) down -v
	@docker system prune -f
	@docker volume prune -f
	@echo "✅ Cleanup completed!"

generate-docs: ## Generate API documentation
	@echo "📚 Generating documentation..."
	@cd services/api-gateway && swag init
	@cd docs && ./generate-docs.sh

update-deps: ## Update all dependencies
	@echo "⬆️ Updating dependencies..."
	@cd services/api-gateway && go mod tidy && go get -u ./...
	@cd services/prediction-service && pip-review --auto
	@cd services/notification-service && npm update

backup: ## Backup entire platform
	@echo "💾 Creating platform backup..."
	@scripts/operations/backup.sh $(ENVIRONMENT)

version: ## Show version information
	@echo "AIOps Platform Version Information:"
	@echo "  Platform: v1.0.0"
	@echo "  Go: $(shell go version)"
	@echo "  Python: $(shell python --version)"
	@echo "  Node.js: $(shell node --version)"
	@echo "  Docker: $(shell docker --version)"
	@echo "  Kubernetes: $(shell kubectl version --client --short)"

# =============================================================================
# DEVELOPMENT UTILITIES
# =============================================================================

shell-api-gateway: ## Open shell in API Gateway container
	@docker-compose exec api-gateway /bin/sh

shell-prediction: ## Open shell in Prediction Service container
	@docker-compose exec prediction-service /bin/bash

shell-postgres: ## Open PostgreSQL shell
	@docker-compose exec postgres psql -U aiops_user -d aiops_platform

shell-redis: ## Open Redis shell
	@docker-compose exec redis redis-cli

# Default environment
ifndef ENVIRONMENT
ENVIRONMENT=development
endif 