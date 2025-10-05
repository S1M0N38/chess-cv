.DEFAULT_GOAL := help
.PHONY: help install installdev lint format typecheck test quality clean docs all eval

# Colors for output
YELLOW := \033[33m
GREEN := \033[32m
BLUE := \033[34m
RED := \033[31m
RESET := \033[0m

# Project variables
PYTHON := python
UV := uv
RUFF := ruff
TYPECHECK := basedpyright
MDFORMAT := mdformat

help: ## Show this help message
	@echo "$(BLUE)Chess CV Development Makefile$(RESET)"
	@echo ""
	@echo "$(YELLOW)Available targets:$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-18s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install package with all dependencies
	@echo "$(YELLOW)Installing all dependencies...$(RESET)"
	$(UV) sync --all-extras

# Code quality targets
lint: ## Run ruff linter (check only)
	@echo "$(YELLOW)Running ruff linter...$(RESET)"
	$(RUFF) check --fix --select I .
	$(RUFF) check --fix .

format: ## Run ruff and mdformat formatters
	@echo "$(YELLOW)Running ruff formatter...$(RESET)"
	$(RUFF) check --select I --fix .
	$(RUFF) format .
	@echo "$(YELLOW)Running mdformat formatter...$(RESET)"
	$(MDFORMAT) .

typecheck: ## Run type checker
	@echo "$(YELLOW)Running type checker...$(RESET)"
	$(TYPECHECK)

quality: lint typecheck format ## Run all code quality checks
	@echo "$(GREEN)✓ All checks completed$(RESET)"

test: ## Run tests
	@echo "$(YELLOW)Running tests...$(RESET)"
	pytest

all: lint format typecheck test ## Run all code quality checks and tests
	@echo "$(GREEN)✓ All checks completed$(RESET)"

# Documentation targets
docs: ## Serve documentation locally
	@echo "$(YELLOW)Serving documentation at http://127.0.0.1:8000$(RESET)"
	mkdocs serve

# Clean targets
clean: ## Clean build artifacts
	@echo "$(YELLOW)Cleaning build artifacts...$(RESET)"
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .coverage htmlcov/ coverage.xml
	rm -rf .ruff_cache/
	rm -rf site/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)✓ Cleanup completed$(RESET)"

# Evaluation targets
eval: ## Evaluate model over multiple datasets
	@echo "$(YELLOW)Evaluating model on test data...$(RESET)"
	python -m chess_cv.test --test-dir data/test --output-dir evals/test
	@echo "$(YELLOW)Evaluating model on openboard dataset...$(RESET)"
	python -m chess_cv.test --hf-test-dir S1M0N38/chess-cv-openboard --output-dir evals/openboard
