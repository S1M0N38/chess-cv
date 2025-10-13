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
	$(MDFORMAT) ./docs README.md AGENTS.md

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
	@echo "$(YELLOW)Generating the augmentation examples...$(RESET)"
	@python docs/assets/generate_augmentations_examples.py
	@echo "$(YELLOW)Serving documentation at http://127.0.0.1:8000$(RESET)"
	@mkdocs serve

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
	@echo "$(YELLOW)=================================== PIECES =====================================$(RESET)"
	@echo "$(YELLOW)Evaluating pieces model on test data...$(RESET)"
	chess-cv test pieces --output-dir evals/pieces/test --checkpoint ./src/chess_cv/weights/pieces.safetensors
	@echo "$(YELLOW)Evaluating pieces model on openboard dataset...$(RESET)"
	chess-cv test pieces --hf-test-dir S1M0N38/chess-cv-openboard --output-dir evals/pieces/openboard --checkpoint ./src/chess_cv/weights/pieces.safetensors 
	@echo "$(YELLOW)Evaluating pieces model on chessvision dataset...$(RESET)"
	chess-cv test pieces --hf-test-dir S1M0N38/chess-cv-chessvision --concat-splits --output-dir evals/pieces/chessvision --checkpoint ./src/chess_cv/weights/pieces.safetensors
	@echo "$(YELLOW)=================================== ARROWS =====================================$(RESET)"
	@echo "$(YELLOW)Evaluating arrow model on test data...$(RESET)"
	chess-cv test arrows --output-dir evals/arrows/test --checkpoint ./src/chess_cv/weights/arrows.safetensors
	@echo "$(YELLOW)=================================== SNAP =====================================$(RESET)"
	@echo "$(YELLOW)Evaluating snap model on test data...$(RESET)"
	chess-cv test snap --output-dir evals/snap/test --checkpoint ./src/chess_cv/weights/snap.safetensors
	@echo "$(YELLOW)Evaluating snap model on chessvision dataset...$(RESET)"
