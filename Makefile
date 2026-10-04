.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test test-slow cov bench check build clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install the package and dev tools, and set up git hooks
	uv sync --all-extras
	uv run pre-commit install

lint: ## Lint and check formatting
	uv run ruff check .
	uv run ruff format --check .

format: ## Auto-fix lint issues and format code
	uv run ruff check --fix .
	uv run ruff format .

typecheck: ## Static type checking
	uv run mypy

test: ## Run the test suite
	uv run pytest

test-slow: ## Run slow agent-strength tests
	uv run pytest -m slow

cov: ## Run tests with coverage
	uv run pytest --cov --cov-report=term-missing --cov-report=xml

bench: ## Run performance benchmarks
	uv run pytest --benchmark-only --benchmark-enable

check: lint typecheck cov ## Everything CI runs

build: ## Build sdist and wheel
	uv build

clean: ## Remove build and cache artifacts
	rm -rf build dist .mypy_cache .pytest_cache .ruff_cache .coverage coverage.xml htmlcov
