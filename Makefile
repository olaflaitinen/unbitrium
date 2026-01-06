# Makefile for Unbitrium
# Provides common development, testing, and deployment commands

.PHONY: help install install-dev install-docs clean lint format type-check \
        test test-cov test-fast docs docs-serve build publish release

# Default target
help:
	@echo "Unbitrium Development Commands"
	@echo "=============================="
	@echo ""
	@echo "Installation:"
	@echo "  make install       Install package"
	@echo "  make install-dev   Install with dev dependencies"
	@echo "  make install-docs  Install with docs dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make format        Format code with black and isort"
	@echo "  make lint          Run linting with ruff"
	@echo "  make type-check    Run type checking with mypy"
	@echo "  make check         Run all checks (format, lint, type-check)"
	@echo ""
	@echo "Testing:"
	@echo "  make test          Run all tests"
	@echo "  make test-cov      Run tests with coverage"
	@echo "  make test-fast     Run fast tests only"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs          Build documentation"
	@echo "  make docs-serve    Serve documentation locally"
	@echo ""
	@echo "Build & Release:"
	@echo "  make build         Build package"
	@echo "  make publish       Publish to PyPI"
	@echo "  make clean         Clean build artifacts"

# ====================
# Installation
# ====================

install:
	pip install .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

install-docs:
	pip install -e ".[dev,docs]"

install-all:
	pip install -e ".[dev,docs]"
	pre-commit install

# ====================
# Code Quality
# ====================

format:
	black src/ tests/
	isort src/ tests/

lint:
	ruff check src/ tests/

type-check:
	mypy src/

check: format lint type-check
	@echo "All checks passed!"

# ====================
# Testing
# ====================

test:
	pytest

test-cov:
	pytest --cov=src/unbitrium --cov-report=html --cov-report=term-missing

test-fast:
	pytest -m "not slow" -x

test-parallel:
	pytest -n auto

# ====================
# Documentation
# ====================

docs:
	mkdocs build

docs-serve:
	mkdocs serve

docs-deploy:
	mkdocs gh-deploy

# ====================
# Build & Release
# ====================

build: clean
	python -m build

publish: build
	twine upload dist/*

publish-test: build
	twine upload --repository testpypi dist/*

# ====================
# Cleanup
# ====================

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf src/*.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf site/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean-all: clean
	rm -rf .venv/
	rm -rf venv/

# ====================
# Development
# ====================

pre-commit:
	pre-commit run --all-files

pre-commit-update:
	pre-commit autoupdate

dev-setup: install-dev
	@echo "Development environment ready!"

# ====================
# Version
# ====================

version:
	@python -c "import unbitrium; print(unbitrium.__version__)"

# ====================
# Benchmarks
# ====================

benchmark:
	python benchmarks/run_benchmark.py

benchmark-all:
	python benchmarks/run_benchmark.py --config benchmarks/configs/full.yaml
