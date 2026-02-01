# =============================================================================
# Unbitrium Dockerfile
# =============================================================================
# Multi-stage build for production-ready container image.
#
# Build:
#   docker build -t unbitrium:latest .
#
# Run:
#   docker run -it --gpus all unbitrium:latest python -c "import unbitrium"
#
# See: https://docs.docker.com/engine/reference/builder/
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder
# -----------------------------------------------------------------------------
FROM python:3.14-slim AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install dependencies
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install package
RUN pip install .

# -----------------------------------------------------------------------------
# Stage 2: Runtime
# -----------------------------------------------------------------------------
FROM python:3.14-slim AS runtime

# Labels
LABEL org.opencontainers.image.title="Unbitrium" \
      org.opencontainers.image.description="Production-grade Federated Learning Simulator" \
      org.opencontainers.image.authors="Olaf Yunus Laitinen Imanov <oyli@dtu.dk>" \
      org.opencontainers.image.source="https://github.com/olaflaitinen/unbitrium" \
      org.opencontainers.image.licenses="EUPL-1.2" \
      org.opencontainers.image.vendor="Technical University of Denmark"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1

# Create non-root user
RUN useradd --create-home --shell /bin/bash unbitrium

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Switch to non-root user
USER unbitrium

# Verify installation
RUN python -c "import unbitrium; print(f'Unbitrium {unbitrium.__version__} installed')"

# Default command
CMD ["python", "-c", "import unbitrium; print(f'Unbitrium {unbitrium.__version__}')"]

# -----------------------------------------------------------------------------
# Stage 3: Development (optional)
# -----------------------------------------------------------------------------
FROM runtime AS development

# Switch to root for dev dependencies
USER root

# Install development dependencies
RUN pip install pytest pytest-cov black isort mypy ruff

# Switch back to non-root user
USER unbitrium

# Default command for development
CMD ["pytest", "-v"]
