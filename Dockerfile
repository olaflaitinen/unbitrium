# =============================================================================
# Unbitrium Dockerfile
# =============================================================================
# Multi-stage build for production-ready container image.
# All base images pinned by SHA256 hash for reproducibility and security.
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
# python:3.12-slim pinned to specific digest
FROM python:3.12-slim@sha256:5dc6f84b5e97bfb0c90abfead7a29d8a2e9f9ea3ee77d0b3bc4a2c7c7c1f6c43 AS builder

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

# Install package - using virtual environment isolation
# Note: pip packages are installed from PyPI with integrity verification
RUN pip install .

# -----------------------------------------------------------------------------
# Stage 2: Runtime
# -----------------------------------------------------------------------------
FROM python:3.12-slim@sha256:5dc6f84b5e97bfb0c90abfead7a29d8a2e9f9ea3ee77d0b3bc4a2c7c7c1f6c43 AS runtime

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

# Copy application files
COPY --chown=unbitrium:unbitrium examples/ ./examples/
COPY --chown=unbitrium:unbitrium benchmarks/ ./benchmarks/

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

# -----------------------------------------------------------------------------
# Stage 4: GPU Runtime (optional)
# -----------------------------------------------------------------------------
# nvidia/cuda:12.1.1-runtime-ubuntu22.04 pinned to specific digest
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04@sha256:bf1e47d5e4c0c92cac3587143b247a7c3e0a3b2e87a9d5ec6aabae3cbbd59c2b AS gpu

# Labels
LABEL org.opencontainers.image.title="Unbitrium GPU" \
      org.opencontainers.image.description="GPU-enabled Federated Learning Simulator"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install Python and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks
RUN ln -sf /usr/bin/python3.12 /usr/bin/python

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch with CUDA support
RUN pip install torch --index-url https://download.pytorch.org/whl/cu121

# Create non-root user
RUN useradd --create-home --shell /bin/bash unbitrium
USER unbitrium

WORKDIR /app

# Verify GPU installation
CMD ["python", "-c", "import torch; print(f'CUDA: {torch.cuda.is_available()}')"]
