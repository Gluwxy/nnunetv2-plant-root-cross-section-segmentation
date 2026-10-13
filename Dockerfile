# Use a lightweight base image
FROM python:3.11-slim-bookworm

# Copy uv directly from the official image for maximum performance
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install required system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy package management files
COPY pyproject.toml uv.lock* ./

# Install dependencies using uv, leveraging Docker layer caching
RUN uv sync --frozen --no-install-project --no-dev

# Copy the source code and the Hydra config (src.main needs both)
COPY src/ ./src/
COPY cfg/ ./cfg/

# Install the project itself
RUN uv sync --frozen --no-dev

# Ensure commands use the created virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# nnU-Net paths inside the mounted data/ directory, for nnUNetv2_* commands run outside src.main
# (src.main also sets them through nnUNetV2Runner)
ENV nnUNet_raw=/app/data/nnUNet_raw \
    nnUNet_preprocessed=/app/data/nnUNet_preprocessed \
    nnUNet_results=/app/data/nnUNet_results

# Containers run with the host user id (see Makefile), which has no home directory in the image:
# give torch, matplotlib and wandb a writable place for their caches
ENV HOME=/tmp
