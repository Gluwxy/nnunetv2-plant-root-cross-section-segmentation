# Use a lightweight base image
FROM python:3.11-slim-bookworm

# Copy uv directly from the official image for maximum performance
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install required system dependencies (git is needed by uv to fetch nnunetv2; gcc and the C headers
# are needed by Triton to build its CUDA helpers when nnU-Net uses torch.compile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    git \
    libc6-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy package management files
COPY pyproject.toml uv.lock* ./

# Install dependencies using uv, leveraging Docker layer caching
RUN uv sync --frozen --no-install-project --no-dev --no-group notebooks

# Copy the source code and the Hydra config (src.main needs both)
COPY src/ ./src/
COPY cfg/ ./cfg/

# Install the project itself
RUN uv sync --frozen --no-dev --no-group notebooks

# Ensure commands use the created virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Print logs immediately: without a TTY (docker run without -t), Python buffers stdout and nnU-Net's
# epoch logs only show up in large blocks or when the process exits. Inherited by nnUNetv2_train.
ENV PYTHONUNBUFFERED=1

# nnU-Net paths inside the mounted data/ directory, for nnUNetv2_* commands run outside src.main
# (src.main also sets them through nnUNetV2Runner)
ENV nnUNet_raw=/app/data/nnUNet_raw \
    nnUNet_preprocessed=/app/data/nnUNet_preprocessed \
    nnUNet_results=/app/data/nnUNet_results

# Containers run with the host user id (see Makefile), which has no home directory in the image:
# give torch, matplotlib and wandb a writable place for their caches
ENV HOME=/tmp
