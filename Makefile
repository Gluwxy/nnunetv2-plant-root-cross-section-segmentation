# Get the absolute path to the project root directory
ROOT_DIR := $(abspath $(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

# Docker configuration variables
IMAGE_NAME := nnunet-trainer
ENV_FILE := $(if $(wildcard $(ROOT_DIR)/.env),--env-file $(ROOT_DIR)/.env,)
# --ipc=host is critical for PyTorch to prevent shared memory limits from crashing the dataloaders.
# No -it, so training can run unattended (nohup, tmux, ssh without a TTY).
# --user writes files in the mounted directories with the host user instead of root.
DOCKER_CMD := docker run --rm --gpus all --ipc=host --user $(shell id -u):$(shell id -g) $(ENV_FILE)
# Data augmentation workers on the VM: all CPU cores but one
VM_N_PROC_DA := $(shell echo $$(( $$(getconf _NPROCESSORS_ONLN) - 1 )))
# Extra Hydra overrides, e.g. make train-vm ARGS="monai.train.fold=1"
ARGS ?=

# Declare phony targets to avoid conflicts with files of the same name
.PHONY: all download test convert integrity visualize summarize \
        docker-build download-vm convert-vm preprocess-vm train-vm visualize-vm summarize-vm

# Default target
all: download convert test integrity

# =====================================================================
# LOCAL WORKFLOW
# These targets run on your local machine using 'uv' for dependency
# management and rely on 'direnv' for the nnU-Net environment variables.
# =====================================================================

download:
	@echo "Downloading dataset from Roboflow..."
	uv run python -m src.utils.download $(ARGS)

test:
	@echo "Testing fold leakage..."
	uv run pytest

convert:
	@echo "Converting Roboflow COCO Segmentation to nnU-net v2 format..."
	uv run python -m src.utils.convert $(ARGS)

integrity:
	@echo "Checking dataset integrity..."
	uv run python -m src.main monai.train.enabled=false $(ARGS)

# Examples:
#   make visualize
#   make visualize ARGS="monai.runner.trainer_class_name=nnUNetTrainerMetrics_250epochs"
#   make visualize ARGS="monai.visualize.folds=[fold_1]"
visualize:
	@echo "Generating 1x5 validation prediction visualizations..."
	uv run python -m src.utils.visualize $(ARGS)

summarize:
	@echo "Summarizing cross-validation results..."
	uv run python -m src.utils.summarize $(ARGS)


# =====================================================================
# CLOUD / VM WORKFLOW (DOCKER)
# These targets are designed for heavy training on cloud instances (e.g., GCP).
# They map the local 'data/' directory into the container to isolate I/O.
# =====================================================================

docker-build:
	@echo "Building Docker image..."
	docker build -t $(IMAGE_NAME) .

# Reusable macro to execute commands inside the Docker container.
# It mounts coco_semantic_segmentation_dataset/ (Roboflow export), data/ (nnU-Net datasets
# and checkpoints), results/ (1x5 visualization figures and CV summary) and logs/ (Hydra run outputs).
# The directories are created first so Docker does not create them as root.
define run_in_docker
	mkdir -p $(ROOT_DIR)/coco_semantic_segmentation_dataset $(ROOT_DIR)/data $(ROOT_DIR)/results $(ROOT_DIR)/logs
	$(DOCKER_CMD) \
		-v $(ROOT_DIR)/coco_semantic_segmentation_dataset:/app/coco_semantic_segmentation_dataset \
		-v $(ROOT_DIR)/data:/app/data \
		-v $(ROOT_DIR)/results:/app/results \
		-v $(ROOT_DIR)/logs:/app/logs \
		$(IMAGE_NAME) $(1)
endef

download-vm:
	@echo "Downloading dataset from Roboflow inside Docker..."
	$(call run_in_docker, python -m src.utils.download $(ARGS))

convert-vm:
	@echo "Converting Roboflow COCO Segmentation to nnU-Net v2 format inside Docker..."
	$(call run_in_docker, python -m src.utils.convert $(ARGS))

# Both targets go through src.main, so they use the same config as local runs (cfg/monai/default.yaml)
preprocess-vm:
	@echo "Running nnU-Net preprocessing inside Docker..."
	$(call run_in_docker, python -m src.main monai.train.enabled=false $(ARGS))

train-vm:
	@echo "Starting nnU-Net training on the VM with GPUs..."
	$(call run_in_docker, python -m src.main monai.preprocess.enabled=false \
		monai.train.device=cuda monai.train.n_proc_da=$(VM_N_PROC_DA) $(ARGS))

visualize-vm:
	@echo "Generating 1x5 validation prediction visualizations inside Docker..."
	$(call run_in_docker, python -m src.utils.visualize $(ARGS))

summarize-vm:
	@echo "Summarizing cross-validation results inside Docker..."
	$(call run_in_docker, python -m src.utils.summarize $(ARGS))
