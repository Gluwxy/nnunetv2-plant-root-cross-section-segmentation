# Root Cross-Section Segmentation

Semantic segmentation of plant root cross-sections (`root`, `stele`, `xylem`) using **nnU-Net v2**, orchestrated through MONAI's `nnUNetV2Runner` and configured with **Hydra** (`cfg/`).

## Setup

Create a `.env` file in the project root with your Roboflow API key (and optionally your Weights & Biases API key for experiment tracking):

```env
ROBOFLOW_API_KEY=your_roboflow_api_key_here
WANDB_API_KEY=your_wandb_api_key_here
```

> **Important:** Write the key values **without quotes** (`"` or `'`). `docker run --env-file` does not strip quotes and will pass them literally as part of the API keys.

## Local Workflow (`uv`)

Requires [uv](https://docs.astral.sh/uv/) (and optionally `direnv` for automatic environment variable loading from `.envrc`).

Install dependencies:

```bash
uv sync
```

Run the full data preparation and verification pipeline (`download` → `convert` → `test` → `integrity`):

```bash
make all
```

Or run each step individually:

| Command | Description |
| :--- | :--- |
| `make download` | Download the COCO segmentation dataset from Roboflow into `coco_semantic_segmentation_dataset/` (skipped if a split with `_annotations.coco.json` already exists; delete `coco_semantic_segmentation_dataset/` to force a re-download). |
| `make convert` | Rasterize COCO polygons in class order (`root` → `stele` → `xylem`) and export the dataset to `data/nnUNet_raw/` with specimen-grouped folds (`splits_final.json`). |
| `make test` | Run `pytest` unit and dataset checks (class nesting, exclusive masks, zero specimen group leakage, and no near-duplicate images across the test set and CV folds). |
| `make integrity` | Extract dataset fingerprint, generate 2D plans, and preprocess into `data/nnUNet_preprocessed/` (`monai.train.enabled=false`). |
| `make visualize` | Generate `1x5` validation comparison figures into `results/fold_<n>/` for all trained folds discovered on disk. |

To train locally (e.g., on Apple Silicon with `mps`):

```bash
uv run python -m src.main monai.preprocess.enabled=false
```

## Cloud / VM Workflow (Docker + GPU)

Designed for GPU instances (e.g., a GCP VM with an NVIDIA L4).

**VM Prerequisites:**
- Clone the repository and create `.env` in the project root (`ROBOFLOW_API_KEY=...` and optionally `WANDB_API_KEY=...`, without quotes).
- `make`, **Docker**, and **NVIDIA Container Toolkit** installed on the host.

All `*-vm` targets automatically mount four host directories into the container so data and outputs persist with your host user permissions:
- `coco_semantic_segmentation_dataset/` → `/app/coco_semantic_segmentation_dataset` (Roboflow COCO export)
- `data/` → `/app/data` (nnU-Net raw data, preprocessed data, and trained models)
- `results/` → `/app/results` (`1x5` validation comparison figures per fold)
- `logs/` → `/app/logs` (Hydra execution logs)

### Step-by-Step Execution

1. **Build the Docker image:**
   ```bash
   make docker-build
   ```

2. **Download and convert the dataset:**
   ```bash
   make download-vm
   make convert-vm
   ```
   *(If `coco_semantic_segmentation_dataset/` already contains a split with annotations, `make download-vm` skips the download. Delete the folder first if you need to download a fresh copy).*

3. **Run fingerprint extraction, planning, and preprocessing:**
   ```bash
   make preprocess-vm
   ```

4. **Start GPU training:**
   ```bash
   make train-vm
   ```

5. **Generate validation visualizations (`1x5`):**
   ```bash
   make visualize-vm
   ```

### Passing Hydra Overrides (`ARGS`)

All `Makefile` pipeline targets accept Hydra configuration overrides via `ARGS`:

```bash
# Re-convert overwriting an existing export (always re-run preprocess-vm afterwards
# so the dataset fingerprint, plans, and preprocessed data are updated)
make convert-vm ARGS="export.overwrite=true"
make preprocess-vm

# Train fold 1 for 250 epochs with Weights & Biases logging enabled
make train-vm ARGS="logger.enabled=true monai.train.fold=1 monai.runner.trainer_class_name=nnUNetTrainerMetrics_250epochs"

# Resume training from the latest checkpoint
make train-vm ARGS="monai.train.continue_training=true"

# Train without torch.compile (no compilation time on the first epoch, but slower epochs)
make train-vm ARGS="monai.train.compile=false"

# Visualize all trained folds of the 250-epoch trainer, or a single specific fold
make visualize-vm ARGS="monai.runner.trainer_class_name=nnUNetTrainerMetrics_250epochs"
make visualize-vm ARGS="monai.visualize.folds=[fold_1]"
```

## Experiment Tracking (Weights & Biases)

`src/main.py` configures nnU-Net v2's built-in `WandbLogger` using `cfg/logger/wandb.yaml` (`enabled`, `project`, `mode`). By default, `logger.enabled` is `false` so local checks and preprocessing runs do not require a W&B account.

To log training runs to W&B:
1. Add `WANDB_API_KEY=your_wandb_api_key_here` (without quotes) to `.env`.
2. Pass `logger.enabled=true` via `ARGS` (or set `enabled: true` in `cfg/logger/wandb.yaml`):
   ```bash
   make train-vm ARGS="logger.enabled=true"
   ```

**Run organization:** each experiment (trainer + configuration, e.g. `nnUNetTrainerMetrics_250epochs__2d`) is a W&B *group* with one run per fold (`fold_0` … `fold_4`), plus tags for the configuration, fold and trainer (`group`, `name` and `tags` in `cfg/logger/wandb.yaml`). Group the runs table by *Group* to compare and average the folds of an experiment. Train all five folds with:

```bash
for f in 0 1 2 3 4; do
  make train-vm ARGS="logger.enabled=true monai.train.fold=$f" || break
done
```

**Metrics:** the default trainer, `nnUNetTrainerMetrics_<N>epochs` (`src/trainers/`), extends nnU-Net's `nnUNetTrainer`, which only logs losses, learning rate and the pseudo Dice per class. It adds, per class (`root`, `stele`, `xylem`) and as the foreground mean:

| W&B key | When | Computed on |
| :--- | :--- | :--- |
| `val_dice/*`, `val_iou/*`, `val_precision/*`, `val_recall/*` | Every epoch (charts) | Validation patches sampled during training (pseudo metrics) |
| `final_val/<metric>/*` (run summary) and `final_val/per_case` (table) | Once, after training | Full validation images (`validation/summary.json`), averaged over cases |

nnU-Net finds these trainers through `nnUNet_extTrainer` (set by `src/main.py`, the Dockerfile and `.envrc`); set it too when running `nnUNetv2_predict` by hand on these models. nnU-Net's own `nnUNetTrainer_<N>epochs` classes still work, without the extra metrics.

## Outputs & Logs

- **nnU-Net checkpoints and validation results:** `data/nnUNet_results/Dataset001_RootVascularBundle/`
- **Validation `1x5` comparison figures:** `results/fold_<n>/`
- **Preprocessed dataset, fingerprint, and plans:** `data/nnUNet_preprocessed/Dataset001_RootVascularBundle/`
- **Hydra run logs and resolved configs:** `logs/runs/<YYYY-MM-DD>/<HH-MM-SS>/`

## Project Structure

```text
├── cfg/                  # Hydra configuration (paths, export, monai, roboflow, logger, hydra)
├── src/
│   ├── main.py           # Planning, preprocessing, and training entry point
│   ├── trainers/         # Custom nnU-Net trainers (extra W&B metrics), found via nnUNet_extTrainer
│   └── utils/
│       ├── download.py   # Roboflow dataset downloader
│       ├── convert.py    # COCO-to-nnU-Net converter and grouped fold splitter
│       ├── coco.py       # COCO reader, class mapping, polygon rasterization, and group extraction
│       ├── visualize.py  # 1x5 validation prediction visualizer
│       └── resolvers.py  # Custom OmegaConf resolvers (splits, trained folds, and device)
├── tests/                # Synthetic and real-dataset validation tests
├── notebooks/            # Exploratory data analysis
├── Dockerfile            # Container image for cloud/VM execution
└── Makefile              # Local and Docker pipeline targets
```
