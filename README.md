# Root Cross-Section Segmentation

Semantic segmentation of plant root cross-sections (`root`, `stele`, `xylem`) using **nnU-Net v2**, orchestrated through MONAI's `nnUNetV2Runner` and configured with **Hydra** (`cfg/`).

## Setup

Create a `.env` file in the project root with your Roboflow API key, **without quotes** (the Docker targets pass it with `docker run --env-file`, which keeps quotes as part of the value):

```bash
ROBOFLOW_API_KEY=your_api_key_here
```

## Local Workflow (`uv`)

Requires [uv](https://docs.astral.sh/uv/). `direnv` is optional: `.envrc` exports the `nnUNet_raw`, `nnUNet_preprocessed` and `nnUNet_results` paths, which you only need to run `nnUNetv2_*` commands by hand (`src.main` sets them itself).

Install dependencies (`uv sync` also installs the `dev` and `notebooks` groups):

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
| `make download` | Download the COCO segmentation dataset from Roboflow into `coco_semantic_segmentation_dataset/`. Skipped if the folder already has a split with COCO annotations; delete the folder to download again. |
| `make convert` | Rasterize COCO polygons in class order (`root` → `stele` → `xylem`) and export the dataset to `data/nnUNet_raw/` with specimen-grouped folds (`splits_final.json`). |
| `make test` | Run `pytest` unit and dataset checks (class nesting, exclusive masks, zero specimen leakage and no near-duplicate images across folds/test set). |
| `make integrity` | Verify the dataset, extract its fingerprint, generate 2D plans and preprocess into `data/nnUNet_preprocessed/` (`monai.train.enabled=false`). |

To train locally:

```bash
uv run python -m src.main monai.preprocess.enabled=false
```

By default this trains **fold 0** with `nnUNetTrainer_20epochs` (a short run; see the overrides below for longer trainings). The device is detected automatically (`cuda` → `mps` → `cpu`); force one with `monai.train.device=mps`.

## Cloud / VM Workflow (Docker + GPU)

Designed for GPU instances (e.g., GCP VM with NVIDIA L4). The VM needs Docker, the NVIDIA Container Toolkit, `make` and a clone of this repository with its own `.env` (see [Setup](#setup)). The Roboflow export (`coco_semantic_segmentation_dataset/`), nnU-Net datasets and checkpoints (`data/`) and run logs (`logs/`) are mounted from the host.

1. **Build the Docker image:**
   ```bash
   make docker-build
   ```

2. **Download and convert the dataset:**
   ```bash
   make download-vm
   make convert-vm
   ```

3. **Run fingerprint extraction, planning, and preprocessing:**
   ```bash
   make preprocess-vm
   ```

4. **Start GPU training** (always on `cuda`, with all CPU cores but one for data augmentation):
   ```bash
   make train-vm
   ```

Rebuild the image (`make docker-build`) after pulling changes to `src/`, `cfg/` or the dependencies.

### Passing Hydra Overrides (`ARGS`)

All `*-vm` targets accept Hydra configuration overrides via `ARGS`:

```bash
# Re-convert overwriting an existing export, then preprocess again
# (the dataset fingerprint and plans change)
make convert-vm ARGS="export.overwrite=true"
make preprocess-vm

# Train fold 1 for 250 epochs
make train-vm ARGS="monai.train.fold=1 monai.runner.trainer_class_name=nnUNetTrainer_250epochs"

# Resume training from the latest checkpoint
make train-vm ARGS="monai.train.continue_training=true"
```

## Outputs

| Path | Content |
| :--- | :--- |
| `data/nnUNet_raw/` | nnU-Net dataset, `case_mapping.csv` (case id ↔ original image, fold and specimen group) and `splits_final.json` |
| `data/nnUNet_preprocessed/` | Fingerprint, plans and preprocessed data |
| `data/nnUNet_results/` | Checkpoints and validation results per fold |
| `logs/runs/<date>/<time>/` | Hydra run outputs (`.hydra/` config and overrides, and the run log) |

## Development

```bash
uv run ruff check src tests   # lint (notebooks/ is excluded)
uv run pytest                 # tests; real-dataset checks are skipped if the dataset is missing
```

Exploratory data analysis (specimen groups, split leakage, image sizes, annotations and pixel distribution per class) is in `notebooks/exploratory_data_analysis.ipynb`.

## Project Structure

```text
├── cfg/                  # Hydra configuration: paths, export, monai, roboflow, hydra (run dir)
│                         # and logger (wandb, not wired into training yet)
├── src/
│   ├── main.py           # Planning, preprocessing, and training entry point
│   └── utils/
│       ├── download.py   # Roboflow dataset downloader
│       ├── convert.py    # COCO-to-nnU-Net converter and grouped fold splitter
│       ├── coco.py       # COCO loading, class mapping, polygon rasterization and specimen ids
│       └── resolvers.py  # Custom OmegaConf resolvers (device and splits)
├── tests/                # Synthetic and real-dataset validation tests
├── notebooks/            # Exploratory data analysis
├── Dockerfile            # Container image for cloud/VM execution
└── Makefile              # Local and Docker pipeline targets
```
