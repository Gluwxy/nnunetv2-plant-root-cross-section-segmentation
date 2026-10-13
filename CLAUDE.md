# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Semantic segmentation of plant root cross-sections (classes: root, stele, xylem) with nnU-Net v2, driven through MONAI's `nnUNetV2Runner` and configured with Hydra (`cfg/`).

## Pipeline

Use `uv` and run entry points as modules from the repo root, in this order:

1. `uv run python -m src.utils.download`: downloads the Roboflow COCO-segmentation export (needs `ROBOFLOW_API_KEY` in `.env`)
2. `uv run python -m src.utils.convert`: converts COCO to nnU-Net raw format in `data/nnUNet_raw` (`export.overwrite=true` to replace)
3. `uv run python -m src.main`: plans, preprocesses and trains (`monai.preprocess.enabled=false` to train only)

Override config with Hydra CLI syntax (e.g. `monai.train.fold=1`, `monai.runner.trainer_class_name=nnUNetTrainer_250epochs`). Run outputs go to `logs/runs/<date>/<time>/`.

Lint: `uv run ruff check src` (config in `pyproject.toml`). Tests: `uv run pytest` (`tests/test_dataset.py` needs the dataset downloaded; it is skipped otherwise).

## Gotchas

- `cfg/paths` resolves `${oc.env:PROJECT_ROOT}`. `PROJECT_ROOT` is set, and `.env` is loaded, by `rootutils.setup_root(...)` at module import, so every new entry point must call it before `@hydra.main`.
- Never use `runner.convert_dataset()`, which only supports NIfTI-like medical images. The dataset is written by `src/utils/convert.py`.
- COCO polygons are rasterized in increasing label order so nested structures aren't hidden. Don't switch to Roboflow's `png-mask-semantic` export.
- The `grouped` split keeps each specimen (file-name prefix, e.g. `1501`) in a single fold or in the test set to avoid leakage between serial sections. `splits_final.json` is written by convert and copied to `nnUNet_preprocessed`. Re-run preprocessing after re-exporting.
- Training epochs are set by the trainer class name (`nnUNetTrainer_20epochs`, `_250epochs`, …), not by a numeric parameter.
- `-device` is appended to the `nnUNetv2_train` command manually in `src/main.py`, because the runner prefixes extra args with `--` and `nnUNetv2_train` only accepts `-device`.
- `data/`, `logs/`, `coco_semantic_segmentation_dataset/` and `.env` are gitignored. Don't commit datasets, checkpoints or keys.
