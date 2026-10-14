# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Semantic segmentation of plant root cross-sections (classes: root, stele, xylem) with nnU-Net v2, driven through MONAI's `nnUNetV2Runner` and configured with Hydra (`cfg/`).

## Pipeline

Use `uv` and run entry points as modules from the repo root, in this order:

1. `uv run python -m src.utils.download`: downloads the Roboflow COCO-segmentation export (needs `ROBOFLOW_API_KEY` in `.env`). It is skipped when the folder already has a split with COCO annotations: delete the folder to download again
2. `uv run python -m src.utils.convert`: converts COCO to nnU-Net raw format in `data/nnUNet_raw` (`export.overwrite=true` to replace)
3. `uv run python -m src.main`: plans, preprocesses and trains (`monai.preprocess.enabled=false` to train only)

Override config with Hydra CLI syntax (e.g. `monai.train.fold=1`, `monai.runner.trainer_class_name=nnUNetTrainer_250epochs`). Run outputs go to `logs/runs/<date>/<time>/`.

The `Makefile` wraps these steps: `download`, `convert`, `test` and `integrity` (preprocess only) run locally with `uv`. On the GCP VM, `docker-build` builds the image and `download-vm`, `convert-vm`, `preprocess-vm` and `train-vm` run the same entry points in Docker, mounting `coco_semantic_segmentation_dataset/`, `data/` and `logs/`. Pass extra Hydra overrides with `ARGS="..."`. `train-vm` forces `monai.train.device=cuda` and sets `n_proc_da` to all CPU cores but one.

Lint: `uv run ruff check src tests` (config in `pyproject.toml`; `notebooks/` is excluded). Tests: `uv run pytest` (`tests/test_dataset.py` needs the dataset downloaded; it is skipped otherwise). EDA lives in `notebooks/exploratory_data_analysis.ipynb`.

## Gotchas

- `cfg/paths` resolves `${oc.env:PROJECT_ROOT}`. `PROJECT_ROOT` is set, and `.env` is loaded, by `rootutils.setup_root(..., indicator="pyproject.toml")` at module import, so every new entry point must call it before `@hydra.main`. Keep the `pyproject.toml` indicator: `.git` is not in the Docker image.
- `cfg/` uses custom OmegaConf resolvers from `src/utils/resolvers.py` (`get_splits` for `export.roboflow_splits`, `get_device` for `monai.train.device`). They are registered on import, so every entry point, test or notebook that reads the config must `import src.utils.resolvers  # noqa: F401`. Without the `noqa`, `ruff --fix` deletes the import.
- Resolvers that return lists must return a `ListConfig`, not a `list`. Otherwise `OmegaConf.to_container`/`--cfg job --resolve` fail.
- `get_splits` returns the dataset folders that have `_annotations.coco.json`, sorted (`test, train, valid`). Case ids follow that order, so changing it renumbers cases.
- To read the config outside `@hydra.main` (tests, notebooks), compose the full config (`initialize_config_dir` + `compose(config_name="config")`). `OmegaConf.load` of a single group file cannot resolve `${paths...}`.
- Never use `runner.convert_dataset()`, which only supports NIfTI-like medical images. The dataset is written by `src/utils/convert.py`; COCO parsing and rasterization helpers are in `src/utils/coco.py`.
- COCO polygons are rasterized in increasing label order so nested structures aren't hidden. Don't switch to Roboflow's `png-mask-semantic` export.
- The `grouped` split keeps each specimen (file-name prefix, e.g. `1501`) in a single fold or in the test set to avoid leakage between serial sections. `splits_final.json` is written by convert and copied to `nnUNet_preprocessed`. Re-run preprocessing after re-exporting.
- Training epochs are set by the trainer class name (`nnUNetTrainer_20epochs`, `_250epochs`, …), not by a numeric parameter.
- On `cuda`, nnU-Net uses `torch.compile` (`monai.train.compile`, passed as `nnUNet_compile` by `src/main.py`). Triton then needs a C compiler, so the Docker image installs `gcc` and `libc6-dev`; without them training fails on the first iteration with `Failed to find C compiler` (followed by misleading "background workers are no longer alive" errors).
- `-device` is appended to the `nnUNetv2_train` command manually in `src/main.py`, because the runner prefixes extra args with `--` and `nnUNetv2_train` only accepts `-device`.
- Dependency groups: `dev` (pytest, ruff) and `notebooks` (ipykernel, nbclient) are both synced by default (`[tool.uv] default-groups`). The Docker image skips them (`--no-dev --no-group notebooks`) and installs `git`, which uv needs to fetch `nnunetv2` from GitHub.
- The Docker targets pass `.env` with `docker run --env-file`, which does not strip quotes: write `ROBOFLOW_API_KEY=value`, not `ROBOFLOW_API_KEY="value"`.
- `data/`, `logs/`, `coco_semantic_segmentation_dataset/` and `.env` are gitignored. Don't commit datasets, checkpoints or keys.
