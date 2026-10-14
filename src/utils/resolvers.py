from pathlib import Path

from omegaconf import ListConfig, OmegaConf

from src.utils.coco import COCO_FILENAME


def get_splits(dataset_path: str) -> ListConfig:
    """Roboflow split folders (those with a COCO annotation file), sorted so case ids are reproducible."""
    root = Path(dataset_path)
    if not root.is_dir():
        return ListConfig([])
    # ListConfig, not list: OmegaConf can only store a resolver's list result as a config node
    return ListConfig(sorted(p.name for p in root.iterdir() if (p / COCO_FILENAME).is_file()))


def get_folds(model_dir: str) -> ListConfig:
    """Trained fold folders (those with validation PNG predictions) inside an nnU-Net model directory."""
    root = Path(model_dir)
    if not root.is_dir():
        return ListConfig([])
    return ListConfig(
        sorted(p.name for p in root.glob("fold_*") if any((p / "validation").glob("*.png")))
    )


def get_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


if not OmegaConf.has_resolver("get_splits"):
    OmegaConf.register_new_resolver("get_splits", get_splits)

if not OmegaConf.has_resolver("get_folds"):
    OmegaConf.register_new_resolver("get_folds", get_folds)

if not OmegaConf.has_resolver("get_device"):
    OmegaConf.register_new_resolver("get_device", get_device)
