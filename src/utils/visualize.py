"""
Generates 1x5 comparison figures (original image, ground truth, prediction, overlay, and pixel error map)
for the validation predictions of all trained nnU-Net v2 folds.

Usage:
    uv run python -m src.utils.visualize
    uv run python -m src.utils.visualize monai.runner.trainer_class_name=nnUNetTrainer_250epochs
    uv run python -m src.utils.visualize "monai.visualize.folds=[fold_1]"
"""
import logging
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import rootutils
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm

import src.utils.resolvers  # noqa: F401  (registers the OmegaConf resolvers used in cfg/)

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)
log = logging.getLogger(__name__)


def load_image(path: Path) -> np.ndarray:
    """Loads an image from a given path and returns it as a numpy array."""
    return np.array(Image.open(path))


def create_colormaps() -> tuple[ListedColormap, LinearSegmentedColormap]:
    """Defines and returns the colormaps for segmentation and error visualization."""
    # 0: background, 1: root, 2: stele, 3: xylem
    cmap_seg = ListedColormap(["#0d1117", "#ff7b72", "#79c0ff", "#d29922"])

    # Custom colormap for errors (solid bright red)
    cmap_err = LinearSegmentedColormap.from_list("error_red", ["red", "red"])

    return cmap_seg, cmap_err


def visualize_and_save(
    case_name: str,
    img: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    output_path: Path,
    cmap_seg: ListedColormap,
    cmap_err: LinearSegmentedColormap,
) -> None:
    """Generates a 1x5 subplot figure for a single case and saves it."""
    error_mask = pred != gt
    error_percentage = error_mask.mean() * 100
    vmax = len(cmap_seg.colors) - 1

    fig, ax = plt.subplots(1, 5, figsize=(25, 5))

    # 1. Original Image
    ax[0].imshow(img, cmap="gray" if img.ndim == 2 else None)
    ax[0].set_title(f"Original ({case_name})", fontsize=14)

    # 2. Ground Truth
    ax[1].imshow(gt, cmap=cmap_seg, vmin=0, vmax=vmax, interpolation="nearest")
    ax[1].set_title("Ground Truth", fontsize=14)

    # 3. Prediction
    ax[2].imshow(pred, cmap=cmap_seg, vmin=0, vmax=vmax, interpolation="nearest")
    ax[2].set_title("Prediction", fontsize=14)

    # 4. Overlay (Prediction over Original)
    ax[3].imshow(img, cmap="gray" if img.ndim == 2 else None)
    masked_pred = np.ma.masked_where(pred == 0, pred)  # Hide background class
    ax[3].imshow(masked_pred, cmap=cmap_seg, vmin=0, vmax=vmax, alpha=0.5, interpolation="nearest")
    ax[3].set_title("Overlay", fontsize=14)

    # 5. Error Map (Errors over Original)
    ax[4].imshow(img, cmap="gray" if img.ndim == 2 else None)
    masked_error = np.ma.masked_where(~error_mask, error_mask)  # Keep only error pixels
    ax[4].imshow(masked_error, cmap=cmap_err, alpha=0.8, interpolation="nearest")
    ax[4].set_title(
        f"Errors ({error_percentage:.2f}%)",
        fontsize=14,
        color="red" if error_percentage > 0 else "black",
    )

    for a in ax:
        a.axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


@hydra.main(version_base="1.3", config_path="../../cfg", config_name="config")
def main(cfg: DictConfig) -> None:
    raw_dir = Path(cfg.paths.nnunet_raw)
    results_dir = Path(cfg.paths.results)
    model_dir = Path(cfg.monai.visualize.model_dir)
    folds = [f if str(f).startswith("fold_") else f"fold_{f}" for f in cfg.monai.visualize.folds]

    if not folds:
        log.warning(f"No trained folds with validation predictions found in {model_dir}")
        return

    cmap_seg, cmap_err = create_colormaps()

    for fold in folds:
        val_dir = model_dir / fold / "validation"
        val_preds_paths = sorted(val_dir.glob("*.png"))
        if not val_preds_paths:
            log.warning(f"No validation predictions found in {val_dir}, skipping {fold}")
            continue

        fold_output_dir = results_dir / fold
        fold_output_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"[{fold}] Generating {len(val_preds_paths)} 1x5 figures into {fold_output_dir}...")

        for pred_path in tqdm(val_preds_paths, desc=f"Visualizing {fold}", unit="img"):
            case_name = pred_path.stem

            img_path = raw_dir / "imagesTr" / f"{case_name}_0000.png"
            gt_path = raw_dir / "labelsTr" / f"{case_name}.png"
            output_path = fold_output_dir / f"{case_name}.png"

            if not img_path.exists() or not gt_path.exists():
                tqdm.write(f"Warning: Skipping {case_name} (Missing original or GT file).")
                continue

            img = load_image(img_path)
            gt = load_image(gt_path)
            pred = load_image(pred_path)

            visualize_and_save(case_name, img, gt, pred, output_path, cmap_seg, cmap_err)


if __name__ == "__main__":
    main()
