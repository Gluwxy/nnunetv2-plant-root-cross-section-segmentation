"""
Checks on the real Roboflow export (coco_semantic_segmentation_dataset/) and on the nnU-Net dataset
built from it. Skipped when the data is not available (download it with src.utils.download).
"""
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
import rootutils
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from PIL import Image

import src.utils.resolvers  # noqa: F401  (registers the OmegaConf resolvers used in cfg/)
from src.utils.coco import COCO_FILENAME, load_coco, normalize_name
from src.utils.convert import convert_to_nnunet_format
from tests.checks import (
    CLASSES,
    assert_exclusive_nested_classes,
    assert_no_group_leakage,
    class_polygons_mask,
)

ROOT = Path(__file__).resolve().parents[1]
rootutils.setup_root(ROOT, indicator="pyproject.toml", pythonpath=True)  # sets PROJECT_ROOT for cfg/paths

DATASET = ROOT / "coco_semantic_segmentation_dataset"
NNUNET_RAW = ROOT / "data" / "nnUNet_raw" / "Dataset001_RootCrossSection"
NNUNET_PREPROCESSED = ROOT / "data" / "nnUNet_preprocessed" / "Dataset001_RootCrossSection"

with initialize_config_dir(config_dir=str(ROOT / "cfg"), version_base="1.3"):
    EXPORT_CFG = OmegaConf.to_container(compose(config_name="config").export, resolve=True)

SPLITS = [s for s in EXPORT_CFG["roboflow_splits"] if (DATASET / s / COCO_FILENAME).exists()]

# Max fraction of a nested class drawn outside its parent (small annotation imprecision on borders)
NESTING_TOLERANCE = 0.02
# Max Hamming distance between 16x16 average hashes to consider two images (near) duplicates
DUPLICATE_HASH_DISTANCE = 5

pytestmark = pytest.mark.skipif(not SPLITS, reason=f"Roboflow export not found in {DATASET}")


def category_ids(coco: dict) -> dict[str, int]:
    names = {normalize_name(c["name"]): c["id"] for c in coco["categories"]}
    return {n: i for n, i in names.items() if n in CLASSES}


def iter_images():
    """(split, image_info, annotations, {class_name: category_id}) for every image of the export."""
    for split in SPLITS:
        coco = load_coco(DATASET / split)
        anns = defaultdict(list)
        for ann in coco["annotations"]:
            anns[ann["image_id"]].append(ann)
        for img in coco["images"]:
            yield split, img, anns[img["id"]], category_ids(coco)


def read_case_mapping(raw_dir: Path) -> list[dict]:
    with open(raw_dir / "case_mapping.csv", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def average_hash(path: Path) -> np.ndarray:
    gray = np.asarray(Image.open(path).convert("L").resize((16, 16), Image.BILINEAR), dtype=float)
    return (gray > gray.mean()).ravel()


@pytest.fixture(scope="module")
def exported(tmp_path_factory) -> tuple[Path, list[dict], list[dict]]:
    """Fresh nnU-Net export of the real dataset with the project config (cfg/export/default.yaml)."""
    out = tmp_path_factory.mktemp("nnunet") / "raw"
    convert_to_nnunet_format(DATASET, out, SPLITS, "Case", EXPORT_CFG["split"],
                             draw_order=EXPORT_CFG["draw_order"])
    return out, read_case_mapping(out), json.loads((out / "splits_final.json").read_text())


# --------------------------------------------------------------------------- #
# COCO annotations
# --------------------------------------------------------------------------- #
def test_categories_are_root_stele_xylem():
    for split in SPLITS:
        assert set(category_ids(load_coco(DATASET / split))) == set(CLASSES), split


def test_every_image_has_all_classes():
    missing = [f"{split}/{img['file_name']}: {sorted(set(ids) - {n for n, i in ids.items() if i in present})}"
               for split, img, anns, ids in iter_images()
               if not set(ids.values()) <= (present := {a["category_id"] for a in anns})]
    assert not missing, "Images without some class:\n" + "\n".join(missing)


def test_classes_are_nested():
    """stele lies inside root and xylem inside stele (the export relies on it to paint them in order)."""
    bad = []
    for split, img, anns, ids in iter_images():
        masks = {n: class_polygons_mask(anns, ids[n], img["height"], img["width"]) for n in CLASSES}
        for inner, outer in (("stele", "root"), ("xylem", "stele")):
            outside = (masks[inner] & ~masks[outer]).sum() / max(masks[inner].sum(), 1)
            if outside > NESTING_TOLERANCE:
                bad.append(f"{split}/{img['file_name']}: {outside:.1%} of {inner} outside {outer}")
    assert not bad, "\n".join(bad)


# --------------------------------------------------------------------------- #
# nnU-Net export
# --------------------------------------------------------------------------- #
def test_export_label_maps_have_exclusive_classes(exported):
    out, rows, _ = exported
    images = {(split, img["file_name"]): (anns, ids) for split, img, anns, ids in iter_images()}
    for r in rows:
        anns, ids = images[(r["roboflow_split"], r["original_name"])]
        sub = "labelsTr" if r["nnunet_split"] == "train" else "labelsTs"
        label_map = np.array(Image.open(out / sub / f"{r['case_identifier']}.png"))
        assert_exclusive_nested_classes(label_map, anns, ids)


def test_export_has_no_group_leakage(exported):
    _, rows, splits_final = exported
    assert len(splits_final) == EXPORT_CFG["split"]["n_folds"]
    assert_no_group_leakage(rows, splits_final)


def test_no_near_duplicate_images_across_partitions(exported):
    """
    The specimen prefix is only a guess of what makes images related: also check by content that no
    (near) identical image ends up in two different partitions (test set or different folds).
    """
    _, rows, _ = exported
    part = {r["case_identifier"]: r["fold"] if r["nnunet_split"] == "train" else "test" for r in rows}
    hashes = {r["case_identifier"]: average_hash(DATASET / r["roboflow_split"] / r["original_name"])
              for r in rows}
    names = {r["case_identifier"]: r["original_name"] for r in rows}
    dups = [f"{names[a]} ({part[a]}) ~ {names[b]} ({part[b]})"
            for a, b in itertools.combinations(hashes, 2)
            if part[a] != part[b] and (hashes[a] != hashes[b]).sum() <= DUPLICATE_HASH_DISTANCE]
    assert not dups, "Near-duplicate images in different partitions:\n" + "\n".join(dups)


@pytest.mark.skipif(not (NNUNET_RAW / "case_mapping.csv").exists(), reason="No nnU-Net export in data/")
def test_existing_export_has_no_group_leakage():
    """The dataset nnU-Net actually trains on (data/nnUNet_raw + nnUNet_preprocessed/splits_final.json)."""
    rows = read_case_mapping(NNUNET_RAW)
    splits_path = NNUNET_PREPROCESSED / "splits_final.json"
    if not splits_path.exists():
        splits_path = NNUNET_RAW / "splits_final.json"
    assert_no_group_leakage(rows, json.loads(splits_path.read_text()))
