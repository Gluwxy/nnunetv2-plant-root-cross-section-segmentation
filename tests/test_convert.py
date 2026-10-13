"""Unit tests for src/utils/convert.py on small synthetic Roboflow COCO exports."""
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.utils.coco import extract_group, get_mapped_classes, rasterize_annotations, resolve_draw_order
from src.utils.convert import convert_to_nnunet_format, make_grouped_partition
from tests.checks import CLASSES, assert_exclusive_nested_classes, assert_no_group_leakage

GROUP_PATTERN = r"^([^-]+)-"
SUBGROUP_PATTERN = r"^[^-]+-(\d)"
CATEGORIES = [
    {"id": 0, "name": "root-cross-section", "supercategory": "none"},  # Roboflow project supercategory
    {"id": 1, "name": "root", "supercategory": "root-cross-section"},
    {"id": 2, "name": "stele", "supercategory": "root-cross-section"},
    {"id": 3, "name": "xylem", "supercategory": "root-cross-section"},
]
SIZE = 40


def square(x0: int, y0: int, x1: int, y1: int) -> list[float]:
    return [x0, y0, x1, y0, x1, y1, x0, y1]


def nested_annotations(image_id: int, start_id: int = 0) -> list[dict]:
    """Root disc ⊇ stele ⊇ two xylem vessels, as in the real annotations."""
    polys = [(1, square(2, 2, 37, 37)), (2, square(10, 10, 30, 30)),
             (3, square(13, 13, 18, 18)), (3, square(22, 22, 27, 27))]
    return [{"id": start_id + i, "image_id": image_id, "category_id": c, "segmentation": [p],
             "area": 0, "bbox": [], "iscrowd": 0} for i, (c, p) in enumerate(polys)]


@pytest.fixture
def roboflow_export(tmp_path: Path) -> Path:
    """
    8 specimen groups x 4 images, with each group scattered over Roboflow's train/valid/test
    folders (like '1501' in the real export), so Roboflow's own split leaks specimens.
    """
    export = tmp_path / "export"
    splits = ["train", "valid", "test"]
    coco = {s: {"images": [], "annotations": [], "categories": CATEGORIES} for s in splits}
    rng = random.Random(0)
    image_id = 0
    for g in range(8):
        for sub in ("1a", "1b", "2a", "2b"):
            split = rng.choice(splits)
            name = f"15{g:02d}-{sub}_jpg.rf.{image_id:032x}.jpg"
            (export / split).mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (SIZE, SIZE), (120, 80, 40)).save(export / split / name)
            coco[split]["images"].append({"id": image_id, "file_name": name, "height": SIZE, "width": SIZE})
            coco[split]["annotations"] += nested_annotations(image_id, start_id=4 * image_id)
            image_id += 1
    for split in splits:
        with open(export / split / "_annotations.coco.json", "w") as fh:
            json.dump(coco[split], fh)
    return export


def run_convert(export: Path, out: Path, **split_overrides) -> tuple[list[dict], list[dict]]:
    split_cfg = {"strategy": "grouped", "group_pattern": GROUP_PATTERN, "subgroup_pattern": SUBGROUP_PATTERN,
                 "test_fraction": 0.2, "n_folds": 3, "seed": 42, **split_overrides}
    convert_to_nnunet_format(export, out, ["train", "valid", "test"], "Root", split_cfg,
                             preprocessed_dir=out.parent / "preprocessed")
    with open(out / "case_mapping.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return rows, json.loads((out / "splits_final.json").read_text())


# --------------------------------------------------------------------------- #
# Classes and rasterization
# --------------------------------------------------------------------------- #
def test_mapped_classes_skip_roboflow_supercategory():
    assert get_mapped_classes({"categories": CATEGORIES}) == CLASSES


def test_default_draw_order_keeps_nested_classes_exclusive():
    anns = nested_annotations(0)
    order = resolve_draw_order({"background": 0, **CLASSES}, None)
    mask = rasterize_annotations(anns, SIZE, SIZE, {1: 1, 2: 2, 3: 3}, order)
    assert_exclusive_nested_classes(mask, anns, {"root": 1, "stele": 2, "xylem": 3})


def test_wrong_draw_order_hides_inner_classes():
    """Painting the outer polygon last covers the nested ones: the reason for the fixed order."""
    mask = rasterize_annotations(nested_annotations(0), SIZE, SIZE, {1: 1, 2: 2, 3: 3}, [3, 2, 1])
    assert set(np.unique(mask)) == {0, CLASSES["root"]}


def test_draw_order_must_list_every_class():
    with pytest.raises(ValueError):
        resolve_draw_order({"background": 0, **CLASSES}, ["root", "stele"])


# --------------------------------------------------------------------------- #
# Grouped partition (no specimen leakage)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name, group", [
    ("1501-11_jpg_jpg.rf.6534aa8210ddf4a2e0cc9019832b2345.jpg", "1501"),
    ("1504-2d_png_png_jpg.rf.136c463a8e7e1fee1da41e9f48cf2a3d.jpg", "1504"),
])
def test_extract_group(name, group):
    assert extract_group(name, GROUP_PATTERN) == group


@pytest.mark.parametrize("seed", range(10))
def test_grouped_partition_keeps_groups_together(seed):
    rng = random.Random(seed)
    sizes = [rng.randint(1, 8) for _ in range(20)]
    group_of = {f"{g}-{i}": str(g) for g, n in enumerate(sizes) for i in range(n)}

    test_keys, fold_of = make_grouped_partition(group_of, None, 0.15, 5, seed)

    assert test_keys.isdisjoint(fold_of) and test_keys | set(fold_of) == set(group_of)
    assert {group_of[k] for k in test_keys}.isdisjoint(group_of[k] for k in fold_of)
    folds_per_group = Counter()
    for group in {group_of[k] for k in fold_of}:
        folds_per_group[group] = len({f for k, f in fold_of.items() if group_of[k] == group})
    assert set(folds_per_group.values()) == {1}
    assert set(fold_of.values()) == set(range(5)), "Every fold must get at least one group"


def test_grouped_partition_needs_enough_groups():
    with pytest.raises(ValueError):
        make_grouped_partition({"a-1": "a", "b-1": "b", "c-1": "c"}, None, 0.15, 5, 0)


# --------------------------------------------------------------------------- #
# End-to-end export
# --------------------------------------------------------------------------- #
def test_export_has_no_group_leakage(roboflow_export, tmp_path):
    rows, splits_final = run_convert(roboflow_export, tmp_path / "raw")

    assert len(splits_final) == 3
    assert_no_group_leakage(rows, splits_final)
    # nnU-Net reads the copy in nnUNet_preprocessed
    assert json.loads((tmp_path / "preprocessed" / "splits_final.json").read_text()) == splits_final


def test_export_label_maps_have_exclusive_classes(roboflow_export, tmp_path):
    out = tmp_path / "raw"
    rows, _ = run_convert(roboflow_export, out)

    anns = nested_annotations(0)
    for r in rows:
        sub = "labelsTr" if r["nnunet_split"] == "train" else "labelsTs"
        label_map = np.array(Image.open(out / sub / f"{r['case_identifier']}.png"))
        assert_exclusive_nested_classes(label_map, anns, {"root": 1, "stele": 2, "xylem": 3})


def test_leakage_check_detects_split_group():
    """Sanity check of the checker itself: a group on both sides of a fold must fail."""
    rows = [{"case_identifier": c, "nnunet_split": "train", "fold": f, "group": g}
            for c, f, g in [("a", 0, "1501"), ("b", 1, "1501"), ("c", 1, "1502")]]
    splits = [{"train": ["b", "c"], "val": ["a"]}, {"train": ["a"], "val": ["b", "c"]}]
    with pytest.raises(AssertionError, match="several folds"):
        assert_no_group_leakage(rows, splits)
