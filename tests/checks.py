"""Assertions shared by the synthetic and real-dataset tests."""
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw

CLASSES = {"root": 1, "stele": 2, "xylem": 3}


def assert_no_group_leakage(case_mapping: list[dict], splits_final: list[dict]) -> None:
    """Every specimen group lives in exactly one place: the test set or a single CV fold."""
    group_of = {r["case_identifier"]: r["group"] for r in case_mapping}
    train = {r["case_identifier"] for r in case_mapping if r["nnunet_split"] == "train"}
    test = {r["case_identifier"] for r in case_mapping if r["nnunet_split"] == "test"}

    # Test set vs training pool
    shared = {group_of[c] for c in test} & {group_of[c] for c in train}
    assert not shared, f"Groups both in the test set and in training: {sorted(shared)}"

    # Each group's training cases are assigned to a single fold
    folds_of_group = defaultdict(set)
    for r in case_mapping:
        if r["nnunet_split"] == "train":
            folds_of_group[r["group"]].add(str(r["fold"]))
    split_groups = {g: f for g, f in folds_of_group.items() if len(f) > 1}
    assert not split_groups, f"Groups spread over several folds: {split_groups}"

    # splits_final.json: val sets partition the training pool, no group on both sides of a fold
    vals = [set(fold["val"]) for fold in splits_final]
    assert set().union(*vals) == train, "Validation sets do not cover exactly the training cases"
    assert sum(map(len, vals)) == len(train), "A case is in more than one validation set"
    for i, fold in enumerate(splits_final):
        tr, val = set(fold["train"]), set(fold["val"])
        assert not (tr | val) & test, f"Fold {i} uses test cases"
        assert tr | val == train and not tr & val, f"Fold {i}: train/val do not partition the training pool"
        leaked = {group_of[c] for c in tr} & {group_of[c] for c in val}
        assert not leaked, f"Fold {i}: groups in both train and val: {sorted(leaked)}"


def class_polygons_mask(annotations: list[dict], cat_id: int, height: int, width: int) -> np.ndarray:
    """Boolean mask of the union of the polygons of one COCO category, rasterized on its own."""
    canvas = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for ann in annotations:
        if ann["category_id"] == cat_id:
            for poly in ann["segmentation"]:
                if len(poly) >= 6:
                    draw.polygon(list(zip(poly[0::2], poly[1::2], strict=True)), fill=1)
    return np.array(canvas, dtype=bool)


def assert_exclusive_nested_classes(label_map: np.ndarray, annotations: list[dict],
                                    cat_ids: dict[str, int]) -> None:
    """
    The COCO polygons are nested (root ⊇ stele ⊇ xylem), so they overlap by design. In the exported
    label map every pixel must hold one class and no class may hide another:
        xylem = xylem polygons, stele = stele - xylem, root = root - stele - xylem.
    """
    h, w = label_map.shape
    root, stele, xylem = (class_polygons_mask(annotations, cat_ids[n], h, w)
                          for n in ("root", "stele", "xylem"))

    assert set(np.unique(label_map)) <= {0, *CLASSES.values()}, f"Unexpected labels {np.unique(label_map)}"
    for name, value in CLASSES.items():
        assert (label_map == value).any(), f"Class {name} is not visible in the label map"

    np.testing.assert_array_equal(label_map == CLASSES["xylem"], xylem, err_msg="xylem region")
    np.testing.assert_array_equal(label_map == CLASSES["stele"], stele & ~xylem, err_msg="stele region")
    np.testing.assert_array_equal(label_map == CLASSES["root"], root & ~stele & ~xylem, err_msg="root region")
