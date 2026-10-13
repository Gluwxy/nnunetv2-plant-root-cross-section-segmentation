"""
Converts a Roboflow "COCO Segmentation" export into the nnU-Net v2 raw format.

COCO polygons are rasterized here in a fixed class order (by default increasing label value:
root -> stele -> xylem), so nested structures are never hidden by an outer polygon, which
happens with Roboflow's "png-mask-semantic" export.

Output (paths.nnunet_raw):
    imagesTr/<prefix>_000_0000.png   RGB image
    labelsTr/<prefix>_000.png        uint8 label map (0 = background)
    imagesTs/, labelsTs/             held-out test set
    dataset.json, case_mapping.csv, splits_final.json (also copied to paths.nnunet_preprocessed)

Split strategies (export.split.strategy):
    grouped   pool all Roboflow images; each specimen group (file name prefix, e.g. '1501') is kept
              entirely in the test set or in a single cross-validation fold (avoids leakage
              between serial sections of the same root)
    roboflow  keep Roboflow's own train/valid/test assignment

Usage:
    uv run python -m src.utils.convert
    uv run python -m src.utils.convert export.overwrite=true
"""
import csv
import json
import logging
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import rootutils
import hydra
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".git", pythonpath=True)
log = logging.getLogger(__name__)

NNUNET_SUBDIRS = ("imagesTr", "labelsTr", "imagesTs", "labelsTs")
COCO_FILENAME = "_annotations.coco.json"

# --------------------------------------------------------------------------- #
# Utils & Parsing
# --------------------------------------------------------------------------- #
def load_coco(split_dir: Path) -> dict:
    """Loads the Roboflow COCO annotation file of a split folder."""
    coco_path = split_dir / COCO_FILENAME
    if not coco_path.exists():
        raise FileNotFoundError(f"No {COCO_FILENAME} found in {split_dir}.")
    with open(coco_path, encoding="utf-8") as fh:
        return json.load(fh)

def normalize_name(name: str) -> str:
    """'Vascular bundle' -> 'vascular_bundle' (lowercase, hyphens and spaces to single underscores)."""
    return "_".join(name.strip().lower().replace("-", " ").split())

def get_mapped_classes(coco: dict) -> dict[str, int]:
    """Returns {normalized_class_name: pixel_value} (1..N in category-id order), skipping the empty
    supercategory Roboflow adds with the project/group name."""
    supercats = {cat.get("supercategory") for cat in coco["categories"]}
    valid_cats = [c for c in sorted(coco["categories"], key=lambda x: x["id"])
                  if not (c["name"] in supercats and c.get("supercategory") in (None, "none"))]
    return {normalize_name(cat["name"]): idx for idx, cat in enumerate(valid_cats, start=1)}

def rasterize_annotations(annotations: list[dict], height: int, width: int,
                          cat_to_val: dict[int, int], value_order: list[int]) -> np.ndarray:
    """Paints the COCO polygons into a (height, width) uint8 mask, class by class in value_order
    (later classes end up on top)."""
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)

    for value in value_order:
        # Annotations of the class painted with this pixel value
        for ann in [a for a in annotations if cat_to_val.get(a["category_id"]) == value]:
            if isinstance(ann.get("segmentation"), dict):
                raise NotImplementedError(f"RLE segmentation not supported: {ann.get('id')}")

            for poly in ann.get("segmentation", []):
                if len(poly) >= 6: # at least 3 (x, y) points
                    draw.polygon(list(zip(poly[0::2], poly[1::2])), fill=value)

    return np.array(canvas, dtype=np.uint8)

def extract_group(file_name: str, pattern: str) -> str:
    """First capture group of `pattern` in the file name (e.g. '1501' in '1501-11_jpg_jpg.rf...jpg')."""
    match = re.search(pattern, file_name)
    if match is None:
        raise ValueError(f"Pattern {pattern!r} does not match {file_name!r}")
    return match.group(1)

def resolve_draw_order(labels: dict[str, int], draw_order: list[str] | None) -> list[int]:
    """Pixel values in painting order (later = on top). Default: increasing label value."""
    foreground = {n: v for n, v in labels.items() if v != 0}
    if not draw_order:
        return sorted(foreground.values())
    unknown, missing = set(draw_order) - set(foreground), set(foreground) - set(draw_order)
    if unknown or missing:
        raise ValueError(f"draw_order must list every class once. Unknown: {sorted(unknown)}, "
                         f"missing: {sorted(missing)}, available: {list(foreground)}")
    return [foreground[n] for n in draw_order]

def log_partition_summary(case_mapping: list[dict], n_folds: int) -> None:
    """Logs images, groups and subgroups of the test set and of every fold."""
    log.info("Partition (grouped by specimen):")
    parts = [("test", lambda r: r["nnunet_split"] == "test")] + \
            [(f"fold {f}", lambda r, f=f: r["fold"] == f) for f in range(n_folds)]
    for name, belongs in parts:
        rows = [r for r in case_mapping if belongs(r)]
        subs = Counter(r["subgroup"] for r in rows if r["subgroup"])
        sub_txt = f" | subgroups {dict(sorted(subs.items()))}" if subs else ""
        log.info(f"  {name:<7} {len(rows):>3} images | groups {sorted({r['group'] for r in rows})}{sub_txt}")

# --------------------------------------------------------------------------- #
# Partitioning
# --------------------------------------------------------------------------- #
def make_grouped_partition(group_of: dict, subgroup_of: dict | None, test_frac: float,
                           n_folds: int, seed: int, n_trials: int = 2000) -> tuple[set[str], dict[str, int]]:
    """
    Builds a test set and n_folds cross-validation folds where every group stays together.

    Test set: among n_trials random group orders, the groups whose total size is closest to
    test_frac of the images and, if subgroups are given, whose subgroup proportions are closest
    to the whole dataset. Folds: remaining groups, largest first, each assigned to the smallest fold.

    Returns (test_keys, fold_of_key).
    """
    rng = random.Random(seed)
    members = defaultdict(list)
    for key, group in sorted(group_of.items()):
        members[group].append(key)
    groups = sorted(members)

    target = test_frac * len(group_of)
    if len(groups) < n_folds + 1:
        raise ValueError(f"Need at least {n_folds + 1} groups, got {len(groups)}")

    def calc_fractions(keys):
        c = Counter(subgroup_of[k] for k in keys)
        return {sg: n / sum(c.values()) for sg, n in c.items()}

    overall = calc_fractions(group_of.keys()) if subgroup_of else None
    best_test, best_score = None, float("inf")

    # Search for the best test set
    for _ in range(n_trials):
        order = groups[:]
        rng.shuffle(order)
        chosen, size = [], 0
        for g in order:
            if size >= target: break
            chosen.append(g)
            size += len(members[g])

        if len(groups) - len(chosen) < n_folds: continue

        score = abs(size - target) / len(group_of)
        if overall:
            fracs = calc_fractions([k for g in chosen for k in members[g]])
            score += sum(abs(fracs.get(sg, 0) - overall[sg]) for sg in overall)

        if score < best_score:
            best_test, best_score = set(chosen), score

    if best_test is None:
        raise ValueError("Could not build a test set leaving enough groups for the folds; lower test_fraction.")
    test_keys = {k for g in best_test for k in members[g]}

    # Greedy fold assignment
    remaining = [g for g in groups if g not in best_test]
    rng.shuffle(remaining)  # random tie-break between groups of equal size
    remaining.sort(key=lambda g: len(members[g]), reverse=True)
    fold_sizes, fold_of = [0] * n_folds, {}
    for group in remaining:
        fold = min(range(n_folds), key=lambda f: (fold_sizes[f], f))
        fold_sizes[fold] += len(members[group])
        fold_of.update({k: fold for k in members[group]})

    return test_keys, fold_of

# --------------------------------------------------------------------------- #
# Main Conversion
# --------------------------------------------------------------------------- #
def convert_to_nnunet_format(export_dir: Path, out_dir: Path, roboflow_splits: list[str],
                             prefix: str, split_cfg: dict, draw_order: list[str] | None = None,
                             overwrite: bool = False, preprocessed_dir: Path | None = None) -> None:
    """Converts the Roboflow COCO export in export_dir into the nnU-Net raw dataset out_dir."""
    strategy = split_cfg.get("strategy", "grouped")
    if strategy not in ("grouped", "roboflow"):
        raise ValueError(f"Unknown split strategy {strategy!r} (use 'grouped' or 'roboflow')")
    log.info(f"Converting {export_dir} -> {out_dir} (split strategy: {strategy})")

    # Prepare output folders
    if any((out_dir / sub).is_dir() and any((out_dir / sub).iterdir()) for sub in NNUNET_SUBDIRS):
        if not overwrite:
            raise FileExistsError(f"{out_dir} already contains cases. Re-run with export.overwrite=true.")
        log.warning(f"Overwriting existing dataset in {out_dir}")
        shutil.rmtree(out_dir)
    for sub in NNUNET_SUBDIRS:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    available = [s for s in roboflow_splits if (export_dir / s).is_dir()]
    for missing in sorted(set(roboflow_splits) - set(available)):
        log.warning(f"Roboflow split folder not found, skipping: {export_dir / missing}")
    if not available:
        raise FileNotFoundError(f"None of the splits {roboflow_splits} exist in {export_dir}")
    keys = [f"{s}/{img['file_name']}" for s in available for img in load_coco(export_dir / s)["images"]]

    group_of = {k: extract_group(Path(k).name, split_cfg["group_pattern"]) for k in keys}
    subp = split_cfg.get("subgroup_pattern")
    subgroup_of = {k: extract_group(Path(k).name, subp) for k in keys} if subp else None

    # Decide test membership and cross-validation folds
    if strategy == "grouped":
        n_folds = int(split_cfg.get("n_folds", 5))
        test_keys, fold_of_key = make_grouped_partition(group_of, subgroup_of,
                                                        float(split_cfg.get("test_fraction", 0.15)),
                                                        n_folds, int(split_cfg.get("seed", 42)))
    else:
        test_splits = set(split_cfg.get("test", ["test"]))
        test_keys = {k for k in keys if k.split("/", 1)[0] in test_splits}
        fold_of_key, n_folds = {}, 0

    global_labels, case_mapping, missing_classes = {"background": 0}, [], []
    case_idx, train_count = 0, 0

    for split in available:
        split_dir = export_dir / split
        coco = load_coco(split_dir)
        classes = get_mapped_classes(coco)
        log.info(f"[{split}] {len(coco['images'])} images, {len(coco['annotations'])} annotations, classes {classes}")

        # Classes must be identical in every split
        if len(global_labels) > 1 and classes != {k: v for k, v in global_labels.items() if k != "background"}:
            raise ValueError(f"Classes in {split} {classes} do not match previous splits {global_labels}")
        global_labels.update(classes)

        # Painting order and COCO category id -> pixel value
        val_order = resolve_draw_order(global_labels, draw_order)
        cat_to_val = {c["id"]: classes[normalize_name(c["name"])] for c in coco["categories"] if normalize_name(c["name"]) in classes}
        anns_by_img = defaultdict(list)
        for ann in coco["annotations"]: anns_by_img[ann["image_id"]].append(ann)

        for img_info in sorted(coco["images"], key=lambda i: i["file_name"]):
            img_path = split_dir / img_info["file_name"]
            key = f"{split}/{img_info['file_name']}"
            is_train = key not in test_keys

            anns = anns_by_img.get(img_info["id"], [])
            if is_train and not anns:
                raise ValueError(f"Training image without annotations: {img_path}")

            case_id = f"{prefix}_{case_idx:03d}"
            target_img_dir = out_dir / ("imagesTr" if is_train else "imagesTs")
            image = Image.open(img_path).convert("RGB")
            image.save(target_img_dir / f"{case_id}_0000.png")

            if anns:
                mask = rasterize_annotations(anns, img_info["height"], img_info["width"], cat_to_val, val_order)
                if mask.shape != (image.height, image.width):
                    raise ValueError(f"Size mismatch: image (h, w)={(image.height, image.width)} "
                                     f"vs COCO {mask.shape} for {img_path}")
                target_lbl_dir = out_dir / ("labelsTr" if is_train else "labelsTs")
                Image.fromarray(mask).save(target_lbl_dir / f"{case_id}.png")

                # Report images where a class has no annotations (usually an annotation mistake)
                present_vals = {cat_to_val.get(a["category_id"]) for a in anns}
                absent = [k for k, v in global_labels.items() if v in val_order and v not in present_vals]
                if absent: missing_classes.append(f"{key}: no {', '.join(absent)}")

            train_count += int(is_train)
            case_mapping.append({
                "case_identifier": case_id, "nnunet_split": "train" if is_train else "test",
                "fold": fold_of_key.get(key, ""), "group": group_of[key],
                "subgroup": subgroup_of[key] if subgroup_of else "",
                "roboflow_split": split, "original_name": img_path.name
            })
            case_idx += 1

    if missing_classes:
        log.warning(f"{len(missing_classes)} images are missing at least one class:")
        for line in missing_classes:
            log.warning(f"  {line}")

    # Case mapping (case id <-> original Roboflow file, split, fold, group) and dataset.json
    with open(out_dir / "case_mapping.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(case_mapping[0]))
        writer.writeheader(); writer.writerows(case_mapping)

    with open(out_dir / "dataset.json", "w", encoding="utf-8") as f:
        json.dump({"channel_names": {"0": "R", "1": "G", "2": "B"}, "labels": global_labels,
                   "numTraining": train_count, "file_ending": ".png"}, f, indent=2)

    # Cross-validation folds for nnU-Net (grouped strategy only)
    if strategy == "grouped":
        splits_final = [{"train": [r["case_identifier"] for r in case_mapping if r["fold"] != f and r["nnunet_split"] == "train"],
                         "val": [r["case_identifier"] for r in case_mapping if r["fold"] == f]} for f in range(n_folds)]

        for target_dir in [out_dir, preprocessed_dir] if preprocessed_dir else [out_dir]:
            target_dir.mkdir(parents=True, exist_ok=True)
            with open(target_dir / "splits_final.json", "w", encoding="utf-8") as f:
                json.dump(splits_final, f, indent=2)

        log_partition_summary(case_mapping, n_folds)
        log.info(f"splits_final.json written to {out_dir}" + (f" and {preprocessed_dir}" if preprocessed_dir else ""))
    elif preprocessed_dir is not None and (preprocessed_dir / "splits_final.json").exists():
        log.warning(f"{preprocessed_dir / 'splits_final.json'} exists and will be used by nnU-Net; "
                    f"delete it to let nnU-Net create random folds.")

    log.info(f"Done: {train_count} training cases, {case_idx - train_count} test cases, "
             f"labels {global_labels} -> {out_dir}")

@hydra.main(version_base="1.3", config_path="../../cfg", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entry point: reads cfg.paths and cfg.export and runs the conversion."""
    out_dir = Path(cfg.paths.nnunet_raw)
    convert_to_nnunet_format(
        export_dir=Path(cfg.paths.dataset), out_dir=out_dir,
        roboflow_splits=OmegaConf.to_container(cfg.export.roboflow_splits, resolve=True),
        prefix=cfg.export.get("prefix") or out_dir.name.split("_", 1)[-1],
        split_cfg=OmegaConf.to_container(cfg.export.split, resolve=True),
        draw_order=OmegaConf.to_container(cfg.export.get("draw_order"), resolve=True) if cfg.export.get("draw_order") else None,
        overwrite=bool(cfg.export.get("overwrite", False)),
        preprocessed_dir=Path(cfg.paths.nnunet_preprocessed) if cfg.paths.get("nnunet_preprocessed") else None
    )

if __name__ == "__main__":
    main()