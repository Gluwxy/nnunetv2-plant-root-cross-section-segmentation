"""
Helpers to read Roboflow "COCO Segmentation" exports and rasterize their polygons.

Used by src/utils/convert.py to build the nnU-Net v2 raw dataset.
"""
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

COCO_FILENAME = "_annotations.coco.json"

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
                    draw.polygon(list(zip(poly[0::2], poly[1::2], strict=True)), fill=value)

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
