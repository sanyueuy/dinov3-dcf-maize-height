#!/usr/bin/env python3
"""
Build an augmented hand-bbox bundle for DiffCorn training.

Augmentation policy:
- Background augmentation: with probability 0.5, replace the farthest 30% grid cells
  from image center by either a solid random color or Gaussian noise.
- Color augmentation: each of brightness / contrast / saturation / hue is applied
  independently with probability 0.5.

Only train split samples are augmented; fixed test split samples remain untouched.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from tqdm import tqdm

from feature_extraction_utils import extract_feature_from_image, load_dinov3_components, validate_feature_mode


ROOT = Path(r"D:\cornTrain\DINOV3")
DATASET_DIR = Path(r"D:\cornTrain\ZeroShotCornPhenotypingAgent\dataset")
ANNOTATIONS_FILE = DATASET_DIR / "plant_annotations.json"
DEFAULT_TEST_INDICES = ROOT / "checkpoints" / "hand_bbox_test_indices.json"
DEFAULT_OUTPUT_FILE = ROOT / "training_bundle_hand_attn_aug.pt"
DEFAULT_FEATURE_MODE = "attn_weighted"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build augmented hand-bbox feature bundle.")
    parser.add_argument("--feature-mode", default=DEFAULT_FEATURE_MODE, choices=["cls", "patch_mean", "attn_weighted"])
    parser.add_argument("--output-file", default=str(DEFAULT_OUTPUT_FILE))
    parser.add_argument("--test-indices-file", default=str(DEFAULT_TEST_INDICES))
    parser.add_argument("--margin", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--background-prob", type=float, default=0.5)
    parser.add_argument("--background-farthest-ratio", type=float, default=0.3)
    parser.add_argument("--brightness-range", type=float, default=0.3)
    parser.add_argument("--contrast-range", type=float, default=0.3)
    parser.add_argument("--saturation-range", type=float, default=0.3)
    parser.add_argument("--hue-range", type=float, default=0.1)
    return parser.parse_args()


def parse_filename_info(filename: str) -> tuple[float | None, float | None]:
    match = re.match(r"(\d+(?:\.\d+)?)", filename)
    cam_h = float(match.group(1)) if match else None
    parts = filename.split("-")
    real_h = None
    if len(parts) >= 2:
        try:
            real_h = float(parts[1])
        except ValueError:
            real_h = None
    return cam_h, real_h


def crop_bbox(image_pil: Image.Image, bbox: list[float] | tuple[float, float, float, float], margin: int = 0) -> Image.Image:
    x1, y1, x2, y2 = bbox
    return image_pil.crop(
        (
            max(0, int(x1) - margin),
            max(0, int(y1) - margin),
            min(image_pil.width, int(x2) + margin),
            min(image_pil.height, int(y2) + margin),
        )
    )


def apply_hue_shift(image: Image.Image, rng: random.Random, hue_range: float) -> Image.Image:
    hsv = np.array(image.convert("HSV"), dtype=np.uint8)
    shift = int(rng.uniform(-hue_range, hue_range) * 255)
    hsv[..., 0] = (hsv[..., 0].astype(np.int16) + shift) % 256
    return Image.fromarray(hsv, mode="HSV").convert("RGB")


def apply_color_augmentation(
    image: Image.Image,
    rng: random.Random,
    brightness_range: float,
    contrast_range: float,
    saturation_range: float,
    hue_range: float,
) -> Image.Image:
    out = image
    if rng.random() < 0.5:
        factor = 1.0 + rng.uniform(-brightness_range, brightness_range)
        out = ImageEnhance.Brightness(out).enhance(factor)
    if rng.random() < 0.5:
        factor = 1.0 + rng.uniform(-contrast_range, contrast_range)
        out = ImageEnhance.Contrast(out).enhance(factor)
    if rng.random() < 0.5:
        factor = 1.0 + rng.uniform(-saturation_range, saturation_range)
        out = ImageEnhance.Color(out).enhance(factor)
    if rng.random() < 0.5:
        out = apply_hue_shift(out, rng, hue_range)
    return out


def apply_background_augmentation(
    image: Image.Image,
    rng: random.Random,
    replace_prob: float,
    farthest_ratio: float,
) -> Image.Image:
    if rng.random() >= replace_prob:
        return image

    rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]
    if h == 0 or w == 0:
        return image

    grid_rows = 14
    grid_cols = 14
    row_edges = np.linspace(0, h, grid_rows + 1, dtype=int)
    col_edges = np.linspace(0, w, grid_cols + 1, dtype=int)

    cells = []
    center = np.array([(grid_rows - 1) / 2.0, (grid_cols - 1) / 2.0], dtype=float)
    for r in range(grid_rows):
        for c in range(grid_cols):
            dist = float(np.linalg.norm(np.array([r, c], dtype=float) - center))
            cells.append((dist, r, c))
    cells.sort(reverse=True)
    n_replace = max(1, math.ceil(len(cells) * farthest_ratio))
    selected = cells[:n_replace]

    mode = rng.choice(["solid", "noise"])
    if mode == "solid":
        fill = np.array([rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)], dtype=np.uint8)
        for _, r, c in selected:
            y1, y2 = row_edges[r], row_edges[r + 1]
            x1, x2 = col_edges[c], col_edges[c + 1]
            rgb[y1:y2, x1:x2] = fill
    else:
        noise_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))
        global_mean = rgb.mean(axis=(0, 1))
        for _, r, c in selected:
            y1, y2 = row_edges[r], row_edges[r + 1]
            x1, x2 = col_edges[c], col_edges[c + 1]
            patch_h = max(1, y2 - y1)
            patch_w = max(1, x2 - x1)
            noise = noise_rng.normal(loc=global_mean, scale=40.0, size=(patch_h, patch_w, 3))
            rgb[y1:y2, x1:x2] = np.clip(noise, 0, 255).astype(np.uint8)

    return Image.fromarray(rgb, mode="RGB")


def load_test_indices(test_indices_file: str) -> list[int]:
    data = json.loads(Path(test_indices_file).read_text(encoding="utf-8"))
    return list(data["test_indices"])


def build_bundle(args: argparse.Namespace) -> dict:
    feature_mode = validate_feature_mode(args.feature_mode)
    test_indices = set(load_test_indices(args.test_indices_file))

    with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    image_files = sorted(f for f in DATASET_DIR.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
    components = load_dinov3_components(ROOT, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    processor = components["processor"]
    model = components["model"]
    device = components["device"]

    all_data = {
        "features": [],
        "heights": [],
        "pixel_heights": [],
        "filenames": [],
        "bbox_list": [],
        "cam_heights": [],
    }

    for idx, image_path in enumerate(tqdm(image_files, desc="Building augmented bundle")):
        filename = image_path.name
        if filename not in annotations:
            raise ValueError(f"Missing bbox annotation for {filename}")

        bbox = annotations[filename]["bbox"]
        image_pil = Image.open(image_path).convert("RGB")
        roi = crop_bbox(image_pil, bbox, margin=args.margin)

        if idx not in test_indices:
            rng = random.Random(args.seed * 1000003 + idx)
            roi = apply_background_augmentation(
                roi,
                rng=rng,
                replace_prob=args.background_prob,
                farthest_ratio=args.background_farthest_ratio,
            )
            roi = apply_color_augmentation(
                roi,
                rng=rng,
                brightness_range=args.brightness_range,
                contrast_range=args.contrast_range,
                saturation_range=args.saturation_range,
                hue_range=args.hue_range,
            )

        feat = extract_feature_from_image(
            roi,
            processor=processor,
            model=model,
            device=device,
            feature_mode=feature_mode,
            resize_to=None,
        )

        x1, y1, x2, y2 = bbox
        pixel_height = y2 - y1
        cam_h, real_h = parse_filename_info(filename)
        all_data["features"].append(feat.cpu().squeeze().numpy())
        all_data["heights"].append(real_h if real_h is not None else 0.0)
        all_data["pixel_heights"].append(pixel_height)
        all_data["filenames"].append(filename)
        all_data["bbox_list"].append(bbox)
        all_data["cam_heights"].append(cam_h if cam_h is not None else 0.0)

    bundle = {
        "features": torch.tensor(np.stack(all_data["features"])),
        "heights": torch.tensor(all_data["heights"], dtype=torch.float32),
        "pixel_heights": torch.tensor(all_data["pixel_heights"], dtype=torch.float32),
        "filenames": all_data["filenames"],
        "bbox_list": all_data["bbox_list"],
        "cam_heights": all_data["cam_heights"],
        "metadata": {
            "total_images": len(image_files),
            "success_count": len(all_data["features"]),
            "failed_count": 0,
            "bbox_type": "hand_annotated",
            "annotations_file": str(ANNOTATIONS_FILE),
            "feature_mode": feature_mode,
            "margin": args.margin,
            "augmentation": {
                "applied_only_to_train_indices": True,
                "test_indices_file": str(Path(args.test_indices_file).resolve()),
                "seed": args.seed,
                "background_prob": args.background_prob,
                "background_farthest_ratio": args.background_farthest_ratio,
                "brightness_range": args.brightness_range,
                "contrast_range": args.contrast_range,
                "saturation_range": args.saturation_range,
                "hue_range": args.hue_range,
            },
        },
    }
    return bundle


def main() -> None:
    args = parse_args()
    bundle = build_bundle(args)
    out_path = Path(args.output_file)
    torch.save(bundle, out_path)
    print(f"Saved augmented bundle: {out_path}")


if __name__ == "__main__":
    main()
