#!/usr/bin/env python3
"""
Evaluate a geometry-only height prior derived from DINOv3 attention maps.

Phase A goal:
- Use ROI-level CLS-to-patch attention to localize top/bottom plant patches.
- Convert the vertical attention span into a geometric height estimate.
- Report correlation against ground-truth hand-bbox training data heights.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from feature_extraction_utils import extract_feature_from_image, load_dinov3_components


DATASET_DIR = Path(r"D:\cornTrain\ZeroShotCornPhenotypingAgent\dataset")
ANNOTATIONS_FILE = DATASET_DIR / "plant_annotations.json"
MODEL_PATH = Path(r"D:\cornTrain\DINOV3")
DEFAULT_OUTPUT_JSON = Path(r"D:\cornTrain\DINOV3\attention_geometry_prior_phaseA.json")
NUM_REGISTER_TOKENS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate attention-derived geometry height prior.")
    parser.add_argument("--threshold-ratio", type=float, default=0.6)
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    return parser.parse_args()


def parse_filename_info(filename: str) -> tuple[float | None, float | None]:
    match = re.match(r"(\d+(?:\.\d+)?)", filename)
    cam_h = float(match.group(1)) if match else None
    parts = filename.split("-")
    if len(parts) >= 2:
        try:
            real_h = float(parts[1])
        except ValueError:
            real_h = None
    else:
        real_h = None
    return cam_h, real_h


def crop_bbox(image_pil: Image.Image, bbox: list[float] | tuple[float, float, float, float]) -> Image.Image:
    x1, y1, x2, y2 = bbox
    return image_pil.crop((max(0, int(x1)), max(0, int(y1)), min(image_pil.width, int(x2)), min(image_pil.height, int(y2))))


def attention_grid(attn_weights, num_register_tokens: int = NUM_REGISTER_TOKENS) -> np.ndarray:
    weights = attn_weights.detach().cpu().numpy().reshape(-1)
    if num_register_tokens:
        if weights.shape[0] <= num_register_tokens:
            raise ValueError("Attention vector is shorter than the number of register tokens.")
        weights = weights[num_register_tokens:]
    side = int(round(math.sqrt(weights.shape[0])))
    if side * side != weights.shape[0]:
        raise ValueError(f"Patch token count {weights.shape[0]} is not a square grid.")
    return weights.reshape(side, side)


def estimate_height_threshold(grid: np.ndarray, camera_height_cm: float, threshold_ratio: float) -> dict[str, float]:
    threshold = float(grid.max()) * threshold_ratio
    mask = grid >= threshold
    if not mask.any():
        peak_row = int(np.argmax(grid.sum(axis=1)))
        top_row = peak_row
        bottom_row = peak_row
    else:
        rows = np.where(mask)[0]
        top_row = int(rows.min())
        bottom_row = int(rows.max())

    patch_h_px = 224.0 / grid.shape[0]
    span_px = (bottom_row - top_row + 1) * patch_h_px
    est_cm = (span_px / 224.0) * float(camera_height_cm)
    return {
        "top_row": top_row,
        "bottom_row": bottom_row,
        "span_px": float(span_px),
        "estimated_height_cm": float(est_cm),
    }


def estimate_height_projection(grid: np.ndarray, camera_height_cm: float, threshold_ratio: float) -> dict[str, float]:
    row_signal = grid.mean(axis=1)
    threshold = float(row_signal.max()) * threshold_ratio
    rows = np.where(row_signal >= threshold)[0]
    if rows.size == 0:
        peak_row = int(np.argmax(row_signal))
        top_row = peak_row
        bottom_row = peak_row
    else:
        top_row = int(rows.min())
        bottom_row = int(rows.max())

    patch_h_px = 224.0 / grid.shape[0]
    span_px = (bottom_row - top_row + 1) * patch_h_px
    est_cm = (span_px / 224.0) * float(camera_height_cm)
    return {
        "top_row": top_row,
        "bottom_row": bottom_row,
        "span_px": float(span_px),
        "estimated_height_cm": float(est_cm),
    }


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    args = parse_args()
    with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    components = load_dinov3_components(MODEL_PATH, device)

    gt_heights = []
    threshold_estimates = []
    projection_estimates = []
    per_sample = []

    for filename in tqdm(sorted(annotations.keys()), desc="Phase A geometry"):
        cam_h, real_h = parse_filename_info(filename)
        bbox = annotations[filename].get("bbox")
        if real_h is None or cam_h is None or not bbox:
            continue

        image_path = DATASET_DIR / filename
        if not image_path.exists():
            continue

        image_pil = Image.open(image_path).convert("RGB")
        roi = crop_bbox(image_pil, bbox)
        if roi.width == 0 or roi.height == 0:
            continue

        _, attn_weights = extract_feature_from_image(
            roi,
            processor=components["processor"],
            model=components["model"],
            device=device,
            feature_mode="attn_weighted",
            resize_to=None,
            return_attention_weights=True,
        )
        if attn_weights is None:
            raise RuntimeError("Expected attention weights in attn_weighted mode.")

        grid = attention_grid(attn_weights)
        threshold_info = estimate_height_threshold(grid, cam_h, args.threshold_ratio)
        projection_info = estimate_height_projection(grid, cam_h, args.threshold_ratio)

        gt_heights.append(real_h)
        threshold_estimates.append(threshold_info["estimated_height_cm"])
        projection_estimates.append(projection_info["estimated_height_cm"])
        per_sample.append(
            {
                "filename": filename,
                "camera_height_cm": cam_h,
                "true_height_cm": real_h,
                "threshold_estimated_height_cm": threshold_info["estimated_height_cm"],
                "projection_estimated_height_cm": projection_info["estimated_height_cm"],
                "threshold_top_row": threshold_info["top_row"],
                "threshold_bottom_row": threshold_info["bottom_row"],
                "projection_top_row": projection_info["top_row"],
                "projection_bottom_row": projection_info["bottom_row"],
            }
        )

    gt = np.asarray(gt_heights, dtype=np.float32)
    threshold_arr = np.asarray(threshold_estimates, dtype=np.float32)
    projection_arr = np.asarray(projection_estimates, dtype=np.float32)

    threshold_corr = pearson_corr(threshold_arr, gt)
    projection_corr = pearson_corr(projection_arr, gt)

    result = {
        "sample_count": int(gt.size),
        "threshold_ratio": args.threshold_ratio,
        "threshold_method": {
            "pearson_corr": threshold_corr,
            "mean_estimated_height_cm": float(threshold_arr.mean()),
            "std_estimated_height_cm": float(threshold_arr.std()),
        },
        "projection_method": {
            "pearson_corr": projection_corr,
            "mean_estimated_height_cm": float(projection_arr.mean()),
            "std_estimated_height_cm": float(projection_arr.std()),
        },
        "ground_truth": {
            "mean_height_cm": float(gt.mean()),
            "std_height_cm": float(gt.std()),
        },
        "samples": per_sample,
    }

    output_path = Path(args.output_json)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        json.dumps(
            {
                "sample_count": int(gt.size),
                "threshold_corr": threshold_corr,
                "projection_corr": projection_corr,
                "output_json": str(output_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
