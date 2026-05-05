#!/usr/bin/env python3
"""
Build a hand-annotated bbox feature bundle from the original dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from feature_extraction_utils import (
    extract_feature_from_image,
    load_dinov3_components,
    validate_feature_mode,
)


MODEL_PATH = Path(".")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_DIR = Path(r"D:\cornTrain\ZeroShotCornPhenotypingAgent\dataset")
ANNOTATIONS_FILE = DATASET_DIR / "plant_annotations.json"
DEFAULT_OUTPUT_FILE = "training_bundle_hand.pt"
DEFAULT_BENCHMARK_OUTPUT_DIR = Path("./hand_bbox_benchmark")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build hand-bbox DINOv3 feature bundle.")
    parser.add_argument("--feature-mode", default="attn_weighted", choices=["cls", "patch_mean", "attn_weighted"])
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--benchmark-output-dir", default=str(DEFAULT_BENCHMARK_OUTPUT_DIR))
    parser.add_argument("--margin", type=int, default=0)
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


def extract_dinov3_features_from_bbox(
    image_pil: Image.Image,
    bbox,
    processor,
    model,
    device: torch.device,
    feature_mode: str,
    margin: int,
) -> torch.Tensor | None:
    roi = crop_bbox(image_pil, bbox, margin=margin)
    if roi.width == 0 or roi.height == 0:
        return None
    features = extract_feature_from_image(
        roi,
        processor=processor,
        model=model,
        device=device,
        feature_mode=feature_mode,
        resize_to=None,
    )
    return features.cpu().squeeze()


def generate_benchmark_report(bundle: dict, failed_list: list[dict]) -> str:
    features = bundle["features"]
    heights = bundle["heights"].tolist()
    pixel_heights = bundle["pixel_heights"].tolist()
    cam_heights = bundle["cam_heights"]
    metadata = bundle["metadata"]
    valid_heights = [h for h in heights if h > 0]

    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("HAND-ANNOTATED BBOX BENCHMARK REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Feature mode:         {metadata.get('feature_mode', 'unknown')}")
    lines.append(f"Total samples:        {metadata['total_images']}")
    lines.append(f"Success count:        {metadata['success_count']}")
    lines.append(f"Failed count:         {metadata['failed_count']}")
    lines.append(f"Success rate:         {100 * metadata['success_count'] / max(metadata['total_images'], 1):.1f}%")
    lines.append(f"Feature dimension:    {features.shape[1]}")
    lines.append("")
    if valid_heights:
        lines.append("Height Statistics")
        lines.append("-" * 40)
        lines.append(f"Min height:           {min(valid_heights):.1f} cm")
        lines.append(f"Max height:           {max(valid_heights):.1f} cm")
        lines.append(f"Mean height:          {np.mean(valid_heights):.1f} cm")
        lines.append(f"Std height:           {np.std(valid_heights):.1f} cm")
        lines.append("")
    lines.append("Pixel Height Statistics")
    lines.append("-" * 40)
    lines.append(f"Min pixel height:     {min(pixel_heights)} px")
    lines.append(f"Max pixel height:     {max(pixel_heights)} px")
    lines.append(f"Mean pixel height:    {np.mean(pixel_heights):.1f} px")
    lines.append(f"Std pixel height:     {np.std(pixel_heights):.1f} px")
    if valid_heights:
        px_per_cm = [p / h for p, h in zip(pixel_heights, heights) if h > 0]
        lines.append(f"Pixel/cm ratio:       {np.mean(px_per_cm):.2f} px/cm")
    lines.append("")
    lines.append("Camera Height Distribution")
    lines.append("-" * 40)
    cam_106 = sum(1 for h in cam_heights if abs(h - 106) < 1)
    cam_140 = sum(1 for h in cam_heights if abs(h - 140) < 1)
    lines.append(f"106 cm camera:        {cam_106} samples")
    lines.append(f"140 cm camera:        {cam_140} samples")
    lines.append("")
    lines.append("Feature Statistics")
    lines.append("-" * 40)
    lines.append(f"Feature mean:         {features.mean():.4f}")
    lines.append(f"Feature std:          {features.std():.4f}")
    lines.append(f"Feature min:          {features.min():.4f}")
    lines.append(f"Feature max:          {features.max():.4f}")
    lines.append("")
    if failed_list:
        lines.append("Failed Samples")
        lines.append("-" * 40)
        for item in failed_list[:20]:
            lines.append(f"  {item['filename']}: {item['reason']}")
        if len(failed_list) > 20:
            lines.append(f"  ... and {len(failed_list) - 20} more")
        lines.append("")
    return "\n".join(lines)


def run_benchmark(feature_mode: str, output_file: str, benchmark_output_dir: str, margin: int) -> tuple[dict, dict]:
    feature_mode = validate_feature_mode(feature_mode)
    output_path = Path(output_file)
    benchmark_dir = Path(benchmark_output_dir)
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Hand-annotated BBox Benchmark")
    print("=" * 70)
    print(f"Feature mode: {feature_mode}")

    with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    print(f"Loaded {len(annotations)} annotations")

    components = load_dinov3_components(MODEL_PATH, DEVICE)
    processor = components["processor"]
    model = components["model"]
    print(f"Model loaded on {DEVICE}")

    image_files = sorted(
        f for f in os.listdir(DATASET_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    print(f"Found {len(image_files)} images")

    all_data = {
        "features": [],
        "heights": [],
        "pixel_heights": [],
        "filenames": [],
        "bbox_list": [],
        "cam_heights": [],
        "failed": [],
    }

    for filename in tqdm(image_files, desc="Processing"):
        image_path = DATASET_DIR / filename
        if filename not in annotations:
            all_data["failed"].append({"filename": filename, "reason": "No bbox"})
            continue

        bbox = annotations[filename].get("bbox")
        if not bbox or len(bbox) != 4:
            all_data["failed"].append({"filename": filename, "reason": "Invalid bbox"})
            continue

        try:
            image_pil = Image.open(image_path).convert("RGB")
            cam_h, real_h = parse_filename_info(filename)
            features = extract_dinov3_features_from_bbox(
                image_pil=image_pil,
                bbox=bbox,
                processor=processor,
                model=model,
                device=DEVICE,
                feature_mode=feature_mode,
                margin=margin,
            )
            if features is None:
                all_data["failed"].append({"filename": filename, "reason": "Feature extraction failed"})
                continue

            x1, y1, x2, y2 = bbox
            pixel_height = y2 - y1
            all_data["features"].append(features.numpy())
            all_data["heights"].append(real_h if real_h else 0.0)
            all_data["pixel_heights"].append(pixel_height)
            all_data["filenames"].append(filename)
            all_data["bbox_list"].append(bbox)
            all_data["cam_heights"].append(cam_h if cam_h else 0.0)
        except Exception as exc:
            all_data["failed"].append({"filename": filename, "reason": str(exc)})

    features_tensor = torch.tensor(np.stack(all_data["features"]))
    heights_tensor = torch.tensor(all_data["heights"])
    pixel_heights_tensor = torch.tensor(all_data["pixel_heights"])

    bundle = {
        "features": features_tensor,
        "heights": heights_tensor,
        "pixel_heights": pixel_heights_tensor,
        "filenames": all_data["filenames"],
        "bbox_list": all_data["bbox_list"],
        "cam_heights": all_data["cam_heights"],
        "metadata": {
            "total_images": len(image_files),
            "success_count": len(all_data["features"]),
            "failed_count": len(all_data["failed"]),
            "bbox_type": "hand_annotated",
            "annotations_file": str(ANNOTATIONS_FILE),
            "feature_mode": feature_mode,
            "margin": margin,
        },
    }
    torch.save(bundle, output_path)

    heights = [h for h in all_data["heights"] if h > 0]
    pixel_heights = all_data["pixel_heights"]
    stats = {
        "dataset_info": {
            "total_samples": len(all_data["features"]),
            "failed_samples": len(all_data["failed"]),
            "success_rate": f"{100 * len(all_data['features']) / max(len(image_files), 1):.1f}%",
            "bbox_type": "hand_annotated",
            "feature_mode": feature_mode,
        },
        "height_statistics": {
            "min_height_cm": min(heights) if heights else 0,
            "max_height_cm": max(heights) if heights else 0,
            "mean_height_cm": float(np.mean(heights)) if heights else 0,
            "std_height_cm": float(np.std(heights)) if heights else 0,
        },
        "pixel_height_statistics": {
            "min_pixel_height": min(pixel_heights),
            "max_pixel_height": max(pixel_heights),
            "mean_pixel_height": float(np.mean(pixel_heights)),
            "std_pixel_height": float(np.std(pixel_heights)),
        },
        "feature_statistics": {
            "feature_dim": int(features_tensor.shape[1]),
            "feature_mean": float(features_tensor.mean()),
            "feature_std": float(features_tensor.std()),
        },
    }

    report = generate_benchmark_report(bundle, all_data["failed"])
    report_path = benchmark_dir / f"benchmark_report_{feature_mode}.txt"
    report_path.write_text(report, encoding="utf-8")
    json_path = benchmark_dir / f"benchmark_results_{feature_mode}.json"
    json_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved bundle: {output_path}")
    print(f"Saved report: {report_path}")
    print(f"Saved stats:  {json_path}")
    return bundle, stats


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(
        feature_mode=args.feature_mode,
        output_file=args.output_file,
        benchmark_output_dir=args.benchmark_output_dir,
        margin=args.margin,
    )
