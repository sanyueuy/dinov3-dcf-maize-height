#!/usr/bin/env python3
"""
Run zero-shot evaluation on DATA325 annotations for one or more checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance

from feature_extraction_utils import extract_feature_from_image, load_dinov3_components, validate_feature_mode
from model import DiffCornMLP


DEFAULT_DATA325_ANNOTATIONS = Path(r"D:\cornTrain\dcf-bbox-eval-tool\data\annotations.json")
DEFAULT_OUTPUT_DIR = Path(r"D:\cornTrain\DINOV3\data325_zero_shot")
DEFAULT_DINO_PATH = Path(r"D:\cornTrain\DINOV3")
DEFAULT_CAPTURE_HEIGHT_MAPPING_CSV = Path(
    r"D:\wechatfile\xwechat_files\wxid_9x0cacfdglz622_706c\msg\file\2026-04\largest_dataset_file_capture_height_mapping.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one or more checkpoints on DATA325 zero-shot.")
    parser.add_argument("--annotations-file", default=str(DEFAULT_DATA325_ANNOTATIONS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dino-path", default=str(DEFAULT_DINO_PATH))
    parser.add_argument("--capture-height-mapping-csv", default=str(DEFAULT_CAPTURE_HEIGHT_MAPPING_CSV))
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--tta-count", type=int, default=1)
    parser.add_argument("--tta-brightness-range", type=float, default=0.2)
    parser.add_argument("--tta-contrast-range", type=float, default=0.2)
    parser.add_argument("--tta-saturation-range", type=float, default=0.2)
    parser.add_argument("--tta-hue-range", type=float, default=0.08)
    parser.add_argument("--tta-seed", type=int, default=42)
    parser.add_argument(
        "--model-spec",
        action="append",
        default=[],
        help="Format: label=checkpoint_path=feature_mode",
    )
    return parser.parse_args()


def load_capture_height_mapping(mapping_csv: str) -> tuple[dict[str, float], list[dict]]:
    mapping: dict[str, float] = {}
    conflicts: list[dict] = []
    with open(mapping_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("kind") != "original":
                continue
            file_name = row["file_name"]
            capture_height_cm = float(row["capture_height_cm"])
            if file_name in mapping and mapping[file_name] != capture_height_cm:
                conflicts.append(
                    {
                        "file_name": file_name,
                        "existing_capture_height_cm": mapping[file_name],
                        "new_capture_height_cm": capture_height_cm,
                    }
                )
                continue
            mapping[file_name] = capture_height_cm
    return mapping, conflicts


def load_data325_records(annotations_file: str, capture_height_mapping_csv: str) -> tuple[list[dict], dict]:
    obj = json.loads(Path(annotations_file).read_text(encoding="utf-8"))
    image_root = Path(obj["image_root"])
    capture_height_mapping, conflicts = load_capture_height_mapping(capture_height_mapping_csv)
    if conflicts:
        raise ValueError(f"Conflicting capture_height_cm rows found: {conflicts[:5]}")

    records = []
    resolved_image_heights: dict[str, float] = {}
    missing_images: list[str] = []
    for img in obj.get("images", []):
        if img.get("status") != "complete":
            continue
        image_path = image_root / img["relative_path"]
        file_name = img["file_name"]
        if file_name not in capture_height_mapping:
            missing_images.append(file_name)
            continue
        camera_height_cm = capture_height_mapping[file_name]
        resolved_image_heights[file_name] = camera_height_cm
        for box in img.get("boxes", []):
            true_height = box.get("true_height_cm")
            if true_height is None:
                continue
            records.append(
                {
                    "image_id": img["image_id"],
                    "file_name": file_name,
                    "image_path": str(image_path),
                    "box_id": box["box_id"],
                    "bbox": box["bbox"],
                    "true_height_cm": float(true_height),
                    "camera_height_cm": camera_height_cm,
                }
            )
    if missing_images:
        raise ValueError(
            f"Missing capture_height_cm mapping for {len(missing_images)} images: {sorted(set(missing_images))[:10]}"
        )
    resolution_summary = {
        "matched_images": len(resolved_image_heights),
        "missing_images": sorted(set(missing_images)),
        "duplicate_mapping_conflicts": conflicts,
        "resolved_image_heights": [
            {"file_name": file_name, "capture_height_cm": capture_height_cm}
            for file_name, capture_height_cm in sorted(resolved_image_heights.items())
        ],
    }
    return records, resolution_summary


def parse_model_spec(spec: str) -> tuple[str, str, str]:
    parts = spec.split("=")
    if len(parts) != 3:
        raise ValueError(f"Invalid --model-spec {spec!r}; expected label=checkpoint_path=feature_mode")
    label, checkpoint_path, feature_mode = parts
    return label, checkpoint_path, validate_feature_mode(feature_mode)


def load_checkpoint_model(checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    first_layer_key = next((k for k in state_dict if "network.0.weight" in k), None)
    use_camera_condition = True
    if first_layer_key:
        use_camera_condition = state_dict[first_layer_key].shape[1] == 1025
    model = DiffCornMLP(input_dim=1024, use_camera_condition=use_camera_condition)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model, device


def crop_bbox(image_pil: Image.Image, bbox) -> Image.Image:
    x1, y1, x2, y2 = bbox
    return image_pil.crop((int(x1), int(y1), int(x2), int(y2)))


def apply_hue_shift(image: Image.Image, rng: random.Random, hue_range: float) -> Image.Image:
    hsv = np.array(image.convert("HSV"), dtype=np.uint8)
    shift = int(rng.uniform(-hue_range, hue_range) * 255)
    hsv[..., 0] = (hsv[..., 0].astype(np.int16) + shift) % 256
    return Image.fromarray(hsv, mode="HSV").convert("RGB")


def apply_tta_color_jitter(
    image: Image.Image,
    rng: random.Random,
    brightness_range: float,
    contrast_range: float,
    saturation_range: float,
    hue_range: float,
) -> Image.Image:
    out = image
    factor = 1.0 + rng.uniform(-brightness_range, brightness_range)
    out = ImageEnhance.Brightness(out).enhance(factor)
    factor = 1.0 + rng.uniform(-contrast_range, contrast_range)
    out = ImageEnhance.Contrast(out).enhance(factor)
    factor = 1.0 + rng.uniform(-saturation_range, saturation_range)
    out = ImageEnhance.Color(out).enhance(factor)
    out = apply_hue_shift(out, rng, hue_range)
    return out


def predict_height_with_tta(
    roi: Image.Image,
    model,
    processor,
    dinov3,
    device,
    feature_mode: str,
    camera_height_cm: float,
    tta_count: int,
    tta_seed: int,
    tta_brightness_range: float,
    tta_contrast_range: float,
    tta_saturation_range: float,
    tta_hue_range: float,
) -> tuple[float, float, list[float]]:
    preds: list[float] = []
    for tta_idx in range(tta_count):
        if tta_idx == 0:
            aug_roi = roi
        else:
            rng = random.Random(
                hash((tta_seed, roi.size[0], roi.size[1], camera_height_cm, tta_idx)) & 0xFFFFFFFF
            )
            aug_roi = apply_tta_color_jitter(
                roi,
                rng=rng,
                brightness_range=tta_brightness_range,
                contrast_range=tta_contrast_range,
                saturation_range=tta_saturation_range,
                hue_range=tta_hue_range,
            )
        feat = extract_feature_from_image(
            aug_roi,
            processor=processor,
            model=dinov3,
            device=device,
            feature_mode=feature_mode,
            resize_to=(224, 224),
        )
        feat = feat.to(device)
        with torch.no_grad():
            cam = torch.tensor([[camera_height_cm]], dtype=torch.float32, device=device)
            output = model(feat, cam if model.use_camera_condition else None) if model.use_camera_condition else model(feat)
            preds.append(float(output[:, 0::4].sum(dim=1).cpu().item()))
    pred_mean = float(np.mean(preds))
    pred_std = float(np.std(preds))
    return pred_mean, pred_std, preds


def evaluate_model(
    label: str,
    checkpoint_path: str,
    feature_mode: str,
    records: list[dict],
    dino_path: str,
    tta_count: int,
    tta_seed: int,
    tta_brightness_range: float,
    tta_contrast_range: float,
    tta_saturation_range: float,
    tta_hue_range: float,
) -> dict:
    model, device = load_checkpoint_model(checkpoint_path)
    components = load_dinov3_components(dino_path, device, cache=True)
    processor = components["processor"]
    dinov3 = components["model"]

    box_results = []
    for rec in records:
        image_pil = Image.open(rec["image_path"]).convert("RGB")
        roi = crop_bbox(image_pil, rec["bbox"])
        pred_height, pred_std, pred_samples = predict_height_with_tta(
            roi=roi,
            model=model,
            processor=processor,
            dinov3=dinov3,
            device=device,
            feature_mode=feature_mode,
            camera_height_cm=rec["camera_height_cm"],
            tta_count=tta_count,
            tta_seed=tta_seed,
            tta_brightness_range=tta_brightness_range,
            tta_contrast_range=tta_contrast_range,
            tta_saturation_range=tta_saturation_range,
            tta_hue_range=tta_hue_range,
        )
        abs_error = abs(pred_height - rec["true_height_cm"])
        box_results.append(
            {
                **rec,
                "pred_height_cm": pred_height,
                "pred_std_cm": pred_std,
                "pred_samples_cm": pred_samples,
                "abs_error_cm": abs_error,
                "model_label": label,
                "feature_mode": feature_mode,
            }
        )

    abs_errors = np.array([x["abs_error_cm"] for x in box_results], dtype=float)
    image_groups: dict[str, list[dict]] = defaultdict(list)
    for row in box_results:
        image_groups[row["file_name"]].append(row)

    image_metrics = []
    for file_name, rows in image_groups.items():
        errs = np.array([r["abs_error_cm"] for r in rows], dtype=float)
        pred_stds = np.array([r["pred_std_cm"] for r in rows], dtype=float)
        image_metrics.append(
            {
                "file_name": file_name,
                "box_count": len(rows),
                "mae_cm": float(errs.mean()),
                "rmse_cm": float(np.sqrt(np.mean(errs ** 2))),
                "max_abs_error_cm": float(errs.max()),
                "mean_pred_std_cm": float(pred_stds.mean()),
            }
        )
    image_metrics.sort(key=lambda x: x["mae_cm"], reverse=True)

    buckets = [(0, 80), (80, 100), (100, 120), (120, 140), (140, 160), (160, 180)]
    bucket_metrics = []
    for lo, hi in buckets:
        rows = [r for r in box_results if lo <= r["true_height_cm"] < hi]
        if not rows:
            continue
        errs = np.array([r["abs_error_cm"] for r in rows], dtype=float)
        bucket_metrics.append({"bucket": f"{lo}-{hi}", "count": len(rows), "mae_cm": float(errs.mean())})

    rows70 = [r for r in box_results if r["true_height_cm"] == 70.0]
    errs70 = np.array([r["abs_error_cm"] for r in rows70], dtype=float) if rows70 else np.array([], dtype=float)

    return {
        "label": label,
        "checkpoint_path": checkpoint_path,
        "feature_mode": feature_mode,
        "tta": {
            "count": tta_count,
            "brightness_range": tta_brightness_range,
            "contrast_range": tta_contrast_range,
            "saturation_range": tta_saturation_range,
            "hue_range": tta_hue_range,
            "seed": tta_seed,
        },
        "summary": {
            "boxes": len(box_results),
            "images": len(image_groups),
            "mae_cm": float(abs_errors.mean()),
            "rmse_cm": float(np.sqrt(np.mean(abs_errors ** 2))),
            "median_abs_error_cm": float(np.median(abs_errors)),
            "mean_pred_std_cm": float(np.mean([r["pred_std_cm"] for r in box_results])),
            "mean_image_pred_std_cm": float(np.mean([item["mean_pred_std_cm"] for item in image_metrics])),
        },
        "image_metrics": image_metrics,
        "bucket_metrics": bucket_metrics,
        "group_70cm": {"count": int(len(rows70)), "mae_cm": float(errs70.mean()) if len(errs70) else None},
        "box_results": box_results,
    }


def main() -> None:
    args = parse_args()
    if not args.model_spec:
        raise SystemExit("At least one --model-spec is required.")
    records, resolution_summary = load_data325_records(args.annotations_file, args.capture_height_mapping_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.output_suffix or ""

    resolution_path = output_dir / f"data325_camera_height_resolution{suffix}.json"
    resolution_path.write_text(json.dumps(resolution_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved camera-height resolution: {resolution_path}")

    all_results = []
    for spec in args.model_spec:
        label, checkpoint_path, feature_mode = parse_model_spec(spec)
        print(f"Evaluating {label} with feature_mode={feature_mode}")
        result = evaluate_model(
            label,
            checkpoint_path,
            feature_mode,
            records,
            args.dino_path,
            args.tta_count,
            args.tta_seed,
            args.tta_brightness_range,
            args.tta_contrast_range,
            args.tta_saturation_range,
            args.tta_hue_range,
        )
        all_results.append(result)

    summary = {
        "annotations_file": args.annotations_file,
        "camera_height_mode": "per_image_csv",
        "capture_height_mapping_csv": args.capture_height_mapping_csv,
        "output_suffix": suffix,
        "tta_count": args.tta_count,
        "model_results": all_results,
    }
    out_path = output_dir / f"data325_zero_shot_comparison{suffix}.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved comparison: {out_path}")


if __name__ == "__main__":
    main()
