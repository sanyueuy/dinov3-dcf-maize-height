#!/usr/bin/env python3
"""
Compute source/target feature statistics and evaluate feature-space alignment on DATA325.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from evaluate_data325_zero_shot import load_data325_records, load_checkpoint_model
from feature_extraction_utils import extract_feature_from_image, load_dinov3_components


DEFAULT_SOURCE_BUNDLE = Path(r"D:\cornTrain\DINOV3\training_bundle_hand_attn.pt")
DEFAULT_DATA325_ANNOTATIONS = Path(r"D:\cornTrain\dcf-bbox-eval-tool\data\annotations.json")
DEFAULT_DINO_PATH = Path(r"D:\cornTrain\DINOV3")
DEFAULT_CAPTURE_HEIGHT_MAPPING_CSV = Path(
    r"D:\wechatfile\xwechat_files\wxid_9x0cacfdglz622_706c\msg\file\2026-04\largest_dataset_file_capture_height_mapping.csv"
)
DEFAULT_MODEL_CHECKPOINT = Path(r"D:\cornTrain\DINOV3\checkpoints\diffcorn_fusion_hand_attn_aug_best.pth")
DEFAULT_OUTPUT_DIR = Path(r"D:\cornTrain\DINOV3\data325_zero_shot_attn_aug_featurealign")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DATA325 using source-target feature-stat alignment.")
    parser.add_argument("--source-bundle", default=str(DEFAULT_SOURCE_BUNDLE))
    parser.add_argument("--annotations-file", default=str(DEFAULT_DATA325_ANNOTATIONS))
    parser.add_argument("--capture-height-mapping-csv", default=str(DEFAULT_CAPTURE_HEIGHT_MAPPING_CSV))
    parser.add_argument("--dino-path", default=str(DEFAULT_DINO_PATH))
    parser.add_argument("--checkpoint-path", default=str(DEFAULT_MODEL_CHECKPOINT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-suffix", default="_attn_aug_featurealign")
    parser.add_argument("--eps", type=float, default=1e-6)
    return parser.parse_args()


def load_source_stats(bundle_path: str, eps: float) -> tuple[np.ndarray, np.ndarray, dict]:
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    feats = bundle["features"].float().cpu().numpy()
    mu = feats.mean(axis=0)
    sigma = feats.std(axis=0)
    sigma = np.maximum(sigma, eps)
    summary = {
        "sample_count": int(feats.shape[0]),
        "feature_dim": int(feats.shape[1]),
        "mean_norm_l2": float(np.linalg.norm(mu)),
        "mean_feature_std": float(np.mean(sigma)),
    }
    return mu, sigma, summary


def load_target_image_paths_from_csv(mapping_csv: str, image_root: Path) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    with open(mapping_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("kind") != "original":
                continue
            file_name = row["file_name"]
            if file_name in seen:
                continue
            seen.add(file_name)
            path = image_root / file_name
            if path.exists():
                paths.append(path)
    return paths


def compute_target_stats(image_paths: list[Path], dino_path: str, eps: float) -> tuple[np.ndarray, np.ndarray, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    components = load_dinov3_components(dino_path, device, cache=True)
    feats = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        feat = extract_feature_from_image(
            image,
            processor=components["processor"],
            model=components["model"],
            device=device,
            feature_mode="attn_weighted",
            resize_to=(224, 224),
        )
        feats.append(feat.detach().cpu().numpy().reshape(-1))
    arr = np.stack(feats).astype(np.float32)
    mu = arr.mean(axis=0)
    sigma = np.maximum(arr.std(axis=0), eps)
    summary = {
        "sample_count": int(arr.shape[0]),
        "feature_dim": int(arr.shape[1]),
        "mean_norm_l2": float(np.linalg.norm(mu)),
        "mean_feature_std": float(np.mean(sigma)),
    }
    return mu, sigma, summary


def crop_bbox(image_pil: Image.Image, bbox) -> Image.Image:
    x1, y1, x2, y2 = bbox
    return image_pil.crop((int(x1), int(y1), int(x2), int(y2)))


def align_feature(feat: np.ndarray, mu_target: np.ndarray, sigma_target: np.ndarray, mu_source: np.ndarray, sigma_source: np.ndarray) -> np.ndarray:
    return ((feat - mu_target) / sigma_target) * sigma_source + mu_source


def evaluate_with_alignment(
    records: list[dict],
    checkpoint_path: str,
    dino_path: str,
    mu_source: np.ndarray,
    sigma_source: np.ndarray,
    mu_target: np.ndarray,
    sigma_target: np.ndarray,
) -> dict:
    model, device = load_checkpoint_model(checkpoint_path)
    components = load_dinov3_components(dino_path, device, cache=True)
    processor = components["processor"]
    dinov3 = components["model"]

    box_results = []
    for rec in records:
        image_pil = Image.open(rec["image_path"]).convert("RGB")
        roi = crop_bbox(image_pil, rec["bbox"])
        feat = extract_feature_from_image(
            roi,
            processor=processor,
            model=dinov3,
            device=device,
            feature_mode="attn_weighted",
            resize_to=(224, 224),
        )
        feat_np = feat.detach().cpu().numpy().reshape(-1)
        aligned_np = align_feature(feat_np, mu_target, sigma_target, mu_source, sigma_source).astype(np.float32)
        aligned = torch.from_numpy(aligned_np).unsqueeze(0).to(device)
        with torch.no_grad():
            cam = torch.tensor([[rec["camera_height_cm"]]], dtype=torch.float32, device=device)
            output = model(aligned, cam if model.use_camera_condition else None) if model.use_camera_condition else model(aligned)
            pred_height = float(output[:, 0::4].sum(dim=1).cpu().item())
        abs_error = abs(pred_height - rec["true_height_cm"])
        box_results.append(
            {
                **rec,
                "pred_height_cm": pred_height,
                "abs_error_cm": abs_error,
                "alignment": "feature_stat_match",
            }
        )

    abs_errors = np.array([x["abs_error_cm"] for x in box_results], dtype=float)
    image_groups: dict[str, list[dict]] = defaultdict(list)
    for row in box_results:
        image_groups[row["file_name"]].append(row)

    image_metrics = []
    for file_name, rows in image_groups.items():
        errs = np.array([r["abs_error_cm"] for r in rows], dtype=float)
        image_metrics.append(
            {
                "file_name": file_name,
                "box_count": len(rows),
                "mae_cm": float(errs.mean()),
                "rmse_cm": float(np.sqrt(np.mean(errs ** 2))),
                "max_abs_error_cm": float(errs.max()),
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
        "label": "attn_aug_featurealign",
        "checkpoint_path": checkpoint_path,
        "feature_mode": "attn_weighted",
        "alignment_mode": "per_feature_target_to_source",
        "summary": {
            "boxes": len(box_results),
            "images": len(image_groups),
            "mae_cm": float(abs_errors.mean()),
            "rmse_cm": float(np.sqrt(np.mean(abs_errors ** 2))),
            "median_abs_error_cm": float(np.median(abs_errors)),
        },
        "image_metrics": image_metrics,
        "bucket_metrics": bucket_metrics,
        "group_70cm": {"count": int(len(rows70)), "mae_cm": float(errs70.mean()) if len(errs70) else None},
        "box_results": box_results,
    }


def main() -> None:
    args = parse_args()
    annotations_obj = json.loads(Path(args.annotations_file).read_text(encoding="utf-8"))
    image_root = Path(annotations_obj["image_root"])
    records, resolution_summary = load_data325_records(args.annotations_file, args.capture_height_mapping_csv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mu_source, sigma_source, source_summary = load_source_stats(args.source_bundle, args.eps)
    target_paths = load_target_image_paths_from_csv(args.capture_height_mapping_csv, image_root)
    mu_target, sigma_target, target_summary = compute_target_stats(target_paths, args.dino_path, args.eps)

    mean_shift = float(np.linalg.norm(mu_target - mu_source) / max(np.linalg.norm(mu_source), args.eps))
    variance_shift = float(np.mean(np.abs(sigma_target - sigma_source) / np.maximum(sigma_source, args.eps)))

    result = evaluate_with_alignment(
        records=records,
        checkpoint_path=args.checkpoint_path,
        dino_path=args.dino_path,
        mu_source=mu_source,
        sigma_source=sigma_source,
        mu_target=mu_target,
        sigma_target=sigma_target,
    )

    stats_summary = {
        "source_bundle": args.source_bundle,
        "target_image_root": str(image_root),
        "target_valid_original_images": len(target_paths),
        "camera_height_resolution": resolution_summary,
        "source_feature_stats": source_summary,
        "target_feature_stats": target_summary,
        "mean_shift_relative_l2": mean_shift,
        "mean_shift_relative_pct": mean_shift * 100.0,
        "variance_shift_relative_mean": variance_shift,
        "variance_shift_relative_pct": variance_shift * 100.0,
    }

    stats_path = output_dir / f"data325_feature_stats{args.output_suffix}.json"
    stats_path.write_text(json.dumps(stats_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    comparison = {
        "annotations_file": args.annotations_file,
        "camera_height_mode": "per_image_csv",
        "capture_height_mapping_csv": args.capture_height_mapping_csv,
        "output_suffix": args.output_suffix,
        "feature_alignment": {
            "mode": "target_to_source_per_feature_standardization",
            "mean_shift_relative_pct": mean_shift * 100.0,
            "variance_shift_relative_pct": variance_shift * 100.0,
        },
        "model_results": [result],
    }
    out_path = output_dir / f"data325_zero_shot_comparison{args.output_suffix}.json"
    out_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        json.dumps(
            {
                "stats_path": str(stats_path),
                "comparison_path": str(out_path),
                "target_valid_original_images": len(target_paths),
                "mean_shift_relative_pct": mean_shift * 100.0,
                "variance_shift_relative_pct": variance_shift * 100.0,
                "mae_cm": result["summary"]["mae_cm"],
                "bucket_0_80_mae_cm": next((x["mae_cm"] for x in result["bucket_metrics"] if x["bucket"] == "0-80"), None),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
