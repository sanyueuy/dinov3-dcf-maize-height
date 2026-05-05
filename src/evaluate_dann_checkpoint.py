#!/usr/bin/env python3
"""
Evaluate a DANN checkpoint on the fixed source split and DATA325 zero-shot.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dann_model import DiffCornDANN
from evaluate_data325_zero_shot import load_data325_records
from feature_extraction_utils import extract_feature_from_image, load_dinov3_components


DEFAULT_CHECKPOINT = Path(r"D:\cornTrain\DINOV3\checkpoints\diffcorn_fusion_hand_dann_best.pth")
DEFAULT_SOURCE_BUNDLE = Path(r"D:\cornTrain\DINOV3\training_bundle_hand_attn.pt")
DEFAULT_TEST_INDICES = Path(r"D:\cornTrain\DINOV3\checkpoints\hand_bbox_test_indices.json")
DEFAULT_DINO_PATH = Path(r"D:\cornTrain\DINOV3")
DEFAULT_CAPTURE_HEIGHT_MAPPING_CSV = Path(
    r"D:\wechatfile\xwechat_files\wxid_9x0cacfdglz622_706c\msg\file\2026-04\largest_dataset_file_capture_height_mapping.csv"
)
DEFAULT_ANNOTATIONS = Path(r"D:\cornTrain\dcf-bbox-eval-tool\data\annotations.json")
DEFAULT_OUTPUT_DIR = Path(r"D:\cornTrain\DINOV3\data325_zero_shot_dann")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a DANN checkpoint.")
    parser.add_argument("--checkpoint-path", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--source-bundle", default=str(DEFAULT_SOURCE_BUNDLE))
    parser.add_argument("--test-indices-file", default=str(DEFAULT_TEST_INDICES))
    parser.add_argument("--dino-path", default=str(DEFAULT_DINO_PATH))
    parser.add_argument("--annotations-file", default=str(DEFAULT_ANNOTATIONS))
    parser.add_argument("--capture-height-mapping-csv", default=str(DEFAULT_CAPTURE_HEIGHT_MAPPING_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-suffix", default="_dann")
    return parser.parse_args()


def load_dann_model(checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = DiffCornDANN(use_camera_condition=True, cam_h_scale=200.0).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    return model, device, checkpoint


def predict_height(outputs: torch.Tensor) -> torch.Tensor:
    return outputs[:, 0::4].sum(dim=1)


def evaluate_source(model: DiffCornDANN, bundle_path: str, test_indices_file: str, device: torch.device) -> dict:
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    test_indices = json.loads(Path(test_indices_file).read_text(encoding="utf-8"))["test_indices"]
    features = bundle["features"].float().to(device)
    cam_heights = torch.tensor(bundle["cam_heights"], dtype=torch.float32).to(device)
    heights = bundle["heights"].float().cpu().numpy()
    filenames = bundle["filenames"]

    preds = []
    gts = []
    rows = []
    with torch.no_grad():
        for idx in test_indices:
            output, _, _ = model(features[idx : idx + 1], cam_heights[idx : idx + 1], grl_lambda=0.0)
            pred = float(predict_height(output).cpu().item())
            gt = float(heights[idx])
            preds.append(pred)
            gts.append(gt)
            rows.append({"filename": filenames[idx], "true_height_cm": gt, "pred_height_cm": pred, "error_cm": pred - gt})

    preds_arr = np.asarray(preds, dtype=float)
    gt_arr = np.asarray(gts, dtype=float)
    errors = preds_arr - gt_arr
    metrics = {
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(errors ** 2))),
        "median_abs_error": float(np.median(np.abs(errors))),
        "n_samples": int(len(rows)),
    }
    return {"metrics": metrics, "samples": rows}


def crop_bbox(image_pil: Image.Image, bbox) -> Image.Image:
    x1, y1, x2, y2 = bbox
    return image_pil.crop((int(x1), int(y1), int(x2), int(y2)))


def evaluate_data325(model: DiffCornDANN, dino_path: str, annotations_file: str, capture_height_mapping_csv: str, device: torch.device) -> dict:
    records, resolution_summary = load_data325_records(annotations_file, capture_height_mapping_csv)
    components = load_dinov3_components(dino_path, device, cache=True)

    box_results = []
    for rec in records:
        image_pil = Image.open(rec["image_path"]).convert("RGB")
        roi = crop_bbox(image_pil, rec["bbox"])
        feat = extract_feature_from_image(
            roi,
            processor=components["processor"],
            model=components["model"],
            device=device,
            feature_mode="attn_weighted",
            resize_to=(224, 224),
        )
        with torch.no_grad():
            cam = torch.tensor([[rec["camera_height_cm"]]], dtype=torch.float32, device=device)
            output, _, _ = model(feat, cam, grl_lambda=0.0)
            pred_height = float(predict_height(output).cpu().item())
        abs_error = abs(pred_height - rec["true_height_cm"])
        box_results.append({**rec, "pred_height_cm": pred_height, "abs_error_cm": abs_error})

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
    result = {
        "camera_height_resolution": resolution_summary,
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
    return result


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, device, checkpoint = load_dann_model(args.checkpoint_path)
    source_result = evaluate_source(model, args.source_bundle, args.test_indices_file, device)
    data325_result = evaluate_data325(model, args.dino_path, args.annotations_file, args.capture_height_mapping_csv, device)

    source_path = output_dir / f"source_eval{args.output_suffix}.json"
    source_path.write_text(json.dumps(source_result, indent=2, ensure_ascii=False), encoding="utf-8")

    comparison = {
        "checkpoint_path": args.checkpoint_path,
        "output_suffix": args.output_suffix,
        "training_history": checkpoint.get("history", {}),
        "source_eval": source_result["metrics"],
        "data325_eval": data325_result,
    }
    comparison_path = output_dir / f"data325_zero_shot_comparison{args.output_suffix}.json"
    comparison_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({
        "source_mae_cm": source_result["metrics"]["MAE"],
        "data325_mae_cm": data325_result["summary"]["mae_cm"],
        "domain_acc_last": checkpoint.get("history", {}).get("domain_acc", [None])[-1] if checkpoint.get("history") else None,
        "source_path": str(source_path),
        "comparison_path": str(comparison_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
