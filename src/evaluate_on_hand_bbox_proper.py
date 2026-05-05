#!/usr/bin/env python3
"""
Evaluate a DCF checkpoint on the fixed hand-bbox test split.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from model import DiffCornMLP


DEFAULT_BUNDLE_FILE = "training_bundle_hand.pt"
DEFAULT_MODEL_DIR = "./checkpoints"
DEFAULT_TEST_INDICES_FILE = "./checkpoints/hand_bbox_test_indices.json"
DEFAULT_OUTPUT_DIR = "./hand_bbox_benchmark/evaluation_proper"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a model on the fixed hand-bbox test split.")
    parser.add_argument("--model-name", type=str, default="diffcorn_fusion_hand_best.pth")
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--bundle-file", type=str, default=DEFAULT_BUNDLE_FILE)
    parser.add_argument("--test-indices-file", type=str, default=DEFAULT_TEST_INDICES_FILE)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-prefix", type=str, default="test_set")
    parser.add_argument("--title", type=str, default="DIFFCORN-FUSION: PROPER TEST SET EVALUATION")
    return parser.parse_args()


def load_bundle(bundle_file: str) -> dict:
    bundle = torch.load(bundle_file)
    print(f"Loaded bundle: {len(bundle['filenames'])} samples")
    return bundle


def load_test_indices(test_indices_file: str) -> list[int]:
    data = json.loads(Path(test_indices_file).read_text(encoding="utf-8"))
    test_indices = data["test_indices"]
    print(f"Loaded test indices: {len(test_indices)} samples")
    return test_indices


def load_model(model_dir: str, model_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(os.path.join(model_dir, model_name), map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

    first_layer_key = next((k for k in state_dict.keys() if "network.0.weight" in k), None)
    use_camera_condition = True
    if first_layer_key:
        use_camera_condition = state_dict[first_layer_key].shape[1] == 1025

    model = DiffCornMLP(input_dim=1024, use_camera_condition=use_camera_condition)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model, device


def predict_with_model(model, bundle: dict, device: torch.device, indices: list[int]):
    features = bundle["features"].to(device)
    cam_heights = torch.tensor(bundle["cam_heights"], dtype=torch.float32).to(device)
    predictions = []
    ground_truths = []
    filenames = []

    with torch.no_grad():
        for idx in indices:
            feat = features[idx : idx + 1]
            cam_h = cam_heights[idx : idx + 1]
            output = model(feat, cam_h)
            pred_height = output[:, 0::4].sum(dim=1).cpu().item()
            predictions.append(pred_height)
            ground_truths.append(float(bundle["heights"][idx]))
            filenames.append(bundle["filenames"][idx])

    return np.array(predictions), np.array(ground_truths), filenames


def calculate_metrics(predictions: np.ndarray, ground_truths: np.ndarray) -> dict:
    errors = predictions - ground_truths
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    median_abs_error = float(np.median(np.abs(errors)))
    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((ground_truths - np.mean(ground_truths)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {
        "MAE": mae,
        "RMSE": rmse,
        "median_abs_error": median_abs_error,
        "R2": r2,
        "n_samples": int(len(predictions)),
    }


def save_report(output_dir: str, report_prefix: str, title: str, metrics: dict, filenames: list[str], ground_truths, predictions):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    errors = predictions - ground_truths
    lines = [
        "=" * 70,
        title,
        "=" * 70,
        f"Samples: {metrics['n_samples']}",
        f"MAE: {metrics['MAE']:.2f} cm",
        f"RMSE: {metrics['RMSE']:.2f} cm",
        f"Median abs error: {metrics['median_abs_error']:.2f} cm",
        f"R2: {metrics['R2']:.4f}",
        "",
        "Detailed results",
        "-" * 70,
    ]
    for fn, gt, pred, err in zip(filenames, ground_truths, predictions, errors):
        lines.append(f"{fn}\ttrue={gt:.1f}\tpred={pred:.1f}\terr={err:+.2f}")
    report_path = Path(output_dir) / f"{report_prefix}_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    results = {
        "metrics": metrics,
        "samples": [
            {"filename": fn, "true_height_cm": float(gt), "pred_height_cm": float(pred), "error_cm": float(err)}
            for fn, gt, pred, err in zip(filenames, ground_truths, predictions, errors)
        ],
    }
    json_path = Path(output_dir) / f"{report_prefix}_results.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(ground_truths, predictions, alpha=0.7, edgecolors="black")
    axes[0].plot([ground_truths.min(), ground_truths.max()], [ground_truths.min(), ground_truths.max()], "r--")
    axes[0].set_xlabel("True Height (cm)")
    axes[0].set_ylabel("Pred Height (cm)")
    axes[0].set_title("Prediction vs Ground Truth")
    axes[0].grid(alpha=0.3)

    axes[1].hist(errors, bins=12, edgecolor="black", alpha=0.75)
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_xlabel("Error (cm)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Error Distribution")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plot_path = Path(output_dir) / f"{report_prefix}_plots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    return report_path, json_path, plot_path


def main() -> None:
    args = parse_args()
    bundle = load_bundle(args.bundle_file)
    test_indices = load_test_indices(args.test_indices_file)
    model, device = load_model(args.model_dir, args.model_name)
    predictions, ground_truths, filenames = predict_with_model(model, bundle, device, test_indices)
    metrics = calculate_metrics(predictions, ground_truths)
    report_path, json_path, plot_path = save_report(
        args.output_dir, args.report_prefix, args.title, metrics, filenames, ground_truths, predictions
    )
    print(json.dumps(metrics, indent=2))
    print(f"Saved report: {report_path}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
