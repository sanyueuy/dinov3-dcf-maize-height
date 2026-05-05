#!/usr/bin/env python3
"""CEA revision diagnostics for the DINOv3-DCF maize height manuscript.

The outputs from this script are intentionally lightweight and deterministic:
they derive from the existing DATA325 annotations, model predictions, and real
image crops. No generated imagery is used for evidence panels.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"D:\cornTrain\DINOV3")
OUT = ROOT / "experiments" / "cea_revision"
MASK_DIR = OUT / "mask_examples"
DATA325_IMAGE_DIR = ROOT / "images"
SEED_RETRAIN_DIR = OUT / "seed_retraining"
SEED_EVAL_JSON = SEED_RETRAIN_DIR / "data325_eval" / "data325_zero_shot_comparison_seed_retraining.json"

MODEL_JSONS = [
    ("cls", ROOT / "data325_zero_shot" / "data325_zero_shot_comparison.json"),
    ("corrected_camheight", ROOT / "data325_zero_shot_corrected_camheight" / "data325_zero_shot_comparison_corrected_camheight.json"),
    ("attn_aug", ROOT / "data325_zero_shot_attn_aug" / "data325_zero_shot_comparison_attn_aug.json"),
    ("featurealign", ROOT / "data325_zero_shot_attn_aug_featurealign" / "data325_zero_shot_comparison_attn_aug_featurealign.json"),
    ("dann", ROOT / "data325_zero_shot_dann" / "data325_zero_shot_comparison_dann.json"),
    ("attn_aug_tta8", ROOT / "data325_zero_shot_attn_aug_tta8" / "data325_zero_shot_comparison_attn_aug_tta8.json"),
]

TARGET_LABEL = "attn_aug_tta8"
BOOTSTRAP_SEED = 20260505
BOOTSTRAP_N = 5000


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def bbox_xyxy(bbox: list[float], image_size: tuple[int, int] | None = None) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if image_size is not None:
        w, h = image_size
        if x2 <= x1 or y2 <= y1:
            x2 = x1 + max(1.0, x2)
            y2 = y1 + max(1.0, y2)
        x1 = min(max(x1, 0.0), float(w - 1))
        x2 = min(max(x2, x1 + 1.0), float(w))
        y1 = min(max(y1, 0.0), float(h - 1))
        y2 = min(max(y2, y1 + 1.0), float(h))
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def load_model_results() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for fallback_label, path in MODEL_JSONS:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for model in data.get("model_results", []):
            label = model.get("label") or fallback_label
            if label in out:
                label = fallback_label
            rows = model.get("box_results", [])
            for row in rows:
                row.setdefault("model_label", label)
            out[label] = {
                "path": str(path),
                "feature_mode": model.get("feature_mode", ""),
                "summary": model.get("summary", {}),
                "rows": rows,
            }
    return out


def keyed(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["file_name"], row["box_id"]): row for row in rows}


def metric_values(rows: list[dict[str, Any]]) -> dict[str, float]:
    true = np.array([float(row["true_height_cm"]) for row in rows], dtype=float)
    pred = np.array([float(row["pred_height_cm"]) for row in rows], dtype=float)
    ae = np.abs(pred - true)
    with np.errstate(divide="ignore", invalid="ignore"):
        ape = np.where(true > 0, ae / true * 100.0, np.nan)
    return {
        "n": int(len(rows)),
        "mae_cm": float(np.nanmean(ae)),
        "rmse_cm": float(np.sqrt(np.nanmean((pred - true) ** 2))),
        "median_abs_error_cm": float(np.nanmedian(ae)),
        "mape_percent": float(np.nanmean(ape)),
    }


def bootstrap_metrics(rows: list[dict[str, Any]], rng: np.random.Generator) -> dict[str, Any]:
    n = len(rows)
    true = np.array([float(row["true_height_cm"]) for row in rows], dtype=float)
    pred = np.array([float(row["pred_height_cm"]) for row in rows], dtype=float)
    boot = {"mae_cm": [], "rmse_cm": [], "mape_percent": []}
    for _ in range(BOOTSTRAP_N):
        idx = rng.integers(0, n, n)
        err = pred[idx] - true[idx]
        ae = np.abs(err)
        with np.errstate(divide="ignore", invalid="ignore"):
            ape = np.where(true[idx] > 0, ae / true[idx] * 100.0, np.nan)
        boot["mae_cm"].append(float(np.nanmean(ae)))
        boot["rmse_cm"].append(float(np.sqrt(np.nanmean(err**2))))
        boot["mape_percent"].append(float(np.nanmean(ape)))
    point = metric_values(rows)
    ci: dict[str, Any] = {"n_bootstrap": BOOTSTRAP_N, "point": point}
    for name, values in boot.items():
        arr = np.array(values, dtype=float)
        ci[name] = {
            "mean": float(np.mean(arr)),
            "ci95_low": float(np.quantile(arr, 0.025)),
            "ci95_high": float(np.quantile(arr, 0.975)),
        }
    return ci


def sign_test_p(values: np.ndarray) -> float:
    positives = int(np.sum(values > 0))
    negatives = int(np.sum(values < 0))
    n = positives + negatives
    if n == 0:
        return 1.0
    k = min(positives, negatives)
    # two-sided exact binomial with p=0.5
    prob = 0.0
    for i in range(0, k + 1):
        prob += math.comb(n, i) * (0.5**n)
    return float(min(1.0, 2.0 * prob))


def paired_tests(models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    target = keyed(models[TARGET_LABEL]["rows"])
    rng = np.random.default_rng(BOOTSTRAP_SEED + 17)
    results: dict[str, Any] = {}
    for label, model in models.items():
        if label == TARGET_LABEL:
            continue
        other = keyed(model["rows"])
        common = sorted(set(target) & set(other))
        if not common:
            continue
        target_ae = np.array([float(target[k]["abs_error_cm"]) for k in common], dtype=float)
        other_ae = np.array([float(other[k]["abs_error_cm"]) for k in common], dtype=float)
        delta = other_ae - target_ae
        samples = []
        for _ in range(BOOTSTRAP_N):
            idx = rng.integers(0, len(delta), len(delta))
            samples.append(float(np.mean(delta[idx])))
        arr = np.array(samples)
        results[label] = {
            "n": len(common),
            "comparison": f"{label} minus {TARGET_LABEL}; positive means {TARGET_LABEL} lower absolute error",
            "mean_abs_error_delta_cm": float(np.mean(delta)),
            "median_abs_error_delta_cm": float(np.median(delta)),
            "ci95_low": float(np.quantile(arr, 0.025)),
            "ci95_high": float(np.quantile(arr, 0.975)),
            "sign_test_p": sign_test_p(delta),
            "target_better_fraction": float(np.mean(delta > 0)),
        }
    return results


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_vals = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_vals[j] == sorted_vals[i]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def corr(x: list[float], y: list[float]) -> dict[str, float]:
    xa = np.array(x, dtype=float)
    ya = np.array(y, dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    if int(mask.sum()) < 3:
        return {"pearson_r": float("nan"), "spearman_r": float("nan")}
    xa = xa[mask]
    ya = ya[mask]
    pearson = float(np.corrcoef(xa, ya)[0, 1])
    spearman = float(np.corrcoef(rankdata(xa), rankdata(ya))[0, 1])
    return {"pearson_r": pearson, "spearman_r": spearman}


def green_mask(arr: np.ndarray) -> np.ndarray:
    rgb = arr.astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    exg = 2.0 * g - r - b
    denom = np.maximum(np.max(rgb, axis=2) - np.min(rgb, axis=2), 1.0)
    saturation_like = denom / np.maximum(np.max(rgb, axis=2), 1.0)
    dynamic = exg > np.quantile(exg, 0.60)
    chroma = (g > r * 1.04) & (g > b * 1.04) & (saturation_like > 0.10)
    bright = np.mean(rgb, axis=2) > 20.0
    mask = (dynamic | chroma) & bright
    return mask


def draw_mask_preview(row: dict[str, Any], mask: np.ndarray, roi: Image.Image, out_path: Path) -> None:
    w, h = roi.size
    max_w = 980
    scale = min(1.0, max_w / (w * 3 + 40))
    panel_w = max(180, int(w * scale))
    panel_h = max(180, int(h * scale))
    roi_small = roi.resize((panel_w, panel_h), Image.Resampling.LANCZOS)
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L").resize((panel_w, panel_h), Image.Resampling.NEAREST)
    overlay = roi_small.convert("RGBA")
    green = Image.new("RGBA", overlay.size, (40, 180, 110, 105))
    overlay = Image.composite(green, overlay, mask_img).convert("RGB")
    canvas = Image.new("RGB", (panel_w * 3 + 40, panel_h + 80), "white")
    draw = ImageDraw.Draw(canvas)
    x = 10
    for title, img in [("ROI crop", roi_small), ("Plant mask", mask_img.convert("RGB")), ("Overlay", overlay)]:
        canvas.paste(img, (x, 36))
        draw.text((x, 12), title, fill=(32, 33, 36))
        x += panel_w + 10
    note = f"true={row['true_height_cm']:.1f} cm, pred={row['pred_height_cm']:.1f} cm, AE={row['abs_error_cm']:.1f} cm"
    draw.text((10, panel_h + 48), note, fill=(90, 90, 90))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)


def roi_quality_metrics(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    previews: list[tuple[float, dict[str, Any], np.ndarray, Image.Image]] = []
    for row in rows:
        img_path = DATA325_IMAGE_DIR / row["file_name"]
        if not img_path.exists():
            continue
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            iw, ih = img.size
            x1, y1, x2, y2 = bbox_xyxy(row["bbox"], img.size)
            roi = img.crop((x1, y1, x2, y2))
        arr = np.array(roi)
        mask = green_mask(arr)
        fg = float(mask.mean())
        bg = 1.0 - fg
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        rgb = arr.astype(np.float32)
        brightness = float(np.mean(rgb))
        edge_touch = int(x1 <= 8 or y1 <= 8 or x2 >= iw - 8 or y2 >= ih - 8)
        row_metrics = {
            "image_id": row["image_id"],
            "file_name": row["file_name"],
            "box_id": row["box_id"],
            "true_height_cm": float(row["true_height_cm"]),
            "pred_height_cm": float(row["pred_height_cm"]),
            "abs_error_cm": float(row["abs_error_cm"]),
            "pred_std_cm": float(row.get("pred_std_cm", 0.0)),
            "camera_height_cm": float(row["camera_height_cm"]),
            "bbox_width_px": bw,
            "bbox_height_px": bh,
            "bbox_area_fraction": float((bw * bh) / (iw * ih)),
            "bbox_aspect_ratio": float(bw / bh),
            "bbox_fill_ratio": fg,
            "foreground_fraction": fg,
            "background_fraction": bg,
            "mean_roi_brightness": brightness,
            "touches_image_edge": edge_touch,
        }
        metrics.append(row_metrics)
        previews.append((float(row["abs_error_cm"]), row, mask, roi))

    corr_fields = [
        "foreground_fraction",
        "background_fraction",
        "bbox_area_fraction",
        "bbox_aspect_ratio",
        "pred_std_cm",
        "mean_roi_brightness",
    ]
    summary: dict[str, Any] = {"n": len(metrics), "correlations_with_abs_error": {}}
    for field in corr_fields:
        summary["correlations_with_abs_error"][field] = corr(
            [float(m[field]) for m in metrics], [float(m["abs_error_cm"]) for m in metrics]
        )
    if metrics:
        summary["foreground_fraction_mean"] = float(np.mean([m["foreground_fraction"] for m in metrics]))
        summary["foreground_fraction_median"] = float(np.median([m["foreground_fraction"] for m in metrics]))
        summary["background_fraction_mean"] = float(np.mean([m["background_fraction"] for m in metrics]))

    previews_sorted = sorted(previews, key=lambda x: x[0], reverse=True)
    for idx, (_, row, mask, roi) in enumerate(previews_sorted[:5], start=1):
        draw_mask_preview(row, mask, roi, MASK_DIR / f"high_error_mask_{idx:02d}.jpg")
    for idx, (_, row, mask, roi) in enumerate(previews_sorted[-5:], start=1):
        draw_mask_preview(row, mask, roi, MASK_DIR / f"low_error_mask_{idx:02d}.jpg")
    return metrics, summary


def ridge_loio_baseline(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if not metrics:
        return {}
    images = sorted({m["file_name"] for m in metrics})
    features = [
        "bbox_width_px",
        "bbox_height_px",
        "bbox_area_fraction",
        "bbox_aspect_ratio",
        "camera_height_cm",
        "foreground_fraction",
        "background_fraction",
    ]
    rows_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metrics:
        rows_by_image[row["file_name"]].append(row)

    preds: list[dict[str, Any]] = []
    alpha = 1.0
    for holdout in images:
        train = [r for img, rs in rows_by_image.items() if img != holdout for r in rs]
        test = rows_by_image[holdout]
        x_train = np.array([[float(r[f]) for f in features] for r in train], dtype=float)
        y_train = np.array([float(r["true_height_cm"]) for r in train], dtype=float)
        x_test = np.array([[float(r[f]) for f in features] for r in test], dtype=float)
        mean = x_train.mean(axis=0)
        std = x_train.std(axis=0)
        std[std == 0] = 1.0
        x_train_z = (x_train - mean) / std
        x_test_z = (x_test - mean) / std
        x_train_aug = np.column_stack([np.ones(len(x_train_z)), x_train_z])
        x_test_aug = np.column_stack([np.ones(len(x_test_z)), x_test_z])
        penalty = np.eye(x_train_aug.shape[1]) * alpha
        penalty[0, 0] = 0.0
        beta = np.linalg.pinv(x_train_aug.T @ x_train_aug + penalty) @ x_train_aug.T @ y_train
        y_pred = x_test_aug @ beta
        for r, pred in zip(test, y_pred):
            preds.append(
                {
                    "image_id": r["image_id"],
                    "file_name": r["file_name"],
                    "box_id": r["box_id"],
                    "true_height_cm": float(r["true_height_cm"]),
                    "pred_height_cm": float(pred),
                    "abs_error_cm": abs(float(pred) - float(r["true_height_cm"])),
                }
            )
    summary = metric_values(preds)
    summary.update(
        {
            "baseline_type": "leave-one-image-out ridge regression on DATA325 morphometric and mask features",
            "interpretation": "diagnostic target-label baseline; not a zero-shot external comparator",
            "features": features,
        }
    )
    return {"summary": summary, "predictions": preds}


def uncertainty_diagnostic(rows: list[dict[str, Any]], roi_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    m_by_key = {(m["file_name"], m["box_id"]): m for m in roi_metrics}
    merged = []
    for row in rows:
        key = (row["file_name"], row["box_id"])
        m = m_by_key.get(key, {})
        true_h = float(row["true_height_cm"])
        if true_h < 80:
            height_bin = "<80"
        elif true_h < 120:
            height_bin = "80-120"
        elif true_h < 160:
            height_bin = "120-160"
        else:
            height_bin = ">=160"
        merged.append(
            {
                "file_name": row["file_name"],
                "box_id": row["box_id"],
                "true_height_cm": true_h,
                "abs_error_cm": float(row["abs_error_cm"]),
                "pred_std_cm": float(row.get("pred_std_cm", 0.0)),
                "background_fraction": float(m.get("background_fraction", float("nan"))),
                "height_bin": height_bin,
            }
        )
    by_bin: dict[str, dict[str, float]] = {}
    for bin_name in ["<80", "80-120", "120-160", ">=160"]:
        sub = [m for m in merged if m["height_bin"] == bin_name]
        if sub:
            by_bin[bin_name] = {
                "n": len(sub),
                "mean_pred_std_cm": float(np.mean([m["pred_std_cm"] for m in sub])),
                "mean_abs_error_cm": float(np.mean([m["abs_error_cm"] for m in sub])),
            }
    return {
        "overall": {
            "pred_std_vs_abs_error": corr([m["pred_std_cm"] for m in merged], [m["abs_error_cm"] for m in merged]),
            "pred_std_vs_background_fraction": corr(
                [m["pred_std_cm"] for m in merged], [m["background_fraction"] for m in merged]
            ),
        },
        "height_bins": by_bin,
        "top_uncertainty_cases": sorted(merged, key=lambda m: m["pred_std_cm"], reverse=True)[:10],
    }


def error_taxonomy(rows: list[dict[str, Any]], roi_metrics: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    m_by_key = {(m["file_name"], m["box_id"]): m for m in roi_metrics}
    pred_std_values = np.array([float(row.get("pred_std_cm", 0.0)) for row in rows], dtype=float)
    std_q75 = float(np.quantile(pred_std_values, 0.75))
    aspects = np.array([float(m["bbox_aspect_ratio"]) for m in roi_metrics], dtype=float)
    low_aspect, high_aspect = float(np.quantile(aspects, 0.10)), float(np.quantile(aspects, 0.90))
    rows_out: list[dict[str, Any]] = []
    for row in rows:
        key = (row["file_name"], row["box_id"])
        m = m_by_key.get(key, {})
        true_h = float(row["true_height_cm"])
        bg = float(m.get("background_fraction", 0.0))
        aspect = float(m.get("bbox_aspect_ratio", 0.0))
        pred_std = float(row.get("pred_std_cm", 0.0))
        if true_h < 80:
            category = "early-stage sparse structure"
        elif bg > 0.72:
            category = "background clutter"
        elif int(m.get("touches_image_edge", 0)):
            category = "partial plant or edge crop"
        elif aspect < low_aspect or aspect > high_aspect:
            category = "bbox ambiguity"
        elif pred_std > std_q75:
            category = "unstable TTA prediction"
        else:
            category = "residual cross-domain shift"
        rows_out.append(
            {
                "image_id": row["image_id"],
                "file_name": row["file_name"],
                "box_id": row["box_id"],
                "true_height_cm": true_h,
                "pred_height_cm": float(row["pred_height_cm"]),
                "abs_error_cm": float(row["abs_error_cm"]),
                "pred_std_cm": pred_std,
                "background_fraction": bg,
                "bbox_aspect_ratio": aspect,
                "primary_error_category": category,
            }
        )
    summary: dict[str, Any] = {}
    for category, count in Counter(r["primary_error_category"] for r in rows_out).items():
        sub = [r for r in rows_out if r["primary_error_category"] == category]
        summary[category] = {
            "n": count,
            "mean_abs_error_cm": float(np.mean([r["abs_error_cm"] for r in sub])),
            "median_abs_error_cm": float(np.median([r["abs_error_cm"] for r in sub])),
        }
    return rows_out, summary


def height_bin_bootstrap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bins = [("<80", -float("inf"), 80), ("80-120", 80, 120), ("120-160", 120, 160), (">=160", 160, float("inf"))]
    rng = np.random.default_rng(BOOTSTRAP_SEED + 23)
    out: dict[str, Any] = {}
    for name, lo, hi in bins:
        sub = [r for r in rows if lo <= float(r["true_height_cm"]) < hi]
        if not sub:
            continue
        ci = bootstrap_metrics(sub, rng)
        out[name] = ci
    return out


def resampling_robustness(models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """A deterministic robustness proxy when independent seed retraining is unavailable.

    It is reported separately from true re-training so the manuscript does not
    overclaim independent random-seed evidence.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED + 41)
    methods = ["base", "cls", "patch_mean", "attn", "attn_aug", TARGET_LABEL]
    out: dict[str, Any] = {}
    for label in methods:
        if label not in models:
            continue
        rows = models[label]["rows"]
        n = len(rows)
        mae = []
        for _ in range(1000):
            idx = rng.integers(0, n, n)
            mae.append(float(np.mean([float(rows[i]["abs_error_cm"]) for i in idx])))
        arr = np.array(mae, dtype=float)
        out[label] = {
            "n": n,
            "point_mae_cm": metric_values(rows)["mae_cm"],
            "resampled_mae_mean_cm": float(np.mean(arr)),
            "resampled_mae_sd_cm": float(np.std(arr, ddof=1)),
            "resampled_mae_ci95_low": float(np.quantile(arr, 0.025)),
            "resampled_mae_ci95_high": float(np.quantile(arr, 0.975)),
            "note": "bootstrap resampling of existing checkpoint predictions; not independent random-seed retraining",
        }
    return out


def parse_training_report(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower().replace(" ", "_")
        value = value.strip()
        if key in {"initial_train_mae", "initial_test_mae", "final_train_mae", "final_test_mae", "best_test_mae"}:
            try:
                out[key + "_cm"] = float(value.split()[0])
            except (ValueError, IndexError):
                pass
    return out


def seed_retraining_summary() -> dict[str, Any]:
    if not SEED_EVAL_JSON.exists():
        return {
            "available": False,
            "note": "Seed retraining outputs were not found; bootstrap robustness is available instead.",
        }
    data = json.loads(SEED_EVAL_JSON.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for result in data.get("model_results", []):
        label = result["label"]
        method, seed_text = label.rsplit("_s", 1)
        seed = int(seed_text)
        report = SEED_RETRAIN_DIR / f"seed_{method}_{seed}_report.txt"
        training = parse_training_report(report)
        summary = result.get("summary", {})
        rows.append(
            {
                "method": method,
                "seed": seed,
                "feature_mode": result.get("feature_mode", ""),
                "source_best_test_mae_cm": training.get("best_test_mae_cm", float("nan")),
                "source_final_test_mae_cm": training.get("final_test_mae_cm", float("nan")),
                "data325_mae_cm": float(summary.get("mae_cm", float("nan"))),
                "data325_rmse_cm": float(summary.get("rmse_cm", float("nan"))),
                "data325_median_abs_error_cm": float(summary.get("median_abs_error_cm", float("nan"))),
                "checkpoint": Path(result.get("checkpoint_path", "")).name,
            }
        )
    aggregate: dict[str, Any] = {}
    for method in sorted({row["method"] for row in rows}):
        sub = [row for row in rows if row["method"] == method]
        aggregate[method] = {
            "n_seeds": len(sub),
            "source_best_test_mae_mean_cm": float(np.nanmean([row["source_best_test_mae_cm"] for row in sub])),
            "source_best_test_mae_sd_cm": float(np.nanstd([row["source_best_test_mae_cm"] for row in sub], ddof=1)),
            "data325_mae_mean_cm": float(np.nanmean([row["data325_mae_cm"] for row in sub])),
            "data325_mae_sd_cm": float(np.nanstd([row["data325_mae_cm"] for row in sub], ddof=1)),
            "data325_rmse_mean_cm": float(np.nanmean([row["data325_rmse_cm"] for row in sub])),
            "data325_median_abs_error_mean_cm": float(
                np.nanmean([row["data325_median_abs_error_cm"] for row in sub])
            ),
        }
    write_csv(
        OUT / "seed_retraining_summary.csv",
        rows,
        [
            "method",
            "seed",
            "feature_mode",
            "source_best_test_mae_cm",
            "source_final_test_mae_cm",
            "data325_mae_cm",
            "data325_rmse_cm",
            "data325_median_abs_error_cm",
            "checkpoint",
        ],
    )
    return {
        "available": True,
        "seeds": sorted({row["seed"] for row in rows}),
        "methods": sorted(aggregate.keys()),
        "evaluation_tta_count": data.get("model_results", [{}])[0].get("tta", {}).get("count", 1),
        "rows": rows,
        "aggregate": aggregate,
        "note": "Independent DCF-head retraining for 3 random seeds per feature mode; DATA325 evaluation used tta_count=1.",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    models = load_model_results()
    if TARGET_LABEL not in models:
        raise RuntimeError(f"Missing target model '{TARGET_LABEL}'")
    target_rows = models[TARGET_LABEL]["rows"]
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    bootstrap = {label: bootstrap_metrics(model["rows"], rng) for label, model in models.items()}
    paired = paired_tests(models)
    roi_metrics_rows, roi_summary = roi_quality_metrics(target_rows)
    morphometric = ridge_loio_baseline(roi_metrics_rows)
    uncertainty = uncertainty_diagnostic(target_rows, roi_metrics_rows)
    taxonomy_rows, taxonomy_summary = error_taxonomy(target_rows, roi_metrics_rows)
    height_bins = height_bin_bootstrap(target_rows)
    robustness = resampling_robustness(models)
    seed_summary = seed_retraining_summary()

    model_summary = {
        label: {
            "feature_mode": model["feature_mode"],
            "path": model["path"],
            **metric_values(model["rows"]),
        }
        for label, model in models.items()
    }
    summary = {
        "generated_by": "experiments/cea_revision/run_cea_revision_experiments.py",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_n": BOOTSTRAP_N,
        "model_summary": model_summary,
        "best_model_label": TARGET_LABEL,
        "best_model_metrics": metric_values(target_rows),
        "roi_quality_summary": roi_summary,
        "morphometric_baseline_summary": morphometric.get("summary", {}),
        "uncertainty_summary": uncertainty.get("overall", {}),
        "error_taxonomy_summary": taxonomy_summary,
        "height_bin_bootstrap": height_bins,
        "seed_retraining_summary": seed_summary,
    }

    write_json(OUT / "bootstrap_ci.json", bootstrap)
    write_json(OUT / "paired_tests.json", paired)
    write_csv(OUT / "roi_quality_metrics.csv", roi_metrics_rows)
    write_json(OUT / "roi_quality_summary.json", roi_summary)
    write_json(OUT / "morphometric_baseline.json", morphometric)
    write_json(OUT / "uncertainty_diagnostic.json", uncertainty)
    write_csv(OUT / "error_taxonomy.csv", taxonomy_rows)
    write_json(OUT / "error_taxonomy_summary.json", taxonomy_summary)
    write_json(OUT / "height_bin_bootstrap.json", height_bins)
    write_json(OUT / "resampling_robustness.json", robustness)
    write_json(OUT / "seed_retraining_summary.json", seed_summary)
    write_json(OUT / "cea_revision_summary.json", summary)

    print(f"Wrote CEA revision diagnostics to {OUT}")
    print(json.dumps(summary["best_model_metrics"], indent=2))


if __name__ == "__main__":
    main()
