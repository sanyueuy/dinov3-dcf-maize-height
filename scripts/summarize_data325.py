#!/usr/bin/env python3
import csv
import math
from pathlib import Path

root = Path(__file__).resolve().parents[1]
csv_path = root / "data" / "DATA325" / "predictions" / "data325_predictions_attn_aug_tta8.csv"
rows = []
with csv_path.open("r", encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        row["true_height_cm"] = float(row["true_height_cm"])
        row["pred_height_cm"] = float(row["pred_height_cm"])
        row["abs_error_cm"] = abs(row["pred_height_cm"] - row["true_height_cm"])
        rows.append(row)

n = len(rows)
mae = sum(r["abs_error_cm"] for r in rows) / n
rmse = math.sqrt(sum((r["pred_height_cm"] - r["true_height_cm"]) ** 2 for r in rows) / n)
med = sorted(r["abs_error_cm"] for r in rows)[n // 2] if n % 2 else (
    sorted(r["abs_error_cm"] for r in rows)[n // 2 - 1] + sorted(r["abs_error_cm"] for r in rows)[n // 2]
) / 2
print(f"DATA325 boxes: {n}")
print(f"MAE_cm: {mae:.6f}")
print(f"RMSE_cm: {rmse:.6f}")
print(f"Median_abs_error_cm: {med:.6f}")
