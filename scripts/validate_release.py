#!/usr/bin/env python3
import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
images = sorted((root / "data" / "DATA325" / "images").glob("*.jpg"))
ann = list(csv.DictReader((root / "data" / "DATA325" / "annotations" / "data325_annotations.csv").open(encoding="utf-8")))
pred = list(csv.DictReader((root / "data" / "DATA325" / "predictions" / "data325_predictions_attn_aug_tta8.csv").open(encoding="utf-8")))
info = json.loads((root / "metadata" / "release_info.json").read_text(encoding="utf-8"))

assert len(images) == info["raw_image_count"], (len(images), info["raw_image_count"])
assert len(ann) == info["box_count"], (len(ann), info["box_count"])
assert len(pred) == info["box_count"], (len(pred), info["box_count"])
missing = [row["file_name"] for row in ann if not (root / "data" / "DATA325" / "images" / row["file_name"]).exists()]
assert not missing, missing[:5]
assert (root / "results" / "cea_revision" / "roi_quality_metrics.csv").exists()
assert (root / "results" / "reproducibility_json" / "cea_revision_summary.json").exists()
print("OK")
print(f"images={len(images)} boxes={len(ann)} predictions={len(pred)}")
