#!/usr/bin/env python3
"""
Extract whole-image DINOv3 features from the original dataset.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from feature_extraction_utils import extract_feature_from_image, load_dinov3_components, validate_feature_mode


DATASET_DIR = Path(r"D:\cornTrain\ZeroShotCornPhenotypingAgent\dataset")
ANNOTATIONS_FILE = DATASET_DIR / "plant_annotations.json"
MODEL_PATH = Path(".")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract whole-image DINOv3 features.")
    parser.add_argument("--feature-mode", default="attn_weighted", choices=["cls", "patch_mean", "attn_weighted"])
    parser.add_argument("--output-file", default="training_bundle.pt")
    return parser.parse_args()


def extract_shooting_height(filename: str) -> float | None:
    match = re.match(r"(\d+(?:\.\d+)?)", filename)
    return float(match.group(1)) if match else None


def main() -> None:
    args = parse_args()
    feature_mode = validate_feature_mode(args.feature_mode)
    annotations = json.loads(ANNOTATIONS_FILE.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    components = load_dinov3_components(MODEL_PATH, device)
    processor = components["processor"]
    model = components["model"]

    features_list = []
    heights_list = []
    filenames_list = []

    for filename in tqdm(sorted(annotations.keys()), desc="Extracting whole-image features"):
        image_path = DATASET_DIR / filename
        if not image_path.exists():
            continue
        image = Image.open(image_path).convert("RGB")
        feat = extract_feature_from_image(
            image,
            processor=processor,
            model=model,
            device=device,
            feature_mode=feature_mode,
            resize_to=None,
        )
        features_list.append(feat.cpu().squeeze())
        filenames_list.append(filename)
        heights_list.append(extract_shooting_height(filename))

    bundle = {
        "features": torch.stack(features_list),
        "heights": torch.tensor(heights_list, dtype=torch.float32).unsqueeze(1),
        "filenames": filenames_list,
        "metadata": {
            "num_samples": len(features_list),
            "feature_dim": 1024,
            "model": "DINOv3-ViT-L",
            "feature_mode": feature_mode,
            "source": str(DATASET_DIR),
        },
    }
    torch.save(bundle, args.output_file)
    print(f"Saved bundle: {args.output_file}")


if __name__ == "__main__":
    main()
