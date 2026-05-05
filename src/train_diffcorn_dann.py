#!/usr/bin/env python3
"""
Train a domain-adversarial DiffCorn model using source hand-bbox features and unlabeled DATA325 images.
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import cycle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from dann_model import DiffCornDANN
from feature_extraction_utils import extract_feature_from_image, load_dinov3_components


DEFAULT_SOURCE_BUNDLE = Path(r"D:\cornTrain\DINOV3\training_bundle_hand_attn.pt")
DEFAULT_TEST_INDICES = Path(r"D:\cornTrain\DINOV3\checkpoints\hand_bbox_test_indices.json")
DEFAULT_DATA325_ROOT = Path(r"C:\Users\Wuhon\OneDrive\桌面\paper\data325")
DEFAULT_CAPTURE_HEIGHT_MAPPING_CSV = Path(
    r"D:\wechatfile\xwechat_files\wxid_9x0cacfdglz622_706c\msg\file\2026-04\largest_dataset_file_capture_height_mapping.csv"
)
DEFAULT_DINO_PATH = Path(r"D:\cornTrain\DINOV3")
DEFAULT_INIT_CHECKPOINT = Path(r"D:\cornTrain\DINOV3\checkpoints\diffcorn_fusion_hand_attn_aug_best.pth")
DEFAULT_OUTPUT_DIR = Path(r"D:\cornTrain\DINOV3\checkpoints")
DEFAULT_CACHE_PATH = Path(r"D:\cornTrain\DINOV3\data325_target_feature_cache_attn.pt")


class SourceFeatureDataset(Dataset):
    def __init__(self, bundle_path: str):
        bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        self.features = bundle["features"].float()
        self.heights = bundle["heights"].float()
        self.cam_heights = torch.tensor(bundle["cam_heights"], dtype=torch.float32)
        self.filenames = list(bundle["filenames"])

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict:
        return {
            "features": self.features[idx],
            "target": self.heights[idx],
            "cam_height": self.cam_heights[idx],
            "filename": self.filenames[idx],
        }


class TargetFeatureDataset(Dataset):
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        item = self.records[idx]
        return {
            "features": item["features"],
            "cam_height": torch.tensor(item["camera_height_cm"], dtype=torch.float32),
            "file_name": item["file_name"],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DiffCorn with DANN.")
    parser.add_argument("--source-bundle", default=str(DEFAULT_SOURCE_BUNDLE))
    parser.add_argument("--test-indices-file", default=str(DEFAULT_TEST_INDICES))
    parser.add_argument("--data325-root", default=str(DEFAULT_DATA325_ROOT))
    parser.add_argument("--capture-height-mapping-csv", default=str(DEFAULT_CAPTURE_HEIGHT_MAPPING_CSV))
    parser.add_argument("--dino-path", default=str(DEFAULT_DINO_PATH))
    parser.add_argument("--init-checkpoint", default=str(DEFAULT_INIT_CHECKPOINT))
    parser.add_argument("--target-cache-path", default=str(DEFAULT_CACHE_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--num-epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_test_indices(path: str) -> list[int]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(data["test_indices"])


def split_source_dataset(dataset: SourceFeatureDataset, test_indices: list[int]) -> tuple[Subset, Subset]:
    test_set = set(test_indices)
    train_indices = [idx for idx in range(len(dataset)) if idx not in test_set]
    return Subset(dataset, train_indices), Subset(dataset, test_indices)


def load_target_image_metadata(mapping_csv: str, root_dir: str) -> list[dict]:
    root = Path(root_dir)
    records = []
    seen = set()
    with open(mapping_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("kind") != "original":
                continue
            file_name = row["file_name"]
            if file_name in seen:
                continue
            seen.add(file_name)
            image_path = root / file_name
            if not image_path.exists():
                continue
            records.append(
                {
                    "file_name": file_name,
                    "image_path": str(image_path),
                    "camera_height_cm": float(row["capture_height_cm"]),
                }
            )
    if not records:
        raise RuntimeError("No DATA325 original images found for DANN target domain.")
    return records


def build_or_load_target_feature_cache(
    cache_path: str,
    records: list[dict],
    dino_path: str,
) -> list[dict]:
    cache_file = Path(cache_path)
    if cache_file.exists():
        cached = torch.load(cache_file, map_location="cpu", weights_only=False)
        if cached.get("count") == len(records):
            return cached["records"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    components = load_dinov3_components(dino_path, device, cache=True)
    out_records = []
    for rec in records:
        image = Image.open(rec["image_path"]).convert("RGB")
        feat = extract_feature_from_image(
            image,
            processor=components["processor"],
            model=components["model"],
            device=device,
            feature_mode="attn_weighted",
            resize_to=(224, 224),
        )
        out_records.append(
            {
                **rec,
                "features": feat.detach().cpu().squeeze(0).float(),
            }
        )
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"count": len(out_records), "records": out_records}, cache_file)
    return out_records


def init_model(init_checkpoint: str) -> DiffCornDANN:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dann = DiffCornDANN(use_camera_condition=True, cam_h_scale=200.0).to(device)
    checkpoint = torch.load(init_checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    dann.regressor.load_state_dict(state_dict, strict=False)
    return dann


def predict_height_from_outputs(outputs: torch.Tensor) -> torch.Tensor:
    return outputs[:, 0::4].sum(dim=1)


def schedule_grl_lambda(epoch: int, num_epochs: int) -> float:
    warmup_epochs = max(1, int(num_epochs * 0.6))
    return min(1.0, epoch / warmup_epochs)


def schedule_domain_weight(epoch: int, num_epochs: int) -> float:
    warmup_epochs = max(1, int(num_epochs * 0.6))
    progress = min(1.0, epoch / warmup_epochs)
    return 0.1 + 0.4 * progress


def evaluate_source_mae(model: DiffCornDANN, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_abs = 0.0
    total_n = 0
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            cam_h = batch["cam_height"].to(device)
            targets = batch["target"].to(device)
            outputs, _, _ = model(features, cam_h, grl_lambda=0.0)
            preds = predict_height_from_outputs(outputs)
            total_abs += torch.abs(preds - targets).sum().item()
            total_n += targets.numel()
    return total_abs / max(total_n, 1)


def plot_training_curves(history: dict, output_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history["epoch"], history["val_mae_cm"], label="Source val MAE", linewidth=2)
    axes[0].plot(history["epoch"], history["train_height_loss"], label="Train height loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Value")
    axes[0].set_title("Source validation / height loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(history["epoch"], history["domain_acc"], label="Domain accuracy", linewidth=2)
    axes[1].plot(history["epoch"], history["domain_loss"], label="Domain loss", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Value")
    axes[1].set_title("Domain branch convergence")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    source_dataset = SourceFeatureDataset(args.source_bundle)
    test_indices = load_test_indices(args.test_indices_file)
    source_train, source_val = split_source_dataset(source_dataset, test_indices)

    target_meta = load_target_image_metadata(args.capture_height_mapping_csv, args.data325_root)
    target_records = build_or_load_target_feature_cache(args.target_cache_path, target_meta, args.dino_path)
    target_dataset = TargetFeatureDataset(target_records)

    per_domain_batch = max(1, args.batch_size // 2)
    src_loader = DataLoader(source_train, batch_size=per_domain_batch, shuffle=True, num_workers=0, drop_last=False)
    tgt_loader = DataLoader(target_dataset, batch_size=per_domain_batch, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(source_val, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = init_model(args.init_checkpoint)
    model.to(device)

    optimizer = optim.Adam(
        [
            {"params": model.adapter.parameters(), "lr": args.learning_rate},
            {"params": model.regressor.parameters(), "lr": args.learning_rate},
            {"params": model.domain_classifier.parameters(), "lr": args.learning_rate},
        ],
        weight_decay=args.weight_decay,
    )

    l1_loss = nn.L1Loss()
    bce_loss = nn.BCELoss()

    history = {"epoch": [], "train_height_loss": [], "domain_loss": [], "domain_acc": [], "val_mae_cm": [], "grl_lambda": [], "domain_weight": []}
    best_val_mae = float("inf")
    best_epoch = 0
    patience_counter = 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "diffcorn_fusion_hand_dann_best.pth"
    final_path = output_dir / "diffcorn_fusion_hand_dann_final.pth"
    curve_path = output_dir / "diffcorn_fusion_hand_dann_domain_curves.png"
    history_path = output_dir / "diffcorn_fusion_hand_dann_history.json"

    for epoch_idx in range(args.num_epochs):
        epoch = epoch_idx + 1
        model.train()
        grl_lambda = schedule_grl_lambda(epoch, args.num_epochs)
        domain_weight = schedule_domain_weight(epoch, args.num_epochs)

        height_losses = []
        domain_losses = []
        domain_correct = 0
        domain_total = 0

        tgt_iter = cycle(tgt_loader)
        for src_batch in src_loader:
            tgt_batch = next(tgt_iter)

            src_features = src_batch["features"].to(device)
            src_cam = src_batch["cam_height"].to(device)
            src_targets = src_batch["target"].to(device)

            tgt_features = tgt_batch["features"].to(device)
            tgt_cam = tgt_batch["cam_height"].to(device)

            optimizer.zero_grad()

            src_outputs, src_domain_probs, _ = model(src_features, src_cam, grl_lambda=grl_lambda)
            _, tgt_domain_probs, _ = model(tgt_features, tgt_cam, grl_lambda=grl_lambda)

            src_height_pred = predict_height_from_outputs(src_outputs)
            src_height_loss = l1_loss(src_height_pred, src_targets)

            src_domain_labels = torch.zeros_like(src_domain_probs)
            tgt_domain_labels = torch.ones_like(tgt_domain_probs)
            src_domain_loss = bce_loss(src_domain_probs, src_domain_labels)
            tgt_domain_loss = bce_loss(tgt_domain_probs, tgt_domain_labels)
            domain_loss = src_domain_loss + tgt_domain_loss

            total_loss = src_height_loss + domain_weight * domain_loss
            total_loss.backward()
            optimizer.step()

            height_losses.append(float(src_height_loss.item()))
            domain_losses.append(float(domain_loss.item()))

            src_preds = (src_domain_probs >= 0.5).float()
            tgt_preds = (tgt_domain_probs >= 0.5).float()
            domain_correct += int((src_preds == src_domain_labels).sum().item() + (tgt_preds == tgt_domain_labels).sum().item())
            domain_total += int(src_domain_labels.numel() + tgt_domain_labels.numel())

        val_mae = evaluate_source_mae(model, val_loader, device)
        mean_height_loss = float(np.mean(height_losses)) if height_losses else 0.0
        mean_domain_loss = float(np.mean(domain_losses)) if domain_losses else 0.0
        domain_acc = float(domain_correct / max(domain_total, 1))

        history["epoch"].append(epoch)
        history["train_height_loss"].append(mean_height_loss)
        history["domain_loss"].append(mean_domain_loss)
        history["domain_acc"].append(domain_acc)
        history["val_mae_cm"].append(val_mae)
        history["grl_lambda"].append(grl_lambda)
        history["domain_weight"].append(domain_weight)

        print(
            f"Epoch {epoch:03d}/{args.num_epochs} | val_mae={val_mae:.3f} | "
            f"height_loss={mean_height_loss:.4f} | domain_loss={mean_domain_loss:.4f} | "
            f"domain_acc={domain_acc:.4f} | grl={grl_lambda:.3f} | lambda_domain={domain_weight:.3f}"
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_mae": best_val_mae,
                    "history": history,
                    "config": vars(args),
                    "source_val_mae": val_mae,
                    "domain_acc": domain_acc,
                },
                best_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience}).")
                break

    torch.save(
        {
            "epoch": history["epoch"][-1] if history["epoch"] else 0,
            "model_state_dict": model.state_dict(),
            "best_val_mae": best_val_mae,
            "best_epoch": best_epoch,
            "history": history,
            "config": vars(args),
        },
        final_path,
    )

    plot_training_curves(history, str(curve_path))
    history_path.write_text(json.dumps({"best_val_mae": best_val_mae, "best_epoch": best_epoch, "history": history}, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({"best_val_mae": best_val_mae, "best_epoch": best_epoch, "best_path": str(best_path), "curve_path": str(curve_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
