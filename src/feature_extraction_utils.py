#!/usr/bin/env python3
"""
Shared DINOv3 feature extraction helpers.

Supports three feature aggregation modes:
- cls
- patch_mean
- attn_weighted
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


FeatureMode = Literal["cls", "patch_mean", "attn_weighted"]
VALID_FEATURE_MODES: tuple[str, ...] = ("cls", "patch_mean", "attn_weighted")

_MODEL_CACHE: dict[tuple[str, str, str], dict[str, object]] = {}


def validate_feature_mode(feature_mode: str) -> str:
    feature_mode = str(feature_mode).strip().lower()
    if feature_mode not in VALID_FEATURE_MODES:
        raise ValueError(f"Unsupported feature_mode={feature_mode!r}. Expected one of {VALID_FEATURE_MODES}.")
    return feature_mode


def load_dinov3_components(model_path: str | Path, device: torch.device, cache: bool = True) -> dict[str, object]:
    resolved = str(Path(model_path).resolve())
    cache_key = (resolved, str(device), "dinov3")
    if cache and cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    processor = AutoImageProcessor.from_pretrained(resolved)
    model = AutoModel.from_pretrained(
        resolved,
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    bundle = {"processor": processor, "model": model, "device": device, "model_path": resolved}
    if cache:
        _MODEL_CACHE[cache_key] = bundle
    return bundle


def aggregate_features(outputs, feature_mode: str) -> torch.Tensor:
    feature_mode = validate_feature_mode(feature_mode)
    if not hasattr(outputs, "last_hidden_state"):
        if hasattr(outputs, "pooler_output"):
            return outputs.pooler_output
        raise ValueError("Model outputs do not expose last_hidden_state or pooler_output.")

    hidden = outputs.last_hidden_state
    if feature_mode == "cls":
        return hidden[:, 0, :]

    patch_tokens = hidden[:, 1:, :]
    if patch_tokens.numel() == 0:
        raise ValueError("No patch tokens available in last_hidden_state.")

    if feature_mode == "patch_mean":
        return patch_tokens.mean(dim=1)

    if not hasattr(outputs, "attentions") or outputs.attentions is None:
        raise ValueError("attn_weighted mode requires attentions in model outputs.")

    attn = outputs.attentions[-1]
    attn_cls = attn[:, :, 0, 1:]
    attn_weights = attn_cls.mean(dim=1)
    denom = attn_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    attn_weights = attn_weights / denom
    return (attn_weights.unsqueeze(-1) * patch_tokens).sum(dim=1)


def extract_feature_from_image(
    image_pil: Image.Image,
    processor,
    model,
    device: torch.device,
    feature_mode: str = "attn_weighted",
    resize_to: tuple[int, int] | None = None,
    return_attention_weights: bool = False,
):
    image = image_pil
    if resize_to is not None:
        image = image.resize(resize_to, Image.Resampling.LANCZOS)

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    need_attn = validate_feature_mode(feature_mode) == "attn_weighted"

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=need_attn)

    features = aggregate_features(outputs, feature_mode)

    if not return_attention_weights:
        return features

    if need_attn:
        attn = outputs.attentions[-1]
        attn_cls = attn[:, :, 0, 1:]
        attn_weights = attn_cls.mean(dim=1)
        attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return features, attn_weights

    return features, None
