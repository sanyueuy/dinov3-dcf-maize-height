#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import textwrap
import zipfile
from datetime import date
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(r"D:\cornTrain\DINOV3")
OUT = ROOT / "open_release"
DIST = ROOT / "open_release_dist"
VERSION = "v0.1.0"
SLUG = "dinov3_dcf_maize_height_open_release_v0_1_0"
AUTHOR = "Hong Wu; Jian Chen"
AFFILIATION = "China Agricultural University"
CONTACT_EMAIL = "jchen@cau.edu.cn"
TITLE = "A reproducible external-greenhouse benchmark for diagnosing DINOv3 feature transfer in maize height estimation"
REPOSITORY_URL = "https://github.com/sanyueuy/dinov3-dcf-maize-height"
CAPTURE_TOOL_URL = "https://github.com/sanyueuy/corn-capture"

DATA325_IMAGE_DIR = ROOT / "images"
FINAL_JSON = ROOT / "data325_zero_shot_attn_aug_tta8" / "data325_zero_shot_comparison_attn_aug_tta8.json"
CEA_EXP_DIR = ROOT / "experiments" / "cea_revision"

REPRO_JSONS = [
    ROOT / "data325_zero_shot_attn_aug_tta8" / "data325_zero_shot_comparison_attn_aug_tta8.json",
    ROOT / "data325_zero_shot_corrected_camheight" / "data325_zero_shot_comparison_corrected_camheight.json",
    ROOT / "data325_zero_shot_attn_aug" / "data325_zero_shot_comparison_attn_aug.json",
    ROOT / "data325_zero_shot_attn_aug_featurealign" / "data325_zero_shot_comparison_attn_aug_featurealign.json",
    ROOT / "data325_zero_shot_attn_aug_featurealign" / "data325_feature_stats_attn_aug_featurealign.json",
    ROOT / "data325_zero_shot_dann" / "data325_zero_shot_comparison_dann.json",
    ROOT / "data325_zero_shot_dann" / "source_eval_dann.json",
    ROOT / "bbox_geometry_prior_phaseA.json",
    ROOT / "attention_geometry_prior_phaseA.json",
    ROOT / "tsne_source_vs_data325.json",
    CEA_EXP_DIR / "bootstrap_ci.json",
    CEA_EXP_DIR / "paired_tests.json",
    CEA_EXP_DIR / "roi_quality_summary.json",
    CEA_EXP_DIR / "morphometric_baseline.json",
    CEA_EXP_DIR / "source_morphometric_baseline.json",
    CEA_EXP_DIR / "uncertainty_diagnostic.json",
    CEA_EXP_DIR / "error_taxonomy_summary.json",
    CEA_EXP_DIR / "height_bin_bootstrap.json",
    CEA_EXP_DIR / "resampling_robustness.json",
    CEA_EXP_DIR / "seed_retraining_summary.json",
    CEA_EXP_DIR / "seed_retraining" / "data325_eval" / "data325_zero_shot_comparison_seed_retraining.json",
    CEA_EXP_DIR / "cea_revision_summary.json",
]

SOURCE_FEATURE_BUNDLES = [
    ROOT / "training_bundle_hand.pt",
    ROOT / "training_bundle_hand_cls.pt",
    ROOT / "training_bundle_hand_patch_mean.pt",
    ROOT / "training_bundle_hand_attn.pt",
    ROOT / "training_bundle_hand_attn_aug.pt",
    ROOT / "data325_target_feature_cache_attn.pt",
]

CHECKPOINTS = [
    ROOT / "checkpoints" / "diffcorn_fusion_hand_best.pth",
    ROOT / "checkpoints" / "diffcorn_fusion_hand_cls_best.pth",
    ROOT / "checkpoints" / "diffcorn_fusion_hand_patch_mean_best.pth",
    ROOT / "checkpoints" / "diffcorn_fusion_hand_attn_best.pth",
    ROOT / "checkpoints" / "diffcorn_fusion_hand_attn_aug_best.pth",
    ROOT / "checkpoints" / "diffcorn_fusion_hand_dann_best.pth",
]

CODE_FILES = [
    "model.py",
    "dann_model.py",
    "feature_extraction_utils.py",
    "extract_real_features.py",
    "train_diffcorn_fusion_hand.py",
    "train_diffcorn_dann.py",
    "evaluate_data325_zero_shot.py",
    "evaluate_data325_feature_alignment.py",
    "evaluate_dann_checkpoint.py",
    "evaluate_attention_geometry_prior.py",
    "run_hand_bbox_benchmark.py",
    "evaluate_on_hand_bbox_proper.py",
    "build_hand_bbox_augmented_bundle.py",
    "experiments/cea_revision/run_cea_revision_experiments.py",
    "build_cea_submission.py",
    "build_open_release.py",
]


def must_be_inside_root(path: Path) -> Path:
    resolved = path.resolve()
    root = ROOT.resolve()
    if resolved == root or root not in resolved.parents:
        raise RuntimeError(f"Refusing to operate outside project root: {resolved}")
    return resolved


def reset_dir(path: Path) -> None:
    resolved = must_be_inside_root(path)
    if resolved.exists():
        if resolved == OUT.resolve() and (resolved / ".git").exists():
            for child in resolved.iterdir():
                if child.name == ".git":
                    continue
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        else:
            shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8", newline="\n")


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def sanitize_checkpoint_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitize_checkpoint_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_checkpoint_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_checkpoint_value(item) for item in value)
    if isinstance(value, str):
        return sanitize_json_value(value)
    return value


def copy_sanitized_checkpoint(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        import torch

        checkpoint = torch.load(src, map_location="cpu", weights_only=False)
        torch.save(sanitize_checkpoint_value(checkpoint), dst)
    except Exception:
        copy_file(src, dst)


def sanitize_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key == "image_path" and isinstance(item, str):
                out[key] = "data/DATA325/images/" + Path(item).name
            elif key == "checkpoint_path" and isinstance(item, str):
                out[key] = "checkpoints/" + Path(item).name
            elif key == "annotations_file" and isinstance(item, str):
                out[key] = "data/DATA325/annotations/data325_annotations.csv"
            elif key == "capture_height_mapping_csv" and isinstance(item, str):
                out[key] = "data/DATA325/annotations/camera_height_mapping.csv"
            elif key == "output_image" and isinstance(item, str):
                out[key] = "figures/" + Path(item).name
            else:
                out[key] = sanitize_json_value(item)
        return out
    if isinstance(value, list):
        return [sanitize_json_value(item) for item in value]
    if isinstance(value, str):
        replacements = [
            (r"C:\Users\Wuhon\OneDrive\桌面\paper\data325", "data/DATA325/images"),
            (r"D:\OneDrive\桌面\paper\data325", "data/DATA325/images"),
            (r"D:\wechatfile", "external_metadata"),
            (r"D:\cornTrain\DINOV3\checkpoints", "checkpoints"),
            (r"D:\cornTrain\DINOV3", "."),
            (r"D:\cornTrain\dcf-bbox-eval-tool\data", "data/DATA325/annotations"),
            (r"C:\Users\Wuhon", "."),
        ]
        out = value
        for old, new in replacements:
            out = out.replace(old, new)
        return out.replace("\\", "/")
    return value


def load_final_rows() -> list[dict[str, Any]]:
    data = json.loads(FINAL_JSON.read_text(encoding="utf-8"))
    return data["model_results"][0]["box_results"]


def write_data_tables() -> dict[str, Any]:
    rows = load_final_rows()
    annotation_dir = OUT / "data" / "DATA325" / "annotations"
    prediction_dir = OUT / "data" / "DATA325" / "predictions"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    ann_fields = [
        "image_id",
        "file_name",
        "box_id",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "true_height_cm",
        "camera_height_cm",
    ]
    with (annotation_dir / "data325_annotations.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ann_fields)
        writer.writeheader()
        for row in rows:
            x1, y1, x2, y2 = row["bbox"]
            writer.writerow(
                {
                    "image_id": row["image_id"],
                    "file_name": row["file_name"],
                    "box_id": row["box_id"],
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2,
                    "true_height_cm": row["true_height_cm"],
                    "camera_height_cm": row["camera_height_cm"],
                }
            )

    pred_fields = ann_fields + [
        "pred_height_cm",
        "pred_std_cm",
        "abs_error_cm",
        "model_label",
        "feature_mode",
        "pred_samples_cm",
    ]
    with (prediction_dir / "data325_predictions_attn_aug_tta8.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pred_fields)
        writer.writeheader()
        for row in rows:
            x1, y1, x2, y2 = row["bbox"]
            samples = row.get("pred_samples_cm", [])
            writer.writerow(
                {
                    "image_id": row["image_id"],
                    "file_name": row["file_name"],
                    "box_id": row["box_id"],
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2,
                    "true_height_cm": row["true_height_cm"],
                    "camera_height_cm": row["camera_height_cm"],
                    "pred_height_cm": row["pred_height_cm"],
                    "pred_std_cm": row.get("pred_std_cm", ""),
                    "abs_error_cm": row["abs_error_cm"],
                    "model_label": row["model_label"],
                    "feature_mode": row["feature_mode"],
                    "pred_samples_cm": ";".join(str(v) for v in samples),
                }
            )

    box_count_by_file: dict[str, int] = {}
    image_id_by_file: dict[str, str] = {}
    cam_by_file: dict[str, float] = {}
    for row in rows:
        box_count_by_file[row["file_name"]] = box_count_by_file.get(row["file_name"], 0) + 1
        image_id_by_file[row["file_name"]] = row["image_id"]
        cam_by_file[row["file_name"]] = row["camera_height_cm"]

    images = sorted(p for p in DATA325_IMAGE_DIR.glob("*.jpg") if not p.name.startswith("._"))
    with (annotation_dir / "data325_images.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["file_name", "image_id", "width_px", "height_px", "annotated", "box_count", "camera_height_cm"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for img in images:
            with Image.open(img) as im:
                width, height = im.size
            writer.writerow(
                {
                    "file_name": img.name,
                    "image_id": image_id_by_file.get(img.name, ""),
                    "width_px": width,
                    "height_px": height,
                    "annotated": str(img.name in box_count_by_file).lower(),
                    "box_count": box_count_by_file.get(img.name, 0),
                    "camera_height_cm": cam_by_file.get(img.name, ""),
                }
            )

    summary_rows = []
    for src in REPRO_JSONS:
        if not src.exists():
            continue
        obj = json.loads(src.read_text(encoding="utf-8"))
        for model in obj.get("model_results", []):
            s = model.get("summary", {})
            summary_rows.append(
                {
                    "source_json": src.name,
                    "model_label": model.get("label", ""),
                    "feature_mode": model.get("feature_mode", ""),
                    "boxes": s.get("boxes", ""),
                    "images": s.get("images", ""),
                    "mae_cm": s.get("mae_cm", ""),
                    "rmse_cm": s.get("rmse_cm", ""),
                    "median_abs_error_cm": s.get("median_abs_error_cm", ""),
                    "mean_pred_std_cm": s.get("mean_pred_std_cm", ""),
                }
            )
    (OUT / "results").mkdir(parents=True, exist_ok=True)
    with (OUT / "results" / "model_summary.csv").open("w", encoding="utf-8", newline="") as f:
        fields = [
            "source_json",
            "model_label",
            "feature_mode",
            "boxes",
            "images",
            "mae_cm",
            "rmse_cm",
            "median_abs_error_cm",
            "mean_pred_std_cm",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    annotated_images = len(box_count_by_file)
    return {
        "raw_image_count": len(images),
        "annotated_image_count": annotated_images,
        "box_count": len(rows),
        "final_model_label": rows[0]["model_label"] if rows else "",
        "contact_email": CONTACT_EMAIL,
    }


def copy_dataset_images() -> None:
    out_dir = OUT / "data" / "DATA325" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(DATA325_IMAGE_DIR.glob("*.jpg")):
        if src.name.startswith("._"):
            continue
        copy_file(src, out_dir / src.name)


def copy_sanitized_jsons() -> None:
    out_dir = OUT / "results" / "reproducibility_json"
    out_dir.mkdir(parents=True, exist_ok=True)
    for src in REPRO_JSONS:
        if not src.exists():
            continue
        obj = json.loads(src.read_text(encoding="utf-8"))
        sanitized = sanitize_json_value(obj)
        (out_dir / src.name).write_text(json.dumps(sanitized, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def copy_assets() -> None:
    for src in SOURCE_FEATURE_BUNDLES:
        copy_sanitized_checkpoint(src, OUT / "data" / "source_feature_bundles" / src.name)
    for src in CHECKPOINTS:
        copy_sanitized_checkpoint(src, OUT / "checkpoints" / src.name)
    for name in CODE_FILES:
        copy_file(ROOT / name, OUT / "src" / name)

    for src in sorted((ROOT / "cea_submission" / "tables").glob("*.csv")):
        copy_file(src, OUT / "paper_assets" / "tables" / src.name)
    for src in sorted((ROOT / "cea_submission" / "figures").glob("*.png")):
        copy_file(src, OUT / "paper_assets" / "figures" / src.name)
    for src in sorted((ROOT / "cea_submission" / "supplementary_figures").glob("*.png")):
        copy_file(src, OUT / "paper_assets" / "supplementary_figures" / src.name)
    for src in [ROOT / "cea_submission" / "graphical_abstract_non_ai.png", ROOT / "cea_submission" / "submission_asset_manifest.md"]:
        copy_file(src, OUT / "paper_assets" / src.name)
    for src in sorted(CEA_EXP_DIR.glob("*.csv")):
        copy_file(src, OUT / "results" / "cea_revision" / src.name)
    for src in sorted(CEA_EXP_DIR.glob("*.json")):
        if src not in REPRO_JSONS:
            if src.name.endswith("_local.json"):
                continue
            copy_file(src, OUT / "results" / "cea_revision" / src.name)
    for src in sorted((CEA_EXP_DIR / "mask_examples").glob("*.jpg")):
        copy_file(src, OUT / "results" / "cea_revision" / "mask_examples" / src.name)


def write_release_docs(stats: dict[str, Any]) -> None:
    write_text(
        OUT / "README.md",
        f"""
        # DINOv3-DCF maize height open release

        This release supports the manuscript:

        **{TITLE}**

        Author: {AUTHOR}
        Affiliation: {AFFILIATION}
        Correspondence: {CONTACT_EMAIL}
        Repository: {REPOSITORY_URL}
        Companion data-acquisition utility: {CAPTURE_TOOL_URL}

        ## What is included

        - DATA325 greenhouse photographs in `data/DATA325/images/`.
        - Clean DATA325 image metadata, bounding boxes, plant-height labels, camera-height values, and final Attn+aug+TTA8 predictions in `data/DATA325/annotations/` and `data/DATA325/predictions/`.
        - Sanitized evaluation JSON files in `results/reproducibility_json/`.
        - CEA diagnostic outputs in `results/cea_revision/`, including ROI quality metrics, source-trained morphometric baselines, error taxonomy, mask QA examples, bootstrap confidence intervals, paired tests, and 3-seed DCF-head retraining summaries.
        - Selected DiffCorn-Fusion/DCF checkpoints in `checkpoints/`.
        - Source-domain feature bundles in `data/source_feature_bundles/`.
        - Paper figures and table sidecars in `paper_assets/`.
        - Reproducibility helper scripts in `scripts/` and archival project scripts in `src/`.
        - Companion maize image-acquisition utility: {CAPTURE_TOOL_URL}.

        ## Dataset summary

        - Raw DATA325 photographs: {stats["raw_image_count"]}
        - Annotated/evaluated DATA325 photographs: {stats["annotated_image_count"]}
        - Manual DATA325 plant boxes: {stats["box_count"]}
        - Final released prediction file: `data/DATA325/predictions/data325_predictions_attn_aug_tta8.csv`

        The raw photographs contain maize plants and greenhouse background. They do not intentionally contain human subjects.

        ## Quick checks

        ```bash
        python scripts/validate_release.py
        python scripts/summarize_data325.py
        ```

        ## Licenses

        - Code written for this study is released under the MIT License; see `LICENSE`.
        - DATA325 photographs, labels, and non-code tabular outputs are released under CC BY 4.0; see `LICENSE_DATA.txt`.
        - DINOv3 foundation-model weights are not redistributed in this release. Obtain them from the upstream DINOv3 source under the upstream license.

        ## Notes for public upload

        This release is prepared for GitHub at {REPOSITORY_URL}. The companion data-acquisition utility is maintained separately at {CAPTURE_TOOL_URL}. Use Git LFS for `*.jpg`, `*.png`, `*.pt`, and `*.pth` files. No Zenodo DOI is assigned in this release.
        """,
    )
    write_text(
        OUT / "DATASET.md",
        """
        # DATA325 dataset notes

        `data/DATA325/images/` contains all local DATA325 greenhouse photographs included in this release. The manuscript evaluation uses the annotated subset listed in `data/DATA325/annotations/data325_annotations.csv`.

        ## Annotation table

        `data325_annotations.csv` columns:

        - `image_id`: internal image identifier used in the manuscript scripts.
        - `file_name`: raw image file name.
        - `box_id`: plant ROI identifier within the image.
        - `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2`: manual bounding box in source-image pixel coordinates.
        - `true_height_cm`: measured plant height in centimeters.
        - `camera_height_cm`: camera-height metadata used for DCF evaluation.

        ## Prediction table

        `data325_predictions_attn_aug_tta8.csv` repeats the annotation columns and adds:

        - `pred_height_cm`: predicted height from the final Attn+aug+TTA8 evaluation.
        - `pred_std_cm`: test-time augmentation prediction standard deviation.
        - `abs_error_cm`: absolute error relative to measured height.
        - `model_label`, `feature_mode`: released model/evaluation identifiers.
        - `pred_samples_cm`: semicolon-separated TTA predictions.

        ## Evaluation protocol

        DATA325 is treated as an external target-greenhouse benchmark. No DATA325 plant-height labels are used for training in the manuscript protocol. The companion maize image-acquisition utility is available separately at https://github.com/sanyueuy/corn-capture.
        """,
    )
    write_text(
        OUT / "REPRODUCIBILITY.md",
        """
        # Reproducibility notes

        The release is designed to preserve the manuscript evaluation assets rather than to redistribute the DINOv3 foundation model.

        ## Minimal reproducibility path

        1. Inspect `data/DATA325/annotations/data325_annotations.csv`.
        2. Inspect final predictions in `data/DATA325/predictions/data325_predictions_attn_aug_tta8.csv`.
        3. Run `python scripts/summarize_data325.py` to recompute MAE/RMSE/median absolute error from the CSV.
        4. Compare detailed method outputs in `results/reproducibility_json/`.
        5. Inspect `results/cea_revision/` for bootstrap CI, paired tests, ROI contamination diagnostics, source-trained and target-label morphometric baseline output, uncertainty diagnostics, 3-seed DCF-head retraining summaries, and rule-based error taxonomy.

        The companion image-acquisition utility is maintained at https://github.com/sanyueuy/corn-capture. It is referenced as collection-support software, not as a model-training dependency.

        ## Full model path

        Full feature extraction requires the upstream DINOv3 model and its license-compliant weights. The archival scripts in `src/` preserve the exact code used during preparation; some of them retain project-specific path constants and may need path edits when run outside the original project root. CEA diagnostics are reproducible from released prediction CSV/JSON files and real DATA325 images without redistributing DINOv3 weights.

        ## Excluded large/upstream files

        - `model.safetensors` is not included because it is the upstream DINOv3 model artifact, not an author-created checkpoint.
        - YOLO pretrained weights are not needed for the CEA zero-shot ROI protocol and are not included.
        """,
    )
    write_text(
        OUT / "NOTICE.md",
        """
        # Third-party model notice

        This release does not redistribute DINOv3 foundation-model weights. Users must obtain DINOv3 from the upstream source and comply with the upstream DINOv3 license.

        The released `.pth` files are DCF/DiffCorn downstream regression checkpoints generated for this study.
        """,
    )
    write_text(
        OUT / "LICENSE",
        f"""
        MIT License

        Copyright (c) {date.today().year} {AUTHOR}

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.
        """,
    )
    write_text(
        OUT / "LICENSE_DATA.txt",
        """
        DATA LICENSE

        DATA325 photographs, bounding boxes, plant-height labels, camera-height metadata,
        and non-code tabular outputs in this release are made available under the
        Creative Commons Attribution 4.0 International License (CC BY 4.0):

        https://creativecommons.org/licenses/by/4.0/

        Please cite the accompanying manuscript or the public archive DOI when using
        the dataset.
        """,
    )
    write_text(
        OUT / "CITATION.cff",
        f"""
        cff-version: 1.2.0
        title: "DINOv3-DCF maize height open release"
        message: "If you use this dataset or code, please cite the accompanying manuscript and archive DOI."
        type: dataset
        authors:
          - family-names: "Wu"
            given-names: "Hong"
            affiliation: "{AFFILIATION}"
          - family-names: "Chen"
            given-names: "Jian"
            affiliation: "{AFFILIATION}"
            email: "{CONTACT_EMAIL}"
        version: "{VERSION}"
        date-released: "{date.today().isoformat()}"
        license: "CC-BY-4.0"
        repository-code: "{REPOSITORY_URL}"
        abstract: "Open data, code, checkpoints, and reproducibility assets for zero-shot external-greenhouse maize height estimation with DINOv3-DCF."
        """,
    )
    write_text(
        OUT / "requirements.txt",
        """
        numpy
        pillow
        matplotlib
        torch
        transformers
        scikit-learn
        python-docx
        """,
    )
    write_text(
        OUT / ".gitattributes",
        """
        *.jpg filter=lfs diff=lfs merge=lfs -text
        *.png filter=lfs diff=lfs merge=lfs -text
        *.pt filter=lfs diff=lfs merge=lfs -text
        *.pth filter=lfs diff=lfs merge=lfs -text
        *.zip filter=lfs diff=lfs merge=lfs -text
        """,
    )
    write_text(
        OUT / ".gitignore",
        """
        __pycache__/
        *.pyc
        .DS_Store
        .venv/
        venv/
        """,
    )
    write_text(
        OUT / "metadata" / "release_info.json",
        json.dumps(
            {
                "title": TITLE,
                "version": VERSION,
                "date": date.today().isoformat(),
                "author": AUTHOR,
                "affiliation": AFFILIATION,
                "contact_email": CONTACT_EMAIL,
                "repository_url": REPOSITORY_URL,
                "capture_tool_repository_url": CAPTURE_TOOL_URL,
                **stats,
            },
            indent=2,
        )
        + "\n",
    )


def write_helper_scripts() -> None:
    write_text(
        OUT / "scripts" / "summarize_data325.py",
        """
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
        """,
    )
    write_text(
        OUT / "scripts" / "validate_release.py",
        """
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
        """,
    )


def write_checksums_and_manifest() -> None:
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name not in {"MANIFEST.tsv", "checksums_sha256.txt"})
    with (OUT / "MANIFEST.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["path", "bytes", "sha256"])
        for p in files:
            writer.writerow([p.relative_to(OUT).as_posix(), p.stat().st_size, sha256(p)])
    with (OUT / "checksums_sha256.txt").open("w", encoding="utf-8", newline="\n") as f:
        for p in sorted(q for q in OUT.rglob("*") if q.is_file() and q.name != "checksums_sha256.txt"):
            f.write(f"{sha256(p)}  {p.relative_to(OUT).as_posix()}\n")


def zip_release() -> Path:
    reset_dir(DIST)
    zip_path = DIST / f"{SLUG}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for p in sorted(OUT.rglob("*")):
            if p.is_file():
                z.write(p, Path(SLUG) / p.relative_to(OUT))
    (DIST / f"{SLUG}.zip.sha256").write_text(f"{sha256(zip_path)}  {zip_path.name}\n", encoding="utf-8")
    return zip_path


def main() -> None:
    reset_dir(OUT)
    copy_dataset_images()
    stats = write_data_tables()
    copy_sanitized_jsons()
    copy_assets()
    write_release_docs(stats)
    write_helper_scripts()
    write_checksums_and_manifest()
    zip_path = zip_release()
    print("OPEN_RELEASE_DIR", OUT)
    print("OPEN_RELEASE_ZIP", zip_path)
    print("RAW_IMAGES", stats["raw_image_count"])
    print("ANNOTATED_IMAGES", stats["annotated_image_count"])
    print("BOXES", stats["box_count"])
    print("CONTACT_EMAIL", CONTACT_EMAIL)


if __name__ == "__main__":
    main()
