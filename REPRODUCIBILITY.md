# Reproducibility notes

The release is designed to preserve the manuscript evaluation assets rather than to redistribute the DINOv3 foundation model.

## Minimal reproducibility path

1. Inspect `data/DATA325/annotations/data325_annotations.csv`.
2. Inspect final predictions in `data/DATA325/predictions/data325_predictions_attn_aug_tta8.csv`.
3. Run `python scripts/summarize_data325.py` to recompute MAE/RMSE/median absolute error from the CSV.
4. Compare detailed method outputs in `results/reproducibility_json/`.
5. Inspect `results/cea_revision/` for bootstrap CI, paired tests, ROI contamination diagnostics, source-trained and target-label morphometric baseline output, uncertainty diagnostics, 3-seed DCF-head retraining summaries, and rule-based error taxonomy.

## Full model path

Full feature extraction requires the upstream DINOv3 model and its license-compliant weights. The archival scripts in `src/` preserve the exact code used during preparation; some of them retain project-specific path constants and may need path edits when run outside the original project root. CEA diagnostics are reproducible from released prediction CSV/JSON files and real DATA325 images without redistributing DINOv3 weights.

## Excluded large/upstream files

- `model.safetensors` is not included because it is the upstream DINOv3 model artifact, not an author-created checkpoint.
- YOLO pretrained weights are not needed for the CEA zero-shot ROI protocol and are not included.
