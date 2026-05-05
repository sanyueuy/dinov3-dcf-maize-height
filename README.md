# DINOv3-DCF maize height open release

This release supports the manuscript:

**A reproducible cross-greenhouse maize height benchmark and attention-guided DINOv3-DCF evaluation pipeline**

Author: Hong Wu; Jian Chen
Affiliation: China Agricultural University
Correspondence: jchen@cau.edu.cn
Repository: https://github.com/sanyueuy/dinov3-dcf-maize-height

## What is included

- DATA325 greenhouse photographs in `data/DATA325/images/`.
- Clean DATA325 image metadata, bounding boxes, plant-height labels, camera-height values, and final Attn+aug+TTA8 predictions in `data/DATA325/annotations/` and `data/DATA325/predictions/`.
- Sanitized evaluation JSON files in `results/reproducibility_json/`.
- CEA revision diagnostics in `results/cea_revision/`, including ROI quality metrics, error taxonomy, mask QA examples, bootstrap confidence intervals, paired tests, and 3-seed DCF-head retraining summaries.
- Selected DiffCorn-Fusion/DCF checkpoints in `checkpoints/`.
- Source-domain feature bundles in `data/source_feature_bundles/`.
- Paper figures and table sidecars in `paper_assets/`.
- Reproducibility helper scripts in `scripts/` and archival project scripts in `src/`.

## Dataset summary

- Raw DATA325 photographs: 75
- Annotated/evaluated DATA325 photographs: 25
- Manual DATA325 plant boxes: 82
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

This release is prepared for GitHub at https://github.com/sanyueuy/dinov3-dcf-maize-height. Use Git LFS for `*.jpg`, `*.png`, `*.pt`, and `*.pth` files. If a Zenodo DOI is minted from a GitHub release, add that DOI to the manuscript before final submission.
