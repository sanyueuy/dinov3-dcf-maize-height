# DINOv3-DCF maize height open release

This release supports the manuscript:

**A reproducible external-greenhouse benchmark for diagnosing DINOv3 feature transfer in maize height estimation**

Author: Hong Wu; Jian Chen
Affiliation: College of Engineering, China Agricultural University, Beijing 100083, China
First-author email: wuhong@cau.edu.cn
Correspondence: jchen@cau.edu.cn
Repository: https://github.com/sanyueuy/dinov3-dcf-maize-height
Companion data-acquisition utility: https://github.com/sanyueuy/corn-capture

## What is included

- DATA325 greenhouse photographs in `data/DATA325/images/`.
- Clean DATA325 image metadata, bounding boxes, plant-height labels, camera-height values, and final Attn+aug+TTA8 predictions in `data/DATA325/annotations/` and `data/DATA325/predictions/`.
- Sanitized evaluation JSON files in `results/reproducibility_json/`.
- CEA diagnostic outputs in `results/cea_revision/`, including ROI quality metrics, source-trained morphometric baselines, error taxonomy, mask QA examples, bootstrap confidence intervals, paired tests, and 3-seed DCF-head retraining summaries.
- Selected DiffCorn-Fusion/DCF checkpoints in `checkpoints/`.
- Source-domain feature bundles in `data/source_feature_bundles/`.
- Paper figures and table sidecars in `paper_assets/`.
- Reproducibility helper scripts in `scripts/` and archival project scripts in `src/`.
- Companion maize image-acquisition utility: https://github.com/sanyueuy/corn-capture.

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

This release is prepared for GitHub at https://github.com/sanyueuy/dinov3-dcf-maize-height. The companion data-acquisition utility is maintained separately at https://github.com/sanyueuy/corn-capture. Use Git LFS for `*.jpg`, `*.png`, `*.pt`, and `*.pth` files. No Zenodo DOI is assigned in this release.
