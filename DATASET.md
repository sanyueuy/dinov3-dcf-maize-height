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
