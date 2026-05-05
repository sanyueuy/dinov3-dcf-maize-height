# CEA submission asset manifest

## Main files
- manuscript_cea.docx
- highlights.docx
- highlights.txt
- cover_letter_cea.docx
- supplementary_material.docx
- graphical_abstract_non_ai.png
- graphical_abstract_non_ai.pdf

## Figures
- figures/Figure_1_Real_image_protocol.png and .pdf: Fig. 1. Real-image protocol for zero-shot cross-greenhouse maize height estimation. (a) Source-domain hand-box examples with measured heights. (b) Independent DATA325 target greenhouse images with manual bounding-box annotations. (c) Extracted ROI crops and per-box evaluation records including ground-truth height, DINOv3-DCF prediction, absolute error, camera height, and TTA standard deviation.
- figures/Figure_2_Dataset_statistics.png and .pdf: Fig. 2. Source and DATA325 dataset statistics. The source-domain hand-box set spans taller plants, whereas DATA325 contains more early-stage plants and a different target greenhouse distribution.
- figures/Figure_3_DINOv3_DCF_workflow.png and .pdf: Fig. 3. DINOv3-DiffCorn-Fusion (DINOv3-DCF) workflow for ROI-level maize height estimation. A frozen DINOv3 ViT-L backbone extracts plant-region features, the DCF head maps visual and camera-height inputs to phytomer parameters, and height is obtained from internode-related outputs.
- figures/Figure_4_Attention_weighted_pooling.png and .pdf: Fig. 4. Attention-weighted patch aggregation. Final-layer CLS-to-patch attention weights are averaged across heads, normalized across patch positions, and used to pool patch tokens before DCF regression.
- figures/Figure_5_Domain_shift_tsne.png and .pdf: Fig. 5. DINOv3 feature-space domain shift between source ROIs and DATA325 target ROIs. The t-SNE visualization uses 20 sampled source and 20 sampled target ROI features.
- figures/Figure_6_Ablation_results.png and .pdf: Fig. 6. Zero-shot DATA325 ablation results. Replacing CLS pooling with attention-weighted patch aggregation produced the largest single reduction in external MAE.
- figures/Figure_7_Height_bin_errors.png and .pdf: Fig. 7. DATA325 error distribution by ground-truth plant-height bin for the best Attn+aug+TTA8 model. Early-stage plants below 80 cm remain the dominant source of relative error.
- figures/Figure_8_DATA325_stage_error_gallery.png and .pdf: Fig. 8. DATA325 real-image stage and error gallery. Full target-greenhouse images with manual boxes and ROI crops show that early plants and cluttered backgrounds are visually overrepresented among the high-error cases.
- figures/Figure_9_Diagnostic_experiments.png and .pdf: Fig. 9. Diagnostic experiments for alternative explanations of the domain gap, including geometric priors, feature-statistic alignment, and domain-adversarial training.
- figures/Figure_10_Qualitative_DATA325_examples.png and .pdf: Fig. 10. Real DATA325 ROI examples and DINOv3 attention overlays. Low-stage examples have less plant-structure evidence and show larger relative errors than taller plants.

## Real-image figure generation
- figures/Figure_1_Real_image_protocol.* is generated from DATA325 captured greenhouse photographs, manual bounding boxes, ROI crops, and Attn+aug+TTA8 prediction records.
- figures/Figure_8_DATA325_stage_error_gallery.* is generated from DATA325 photographs grouped by ground-truth height bin and absolute-error rank.
- figures/Figure_9_Diagnostic_experiments.* is redrawn deterministically from bbox geometry, feature-alignment, and DANN-history JSON files.

## Open-source release package
- dinov3_dcf_maize_height_open_release_v0_1_0.zip: local public-release archive containing DATA325 images, cleaned annotations, sanitized evaluation JSON, selected DCF checkpoints, reproducibility scripts, licenses, citation metadata, and checksums. Public repository: https://github.com/sanyueuy/dinov3-dcf-maize-height.

## Tables
- tables/table1_ablation.csv
- tables/table2_height_bins.csv

## Newly added literature sources
- CEA crop-height and protected-facility machine vision: Chang et al. 2017; Xie et al. 2021; Kim et al. 2021; Jayasuriya et al. 2024.
- CEA agricultural computer vision and phenotyping reviews: Patricio and Rieder 2018; Kamilaris and Prenafeta-Boldu 2018; Li et al. 2020; Liu et al. 2020; Ariza-Sentis et al. 2024.
- CEA crop-specific and multi-environment examples: Li et al. 2019; Xing et al. 2023; Che et al. 2024; Veramendi and Cruvinel 2024; Reena et al. 2025.
- Domain-shift and foundation-model background: Sun and Saenko 2016; Koh et al. 2021; Gulrajani and Lopez-Paz 2021; Caron et al. 2021; Kirillov et al. 2023.

## Reproducibility JSON
- reproducibility_json/attention_geometry_prior_phaseA.json
- reproducibility_json/bbox_geometry_prior_phaseA.json
- reproducibility_json/data325_feature_stats_attn_aug_featurealign.json
- reproducibility_json/data325_zero_shot_comparison_attn_aug.json
- reproducibility_json/data325_zero_shot_comparison_attn_aug_featurealign.json
- reproducibility_json/data325_zero_shot_comparison_attn_aug_tta8.json
- reproducibility_json/data325_zero_shot_comparison_corrected_camheight.json
- reproducibility_json/data325_zero_shot_comparison_dann.json
- reproducibility_json/diffcorn_fusion_hand_dann_history.json
- reproducibility_json/tsne_source_vs_data325.json
