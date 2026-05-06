# CEA submission asset manifest

## Main files
- manuscript_cea.docx
- highlights.docx
- highlights.txt
- cover_letter_cea.docx
- supplementary_material.docx
- graphical_abstract_non_ai.png/tif/pdf
- tables/roi_quality_metrics.csv
- tables/error_taxonomy.csv
- tables/seed_retraining_summary.csv
- tables/source_morphometric_baseline.csv

## Main figures
- figures/Figure_1_DATA325_acquisition_annotation_ROI_extraction.png and .pdf: Fig. 1. DATA325 acquisition, annotation, and ROI extraction. Source hand-box examples, independent target-greenhouse images, manual DATA325 boxes, ROI crops, height/camera metadata, and per-box prediction records are shown to make the benchmark auditable.
- figures/Figure_2_Source_DATA325_benchmark_comparison.png and .pdf: Fig. 2. Source and DATA325 benchmark comparison. Real source hand-box examples and real DATA325 target images illustrate the external-domain shift in greenhouse appearance, height range, camera height, and evaluation role.
- figures/Figure_3_DATA325_distribution_imbalance.png and .pdf: Fig. 3. DATA325 distribution and imbalance. The target benchmark contains many early-stage plants, varied camera heights, heterogeneous bbox areas, and different foreground fractions across manually cropped ROIs.
- figures/Figure_4_DINOv3_DCF_zero_shot_workflow.png and .pdf: Fig. 4. DINOv3-DiffCorn-Fusion (DINOv3-DCF) zero-shot workflow. Frozen DINOv3 ROI tokens are aggregated by CLS, patch-mean, or attention-weighted pooling and passed with camera-height context to a phytomer-inspired structured latent DCF head before external DATA325 evaluation.
- figures/Figure_5_Attention_pooling_real_ROIs.png and .pdf: Fig. 5. Attention pooling behavior on real maize ROIs. Real DATA325 crops are paired with deterministic foreground diagnostics and the attention-weighted token-pooling mechanism; the mask overlay is a QA diagnostic, not generated experimental evidence.
- figures/Figure_6_Feature_domain_shift_with_ROI_thumbnails.png and .pdf: Fig. 6. Feature-space domain shift with ROI thumbnails. DINOv3 feature embeddings separate source and DATA325 ROIs, and thumbnail panels show that the clusters correspond to visibly different greenhouse and growth-stage conditions.
- figures/Figure_7_Main_ablation_results_with_CI.png and .pdf: Fig. 7. Main model and baseline ablation with bootstrap confidence intervals. Attention-weighted pooling provides the main DINOv3-DCF gain, while source-trained morphometric baselines show how much manual bbox geometry contributes under the same external target set.
- figures/Figure_8_Seed_robustness_paired_comparison.png and .pdf: Fig. 8. Seed robustness and paired comparison. Independent DCF-head retraining confirms that attention pooling remains the most stable external feature mode among CLS, patch-mean, attention, and attention+augmentation under TTA1 evaluation.
- figures/Figure_9_Height_bin_error_early_stage_failure.png and .pdf: Fig. 9. Height-bin error and early-stage failure concentration. Bootstrap intervals confirm that plants below 80 cm dominate relative error and remain the key target for future stage-aware adaptation.
- figures/Figure_10_ROI_contamination_morphometric_diagnostics.png and .pdf: Fig. 10. ROI contamination and morphometric diagnostics. Foreground/background fractions from deterministic color-index masks are weakly correlated with absolute error, while source-trained bbox/mask baselines quantify the strength and limits of simple agricultural geometry.
- figures/Figure_11_Qualitative_stage_attention_error_gallery.png and .pdf: Fig. 11. Qualitative stage-wise and attention/error gallery. Real DATA325 examples combine low-, mid-, and tall-stage cases with success, over-estimation, under-estimation, sparse-plant, and high-uncertainty ROIs.
- figures/Figure_12_Diagnostic_negative_controls.png and .pdf: Fig. 12. Diagnostic negative controls. Camera-height correction, bbox geometry, feature-statistic alignment, and DANN do not remove the external-greenhouse gap, so the remaining error is not a single metadata or marginal-alignment artifact.

## Supplementary figures
- supplementary_figures/Supplementary_Figure_1_Preprocessing_ROI_quality_examples.png and .pdf: Supplementary Fig. 1. Preprocessing and ROI-quality examples. Raw DATA325 images are converted to manual boxes, ROI crops, resized model inputs, and deterministic plant-mask diagnostics used only to quantify foreground/background contamination.
- supplementary_figures/Supplementary_Figure_2_STAGE_wise_DATA325_qualitative_gallery.png and .pdf: Supplementary Fig. 2. Stage-wise DATA325 qualitative gallery. Low-, mid-, and tall-stage real images are shown with manual boxes, ROI crops, ground truth, prediction, and absolute error.
- supplementary_figures/Supplementary_Figure_3_Attention_error_overlay_gallery.png and .pdf: Supplementary Fig. 3. Error-overlay gallery for real DATA325 ROIs. Successful, over-estimated, under-estimated, sparse, and uncertain cases are shown with ROI crops, deterministic plant-focus overlays, ground truth, prediction, and error.
- supplementary_figures/Supplementary_Figure_4_Open_release_future_deployment_map.png and .pdf: Supplementary Fig. 4. Open-release and future-deployment map. The release packages DATA325 images, annotations, predictions, diagnostics, scripts, and checkpoints, while future work connects automatic detection, segmentation-guided ROI normalization, and multi-greenhouse adaptation.

## Diagnostic outputs
- experiments/cea_revision/run_cea_revision_experiments.py generated bootstrap CI, paired tests, ROI quality metrics, morphometric baselines, uncertainty diagnostics, and error taxonomy.
- experiments/cea_revision/seed_retraining contains the 3-seed DCF-head training reports and DATA325 zero-shot evaluation output; summary sidecars are seed_retraining_summary.json/csv.
- source_morphometric_baseline.json/csv reports source-trained RidgeCV and RandomForestRegressor bbox/mask baselines evaluated zero-shot on DATA325.
- Deterministic color-index masks quantify foreground/background only; they are not generated images, labels, or model training inputs.
- Multi-seed robustness is DCF-head retraining over frozen feature bundles; it does not fine-tune DINOv3.

## Open-source release
- Repository: https://github.com/sanyueuy/dinov3-dcf-maize-height
- Planned archive: dinov3_dcf_maize_height_open_release_v0_1_0.zip

## Added literature sources
- CEA crop-height/protected crop: Chang 2017; Xie 2021; Kim 2021; Jayasuriya 2024.
- CEA phenotyping/AI reviews and datasets: Patricio 2018; Kamilaris 2018; Li 2020; Liu 2020; Reena 2025.
- CEA crop-specific examples: Li 2019; Xing 2023; Che 2024; Veramendi and Cruvinel 2024; Ariza-Sentis 2024.
- Generalization/foundation models: Sun and Saenko 2016; Koh 2021; Gulrajani and Lopez-Paz 2021; Caron 2021; Kirillov 2023.

## Reproducibility JSON
- reproducibility_json/attention_geometry_prior_phaseA.json
- reproducibility_json/bbox_geometry_prior_phaseA.json
- reproducibility_json/bootstrap_ci.json
- reproducibility_json/cea_revision_summary.json
- reproducibility_json/data325_feature_stats_attn_aug_featurealign.json
- reproducibility_json/data325_zero_shot_comparison_attn_aug.json
- reproducibility_json/data325_zero_shot_comparison_attn_aug_featurealign.json
- reproducibility_json/data325_zero_shot_comparison_attn_aug_tta8.json
- reproducibility_json/data325_zero_shot_comparison_corrected_camheight.json
- reproducibility_json/data325_zero_shot_comparison_dann.json
- reproducibility_json/data325_zero_shot_comparison_seed_retraining.json
- reproducibility_json/diffcorn_fusion_hand_dann_history.json
- reproducibility_json/error_taxonomy_summary.json
- reproducibility_json/height_bin_bootstrap.json
- reproducibility_json/morphometric_baseline.json
- reproducibility_json/paired_tests.json
- reproducibility_json/resampling_robustness.json
- reproducibility_json/roi_quality_summary.json
- reproducibility_json/seed_retraining_summary.json
- reproducibility_json/source_morphometric_baseline.json
- reproducibility_json/tsne_source_vs_data325.json
- reproducibility_json/uncertainty_diagnostic.json
