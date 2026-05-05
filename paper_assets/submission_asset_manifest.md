# CEA submission asset manifest

## Main files
- manuscript_cea.docx
- highlights.docx
- highlights.txt
- cover_letter_cea.docx
- supplementary_material.docx
- graphical_abstract_non_ai.png/pdf

## Main figures
- figures/Figure_1_DATA325_acquisition_annotation_protocol.png and .pdf: Fig. 1. DATA325 acquisition and annotation protocol. Source hand-box examples, independent target-greenhouse images, manual DATA325 boxes, ROI crops, height/camera metadata, and per-box prediction records are shown to make the benchmark auditable.
- figures/Figure_2_Source_DATA325_benchmark_comparison.png and .pdf: Fig. 2. Source and DATA325 benchmark comparison. Real source hand-box examples and real DATA325 target images illustrate the external-domain shift in greenhouse appearance, height range, camera height, and evaluation role.
- figures/Figure_3_DATA325_distribution_imbalance.png and .pdf: Fig. 3. DATA325 distribution and imbalance. The target benchmark contains many early-stage plants, varied camera heights, heterogeneous bbox areas, and different foreground fractions across manually cropped ROIs.
- figures/Figure_4_Preprocessing_ROI_quality_examples.png and .pdf: Fig. 4. Preprocessing and ROI-quality examples. Raw DATA325 images are converted to manual boxes, ROI crops, resized model inputs, and deterministic plant-mask diagnostics used only to quantify foreground/background contamination.
- figures/Figure_5_DINOv3_DCF_zero_shot_workflow.png and .pdf: Fig. 5. DINOv3-DCF zero-shot workflow. Frozen DINOv3 ROI tokens are aggregated by CLS, patch-mean, or attention-weighted pooling and passed with camera-height context to the DCF regression head before external DATA325 evaluation.
- figures/Figure_6_Attention_pooling_real_ROIs.png and .pdf: Fig. 6. Attention pooling with real maize ROIs. Real DATA325 crops are paired with deterministic foreground diagnostics and the attention-weighted token-pooling mechanism; the mask overlay is a QA diagnostic, not generated experimental evidence.
- figures/Figure_7_Feature_domain_shift_with_ROI_thumbnails.png and .pdf: Fig. 7. Feature-space domain shift with ROI thumbnails. DINOv3 feature embeddings separate source and DATA325 ROIs, and thumbnail panels show that the clusters correspond to visibly different greenhouse and growth-stage conditions.
- figures/Figure_8_Main_ablation_results_with_CI.png and .pdf: Fig. 8. Main ablation results with bootstrap confidence intervals. Attention-weighted pooling provides the largest external DATA325 gain; augmentation and TTA8 provide smaller additional improvements.
- figures/Figure_9_Resampling_robustness_statistical_comparison.png and .pdf: Fig. 9. Resampling robustness and statistical comparison. Bootstrap resampling and paired per-box differences show that the attention/TTA8 result is not explained by a small number of boxes; this is not independent seed retraining.
- figures/Figure_10_Height_bin_error_early_stage_failure.png and .pdf: Fig. 10. Height-bin error and early-stage failure concentration. Bootstrap intervals confirm that plants below 80 cm dominate relative error and remain the key target for future stage-aware adaptation.
- figures/Figure_11_ROI_contamination_error_correlation.png and .pdf: Fig. 11. ROI contamination and error correlation. Foreground/background fractions from deterministic color-index masks are weakly correlated with absolute error, indicating that clutter contributes but does not fully explain the domain gap.
- figures/Figure_12_STAGE_wise_DATA325_qualitative_gallery.png and .pdf: Fig. 12. Stage-wise DATA325 qualitative gallery. Low-, mid-, and tall-stage real images are shown with manual boxes, ROI crops, ground truth, prediction, and absolute error.
- figures/Figure_13_Attention_error_overlay_gallery.png and .pdf: Fig. 13. Error-overlay gallery for real DATA325 ROIs. Successful, over-estimated, under-estimated, sparse, and uncertain cases are shown with ROI crops, deterministic plant-focus overlays, ground truth, prediction, and error.
- figures/Figure_14_Diagnostic_negative_controls.png and .pdf: Fig. 14. Diagnostic negative controls. Camera-height correction, bbox geometry, feature-statistic alignment, and DANN do not remove the cross-greenhouse gap, so the remaining error is not a single metadata or marginal-alignment artifact.
- figures/Figure_15_Open_release_future_deployment_map.png and .pdf: Fig. 15. Open-release and future-deployment map. The release packages DATA325 images, annotations, predictions, diagnostics, scripts, and checkpoints, while future work connects automatic detection, segmentation-guided ROI normalization, and multi-greenhouse adaptation.

## Revision diagnostics
- experiments/cea_revision/run_cea_revision_experiments.py generated bootstrap CI, paired tests, ROI quality metrics, morphometric baseline, uncertainty diagnostics, and error taxonomy.
- Deterministic color-index masks quantify foreground/background only; they are not generated images, labels, or model training inputs.
- Resampling robustness is bootstrap analysis of existing predictions and is not reported as independent random-seed retraining.

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
- reproducibility_json/diffcorn_fusion_hand_dann_history.json
- reproducibility_json/error_taxonomy_summary.json
- reproducibility_json/height_bin_bootstrap.json
- reproducibility_json/morphometric_baseline.json
- reproducibility_json/paired_tests.json
- reproducibility_json/resampling_robustness.json
- reproducibility_json/roi_quality_summary.json
- reproducibility_json/tsne_source_vs_data325.json
- reproducibility_json/uncertainty_diagnostic.json
