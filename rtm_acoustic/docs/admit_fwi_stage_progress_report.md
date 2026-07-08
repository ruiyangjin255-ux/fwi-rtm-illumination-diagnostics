# ADMIT-FWI split/ROI/evidence 图件阶段进展报告

生成日期：2026-07-08

## 完成情况

- 已基于真实 `subset_A/subset_B` RTM 输出生成 split consistency 图件。
- 已基于 ROI diagnostics 生成 ROI 能量、更新量和 admissibility matrix 图件。
- 已基于 ADMIT evidence matrix 生成跨域证据热图和方法排名图。
- 图件已同步复制到 `outputs/admit_fwi_v1/paper_assets/figures/`。

## 关键 split 指标

- illumination `rtm_split_laplacian_correlation`: `8.96963e-05`
- ECG `rtm_split_laplacian_correlation`: `7.96449e-05`
- random_seed_4 `rtm_split_laplacian_correlation`: `8.99569e-05`

当前 split 指标不支持“ECG 显著优于 illumination-only”。

## Evidence verdict 摘要

| method | overall_admissibility_verdict | deep_time_status |
|---|---|---|
| initial | NOT_SUPPORTED_FOR_DEEP_SUBSALT | NOT_RELEASED_FOR_DEEP_INTERPRETATION |
| full_fwi | REQUIRES_IMAGE_CONSISTENCY_CHECK | NOT_RELEASED_FOR_DEEP_INTERPRETATION |
| global | REQUIRES_IMAGE_CONSISTENCY_CHECK | NOT_RELEASED_FOR_DEEP_INTERPRETATION |
| illumination | REQUIRES_IMAGE_CONSISTENCY_CHECK | NOT_RELEASED_FOR_DEEP_INTERPRETATION |
| consensus | REQUIRES_IMAGE_CONSISTENCY_CHECK | NOT_RELEASED_FOR_DEEP_INTERPRETATION |
| depth | REQUIRES_IMAGE_CONSISTENCY_CHECK | NOT_RELEASED_FOR_DEEP_INTERPRETATION |
| inverse | REQUIRES_IMAGE_CONSISTENCY_CHECK | NOT_RELEASED_FOR_DEEP_INTERPRETATION |
| ecg | REQUIRES_IMAGE_CONSISTENCY_CHECK | NOT_RELEASED_FOR_DEEP_INTERPRETATION |
| random_seed_0 | REQUIRES_IMAGE_CONSISTENCY_CHECK | NOT_RELEASED_FOR_DEEP_INTERPRETATION |
| random_seed_1 | REQUIRES_IMAGE_CONSISTENCY_CHECK | NOT_RELEASED_FOR_DEEP_INTERPRETATION |
| random_seed_2 | REQUIRES_IMAGE_CONSISTENCY_CHECK | NOT_RELEASED_FOR_DEEP_INTERPRETATION |
| random_seed_3 | REQUIRES_IMAGE_CONSISTENCY_CHECK | NOT_RELEASED_FOR_DEEP_INTERPRETATION |
| random_seed_4 | REQUIRES_IMAGE_CONSISTENCY_CHECK | NOT_RELEASED_FOR_DEEP_INTERPRETATION |

## 生成图件

- `D:\ryjin\rtm_acoustic\outputs\admit_fwi_v1\seg_salt_main_case\split_consistency\figures\figure_pairwise_split_delta.png`
- `D:\ryjin\rtm_acoustic\outputs\admit_fwi_v1\seg_salt_main_case\split_consistency\figures\figure_split_metric_bars.png`
- `D:\ryjin\rtm_acoustic\outputs\admit_fwi_v1\seg_salt_main_case\split_consistency\figures\figure_split_rtm_images.png`
- `D:\ryjin\rtm_acoustic\outputs\admit_fwi_v1\seg_salt_main_case\roi_diagnostics\figures\figure_roi_admissibility_matrix.png`
- `D:\ryjin\rtm_acoustic\outputs\admit_fwi_v1\seg_salt_main_case\roi_diagnostics\figures\figure_roi_rtm_energy.png`
- `D:\ryjin\rtm_acoustic\outputs\admit_fwi_v1\seg_salt_main_case\roi_diagnostics\figures\figure_roi_update_energy.png`
- `D:\ryjin\rtm_acoustic\outputs\admit_fwi_v1\seg_salt_main_case\evidence_matrix\figures\figure_data_model_image_deeptime_matrix.png`
- `D:\ryjin\rtm_acoustic\outputs\admit_fwi_v1\seg_salt_main_case\evidence_matrix\figures\figure_method_ranking_by_domain.png`

## 可写入论文的结论

- ADMIT-FWI now includes true split-RTM image consistency for the SEG/Salt short-record case.
- Spatial selective gates can be audited against global, inverse, and random controls.
- Illumination-only remains a strong baseline.
- ECG is an evidence-calibrated candidate, but current split/ROI metrics do not establish unique superiority.

## 仍禁止的结论

- ECG significantly improves FWI/RTM imaging.
- ADMIT-FWI solves subsalt velocity building.
- Short-record split RTM proves deep imaging quality.
- Full FWI is most reliable only because residual is lowest.
