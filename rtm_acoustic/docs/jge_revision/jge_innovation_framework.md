# JGE Innovation Framework

- Official author guide: https://academic.oup.com/jge/pages/general_instructions
- Paper limits used by the automated framework: abstract <=250 words; keywords <=5; references <=50; figures plus tables <=10; main text <=8000 words.
- Framework purpose: keep each manuscript innovation tied to a runnable program, numeric evidence, a figure/table output, and an explicit claim boundary.

## Innovation-to-evidence matrix

| rank | claim_id | manuscript_claim | implemented_program | primary_outputs | main_figure | claim_boundary |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | illumination_trust_spatial_update_gate | Illumination-trust spatial FWI update gate for salt-model velocity updating. | build_spatial_update_gate.py; optimize_fwi_update_scale.py; run_optimized_fwi_rtm_pipeline.py | spatial_update_gate_candidates.csv; spatial_update_gate_model.npy; figure4_spatial_update_gate.* | Figure 1; Figure 4 | Supports SEG/Salt benchmark update selection, not direct field-data deployment without proxy trust metrics. |
| 2 | quality_gated_fwi_rtm | Quality-gated FWI-to-RTM validation after update selection. | run_rtm_before_after_fwi.py; build_target_zone_illumination_diagnostics.py | fwi_update_scale_optimization.csv; rtm_before_after_summary.json; target_zone_illumination_metrics.csv | Figure 2; Figure 5 | Supports conservative velocity-update screening, not high-resolution FWI velocity recovery. |
| 3 | target_zone_illumination_diagnostics | Target-zone illumination, RTM-response, and FWI-update energy diagnostics. | build_target_zone_illumination_diagnostics.py | target_zone_illumination_metrics.csv; figure5_target_zone_illumination_diagnostics.* | Figure 5 | Shows the subsalt target remains weakly illuminated and almost unupdated; it does not claim solved subsalt imaging. |
| 4 | imaging_condition_separation | Explicit separation of RTM image normalization and FWI gradient preconditioning. | run_scheme2_imaging_condition_compare.py; make_jge_main_figures.py | rtm_imaging_condition_metrics.csv; figure3_imaging_condition_diagnostics.* | Figure 3 | Supports imaging-condition interpretation; FWI preconditioning remains a separate ablation result. |
| 5 | negative_ablation_boundary | Negative-result-aware local FWI ablation as supporting evidence. | run_small_salt_fwi.py; optimize_existing_salt_result.py | local_fwi_strategy_ranking.csv; adaptive_line_search_summary.json | Supplementary table | Positions illumination preconditioning as a limitation and future method target, not as the current main improvement. |

## Program-level rule

A claim is manuscript-ready only when it has all four items: runnable script, persisted metric file, figure/table output, and a written claim boundary.
