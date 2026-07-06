# JGE Innovation Framework

- Official author guide: https://academic.oup.com/jge/pages/general_instructions
- Paper limits used by the automated framework: abstract <=250 words; keywords <=5; references <=50; figures plus tables <=10; main text <=8000 words.
- Framework purpose: keep each manuscript innovation tied to a runnable program, numeric evidence, a figure/table output, and an explicit claim boundary.

## Innovation-to-evidence matrix

| rank | claim_id | manuscript_claim | implemented_program | primary_outputs | main_figure | claim_boundary |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | quality_gated_fwi_rtm | Quality-gated FWI-to-RTM workflow for salt-model imaging. | run_optimized_fwi_rtm_pipeline.py; optimize_fwi_update_scale.py; run_rtm_before_after_fwi.py | optimized_fwi_rtm_pipeline_report.json; fwi_update_scale_optimization.csv; rtm_before_after_summary.json | Figure 1; Figure 2 | Supports conservative velocity-update screening, not high-resolution FWI velocity recovery. |
| 2 | target_zone_illumination_diagnostics | Target-zone illumination, RTM-response, and FWI-update energy diagnostics. | build_target_zone_illumination_diagnostics.py | target_zone_illumination_metrics.csv; figure5_target_zone_illumination_diagnostics.* | Figure 5 | Shows the subsalt target remains weakly illuminated and almost unupdated; it does not claim solved subsalt imaging. |
| 3 | imaging_condition_separation | Explicit separation of RTM image normalization and FWI gradient preconditioning. | run_scheme2_imaging_condition_compare.py; make_jge_main_figures.py | rtm_imaging_condition_metrics.csv; figure3_imaging_condition_diagnostics.* | Figure 3 | Supports imaging-condition interpretation; FWI preconditioning remains a separate ablation result. |
| 4 | negative_ablation_boundary | Negative-result-aware local FWI ablation for illumination preconditioning. | run_small_salt_fwi.py; optimize_existing_salt_result.py | local_fwi_strategy_ranking.csv; adaptive_line_search_summary.json | Figure 4 | Positions illumination preconditioning as a limitation and future method target, not as the current main improvement. |
| 5 | jge_reproducible_submission_pipeline | JGE-oriented reproducible result packaging and readiness checking. | make_jge_main_figures.py; build_jge_submission_package.py; check_jge_submission_readiness.py | jge_submission_package_mainfigures; JGE_readiness_report.md; jge_figure_alt_text.md | Figure 1-5; Table 1-5 | Checks structure and evidence completeness; journal template formatting and final citation verification remain manual. |

## Program-level rule

A claim is manuscript-ready only when it has all four items: runnable script, persisted metric file, figure/table output, and a written claim boundary.
