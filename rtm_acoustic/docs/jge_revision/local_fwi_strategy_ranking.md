# Local-window FWI strategy ranking

| rank_metric | case | epsilon | max_update | line_search | selected_steps | initial_misfit | final_misfit | misfit_reduction_pct | claim_role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| local_window_misfit_reduction | adaptive_line_search_baseline |  |  | adaptive extended | 8.0,8.0,8.0 | 0.2963276605 | 0.2774544110 | 6.3690 | core optimizer evidence |
| local_window_misfit_reduction | adaptive_line_search_illumination_preconditioned | 0.5 |  | adaptive extended | 8.0,8.0,8.0 | 0.2963276605 | 0.2840638161 | 4.1386 | negative ablation |
| local_window_misfit_reduction | line_search_baseline |  |  | fixed candidates | 2.0,2.0,2.0 | 0.2963276605 | 0.2910634528 | 1.7765 | optimizer evidence |
| local_window_misfit_reduction | line_search_illumination_preconditioned | 0.5 |  | fixed candidates | 2.0,2.0,2.0 | 0.2963276605 | 0.2931125114 | 1.0850 | optimizer evidence |
| local_window_misfit_reduction | fixed_step_baseline |  |  | none |  | 0.2963276605 | 0.2936459879 | 0.9050 | baseline reference |
| local_window_misfit_reduction | fixed_step_illumination_preconditioned_best_2d | 0.5 | 20.0 | none |  | 0.2963276605 | 0.2947063794 | 0.5471 | negative ablation |
| local_window_misfit_reduction | fixed_step_illumination_preconditioned_best_1d | 0.2 |  | none |  | 0.2963276605 | 0.2950215439 | 0.4408 | negative ablation |
