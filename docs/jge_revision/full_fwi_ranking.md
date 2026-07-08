# Full-model FWI ranking

| rank_metric | case | optimizer | shots | iterations | nt | f0_hz | initial_misfit | final_misfit | misfit_reduction_pct | evidence_level |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_model_misfit_reduction | CG_allshots_v2 | cg | 224 | 3 | 900 | 8.0 | 4.44358995e-05 | 2.04560124e-05 | 53.9651 | core |
| full_model_misfit_reduction | P-CG_allshots_v2 | p-cg | 224 | 3 | 900 | 8.0 | 4.44358995e-05 | 2.31982770e-05 | 47.7938 | core |
| full_model_misfit_reduction | CG_nt2500_2iter | cg | 224 | 3 | 2500 | 8.0 | 9.00145530e-04 | 7.00359282e-04 | 22.1949 | supporting |
| full_model_misfit_reduction | P-CG_nt2500_2iter | p-cg | 224 | 3 | 2500 | 8.0 | 9.00145530e-04 | 7.06910451e-04 | 21.4671 | supporting |
| full_model_misfit_reduction | CG_nt4000_continue | cg | 224 | 1 | 4000 | 8.0 | 7.84438778e-04 | 7.84438778e-04 | 0.0000 | supporting |
| full_model_misfit_reduction | CG_f15_outerpad_continue | cg | 112 | 1 | 2500 | 15.0 | 1.27860308e-02 | 1.27860308e-02 | 0.0000 | supporting |
