# FWI model quality ranking

| rank_metric | case | optimizer | shots | iterations | mae_improvement_pct | rmse_improvement_pct | edge_mae_improvement_pct | gradient_mae_improvement_pct | update_l1_edge_fraction | update_true_error_correlation | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| model_quality_improvement | CG_allshots_v2 | cg | 224 | 3 | 0.1119 | 0.0100 | -0.0893 | -3.6250 | 0.165082 | 0.035303 | numerical_improvement_without_gradient_improvement |
| model_quality_improvement | CG_f15_outerpad_continue | cg | 112 | 1 | 0.0182 | 0.0008 | 0.0015 | -0.0059 | 0.067023 | 0.021610 | numerical_improvement_without_gradient_improvement |
| model_quality_improvement | CG_nt4000_continue | cg | 224 | 1 | -0.0119 | -0.0054 | -0.0019 | 0.0330 | 0.036994 | 0.007257 | not_improved |
| model_quality_improvement | P-CG_allshots_v2 | p-cg | 224 | 3 | -0.6938 | -0.0724 | -0.7460 | -6.0091 | 0.144808 | 0.035096 | not_improved |
| model_quality_improvement | CG_nt2500_2iter | cg | 224 | 3 | -8.7112 | -1.7829 | -2.2031 | -5.6595 | 0.091733 | -0.070882 | not_improved |
| model_quality_improvement | P-CG_nt2500_2iter | p-cg | 224 | 3 | -9.4243 | -1.9163 | -2.4975 | -5.8165 | 0.093129 | -0.073210 | not_improved |
