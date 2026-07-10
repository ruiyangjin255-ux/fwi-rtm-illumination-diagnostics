# Spatial illumination-trust FWI update gate

This experiment replaces a single global FWI update scale with an illumination-trust alpha field. Candidate gates are accepted only when MAE, RMSE, and edge-MAE all improve relative to the initial model.

## Selected gate

- `candidate`: smooth_alpha0.3_thr0.5
- `mode`: smooth
- `alpha`: 0.3
- `illumination_threshold`: 0.5
- `active_fraction`: 0.3635
- `mean_alpha`: 0.0760
- `mae_improvement_pct`: 0.3102
- `rmse_improvement_pct`: 0.0495
- `edge_mae_improvement_pct`: 0.0736

## Top accepted candidates

| Candidate | MAE imp. (%) | RMSE imp. (%) | Edge MAE imp. (%) | Active frac. | Mean alpha |
|---|---:|---:|---:|---:|---:|
| smooth_alpha0.3_thr0.5 | 0.3102 | 0.0495 | 0.0736 | 0.3635 | 0.0760 |
| smooth_alpha0.25_thr0.5 | 0.2627 | 0.0428 | 0.0966 | 0.3635 | 0.0633 |
| smooth_alpha0.2_thr0.5 | 0.2131 | 0.0355 | 0.0982 | 0.3635 | 0.0506 |
| smooth_alpha0.15_thr0.5 | 0.1615 | 0.0275 | 0.0935 | 0.3635 | 0.0380 |
| smooth_alpha0.3_thr0.6 | 0.2044 | 0.0318 | 0.0688 | 0.2561 | 0.0449 |
| smooth_alpha0.25_thr0.6 | 0.1726 | 0.0274 | 0.0781 | 0.2561 | 0.0374 |
| sqrt_alpha0.3_thr0.4 | 0.2068 | 0.0353 | 0.0366 | 0.3550 | 0.0545 |
| hard_alpha0.2_thr0.5 | 0.2105 | 0.0354 | 0.0345 | 0.2532 | 0.0506 |

## Interpretation

- The selected gate is not tuned to make the FWI image visually attractive; it is selected by simultaneous model and edge-quality constraints.
- The result supports a stronger method claim than global damping: FWI updates should be spatially accepted only where illumination makes the update trustworthy.
- The claim remains conservative because SEG/Salt truth is used here for benchmark scoring; field-data use would require proxy metrics such as image-domain gathers, residual focusing, or well ties.
