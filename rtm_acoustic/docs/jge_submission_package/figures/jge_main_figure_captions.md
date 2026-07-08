# JGE Main Figure Captions

## Figure 1. FWI model-quality gate and update-scale selection

The full SEG/Salt FWI update reduces data misfit but does not improve salt-boundary metrics. The quality gate therefore evaluates model MAE, RMSE, edge MAE, and gradient MAE over candidate update scales and selects alpha=0.1 as the only non-degrading accepted update for downstream RTM.

## Figure 2. RTM validation before and after quality-gated FWI updating

RTM is performed with the true velocity, the initial smoothed velocity, and the selected quality-gated FWI velocity. The damped FWI update slightly reduces the Laplacian-filtered image RMSE relative to the true-velocity reference from 0.027130 to 0.027109 in the 12-shot nt=1200 validation.

## Figure 3. Full-aperture RTM imaging-condition diagnostics

Source-normalized, source-receiver-normalized, Laplacian-enhanced, and receiver-illumination panels summarize the full-aperture imaging-condition behavior. The source-receiver normalized image remains close to the source-normalized image, while the Laplacian condition changes the spectral emphasis and reflector expression.

## Figure 4. Local FWI ablation and conservative claim boundary

The local-window FWI ablation shows that adaptive line search is more effective than the current lightweight illumination preconditioner. The bottom panels summarize the conservative claim boundary: full FWI updates must pass model-quality gates before being used for RTM interpretation.
