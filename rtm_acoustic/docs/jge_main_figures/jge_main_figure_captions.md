# JGE Main Figure Captions

## Figure 1. Literature-guided FWI-RTM synthesis and claim boundary

The figure reframes the weak FWI image result into a defensible integrated method. It maps recent FWI/RTM literature needs to the implemented response, ranks update-scale candidates, summarizes evidence strength and explicitly separates publishable claims from unsupported claims.

## Figure 2. RTM validation before and after quality-gated FWI updating

RTM is performed with the true velocity, the initial smoothed velocity, and the selected quality-gated FWI velocity. The damped FWI update slightly reduces the Laplacian-filtered image RMSE relative to the true-velocity reference from 0.027130 to 0.027109 in the 12-shot nt=1200 validation.

## Figure 3. Full-aperture RTM imaging-condition diagnostics

Source-normalized, source-receiver-normalized, Laplacian-enhanced, and receiver-illumination panels summarize the full-aperture imaging-condition behavior. The source-receiver normalized image remains close to the source-normalized image, while the Laplacian condition changes the spectral emphasis and reflector expression.

## Figure 4. Illumination-trust spatial FWI update gate

The figure replaces a single global FWI update scale with a spatially varying alpha field controlled by source-receiver illumination. Candidate gates are accepted only when MAE, RMSE, and edge-MAE all improve relative to the initial model. The selected gate improves model error while avoiding the edge degradation observed for the global alpha=0.1 update.

## Figure 5. Target-zone illumination, RTM response, and FWI update diagnostics

Salt-top, salt-flank, and subsalt zones are derived from the high-velocity salt body. The figure evaluates source-receiver illumination, RTM response, and FWI update energy over the same target zones, showing that the subsalt shadow zone has weaker illumination and RTM response while receiving negligible FWI update energy in the current gated workflow.
