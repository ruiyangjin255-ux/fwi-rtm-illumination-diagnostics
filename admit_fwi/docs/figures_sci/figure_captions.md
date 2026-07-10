# SCI figure package

## Figure 1. Acoustic RTM workflow

The workflow starts from the SEG/Salt velocity model, applies model padding and smoothing, performs finite-difference source modeling and receiver reverse-time propagation, forms a zero-lag cross-correlation image, and generates illumination-normalized and Laplacian-enhanced image candidates. Checkpointed accumulation records completed shots for recoverable full-shot computation.

## Figure 2. Conservative scheme-1 RTM display

The scheme-1 result is generated from the existing full RTM image by conservative display optimization only. The acoustic propagator and zero-lag cross-correlation imaging condition are unchanged. This panel is recommended as the stable main result because it improves readability without aggressive edge enhancement.

## Figure 3. RTM before and after FWI velocity updating

The reference image is migrated with the true velocity, while the two test images use the initial smoothed velocity and the model-quality-gated FWI velocity, respectively. In the 12-shot, nt=1200 comparison, the selected damped update scale is alpha=0.1; this slightly reduces the Laplacian-filtered RTM image RMSE relative to the true-velocity reference, while the full alpha=1.0 update is rejected by the model-quality gate.

## Figure 4. Scheme-2 imaging-condition comparison

Source-receiver normalization preserves the main image structure relative to source-only normalization, whereas Laplacian source-normalized imaging suppresses low-wavenumber background and highlights salt-boundary and reflector detail. Receiver illumination is shown on a log10 scale to document the full-aperture illumination distribution.
