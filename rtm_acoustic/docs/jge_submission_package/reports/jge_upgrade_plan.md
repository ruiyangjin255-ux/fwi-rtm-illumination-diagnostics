# JGE-oriented Upgrade Plan

## Journal fit

Journal of Geophysics and Engineering papers should be written as focused geophysical or engineering-geophysics studies, with a clear method, reproducible data processing path, quantitative results, and concise discussion. For a regular Paper, keep the manuscript within roughly 8000 words, 250-word abstract, up to 5 keywords, up to 10 figures/tables, and a compact reference list.

## Current readiness verdict

The current FWI effect is not yet strong enough to claim a high-quality inversion result. The most defensible JGE positioning is therefore not "a superior FWI algorithm", but "a reproducible diagnostic framework showing how FWI updates, RTM illumination normalization, imaging-condition selection, and line-search behavior interact in a salt model".

## Required result upgrades before submission

1. Use the structural FWI model-quality table as a gate: velocity MAE/RMSE, salt-boundary edge error, gradient error, update localization, and update/error correlation must be reported with the misfit curve.
2. Run the RTM-before/after-FWI comparison as a formal result: RTM with initial smoothed velocity versus RTM with inverted velocity, measured by reflector continuity, image focus, or objective image metrics.
3. Extend full-model FWI beyond 3 iterations only if misfit and model metrics improve together; otherwise keep the short run as a diagnostic case.
4. Convert smoke-level scheme2 results into a formal run or explicitly label it as imaging-condition diagnostic.
5. Keep ML/OpenFWI results out of the main claim unless a controlled initialization experiment is added.

## Recommended paper title

Diagnostic coupling of acoustic full-waveform inversion and reverse-time migration illumination analysis for salt-model imaging

## Recommended core figures

1. Workflow and reproducibility map.
2. Full-model FWI misfit and velocity update.
3. Initial-velocity RTM versus FWI-updated-velocity RTM.
4. RTM imaging-condition comparison.
5. Local-window line-search and illumination-preconditioner ranking.
6. Failure/limitation figure: where lightweight illumination preconditioning underperforms.

## Recommended claims

- Claim: The workflow provides a reproducible diagnostic bridge between FWI velocity updating and RTM imaging-condition assessment.
- Claim: For the current full-aperture salt RTM setup, source-receiver normalization is close to source-only normalization, while Laplacian filtering changes the spectral content more strongly.
- Claim: In the local-window FWI ablation, adaptive line search contributes more to misfit reduction than the current lightweight source-illumination preconditioner.
- Do not claim: The current FWI result has reached production-quality velocity recovery.
- Do not claim: The current illumination preconditioner improves FWI over baseline.
- Do not claim: The current ML foundation-model branch is a validated JGE-level contribution.

## Program framework

```text
rtm_acoustic/
  run_full_salt_fwi.py                  existing
  run_multishot_rtm.py                  existing
  run_scheme2_imaging_condition_compare.py existing
  build_jge_core_results.py             added summary/ranking generator
  evaluate_fwi_model_quality.py         added model MAE/RMSE/edge metrics
  optimize_fwi_update_scale.py          added damped update gate before RTM
  run_rtm_before_after_fwi.py           added RTM with initial vs inverted velocity
  run_optimized_fwi_rtm_pipeline.py     added recommended one-command pipeline
  build_jge_submission_package.py       added figures, captions, tables, checklist
```
