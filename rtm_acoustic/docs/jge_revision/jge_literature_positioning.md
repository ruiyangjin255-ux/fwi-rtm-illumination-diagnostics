# JGE literature positioning and submission-oriented content ranking

## 1. JGE positioning

`Journal of Geophysics and Engineering` is a suitable target only if the paper is framed as a focused geophysical engineering workflow with reproducible modelling, quantitative diagnostics, and clear limitations. The current FWI images are not yet strong enough for a "high-quality FWI result" claim. The safer JGE-facing angle is:

> A reproducible acoustic FWI-RTM diagnostic workflow for salt-model imaging, showing how velocity updating, RTM illumination normalization, imaging-condition selection, and line-search strategy interact.

This is narrower than "new FWI algorithm", but it is closer to what the current evidence can support.

## 2. Similar JGE articles found through Crossref metadata

### FWI-related JGE papers

| Year | Article | DOI | Relevance to this project |
|---:|---|---|---|
| 2024 | Full waveform inversion with smoothing of dilated convolutions | https://doi.org/10.1093/jge/gxae097 | JGE accepts FWI work only when the method improves inversion stability or structure recovery, not just when a misfit curve decreases. |
| 2024 | Inexact augmented Lagrangian method-based full-waveform inversion with randomized singular value decomposition | https://doi.org/10.1093/jge/gxae015 | Shows that optimization strategy and convergence behaviour are valid JGE topics; our line-search evidence should be expanded. |
| 2024 | Improving full-waveform inversion based on sparse regularization for geophysical data | https://doi.org/10.1093/jge/gxae036 | Indicates that regularization/constraint design is a common FWI contribution route. |
| 2023 | An efficient plug-and-play regularization method for full waveform inversion | https://doi.org/10.1093/jge/gxad073 | Supports a future route using learned or denoising priors, but our `fwi_visionfm` branch is not yet strong enough for the main claim. |
| 2021 | A high-order total-variation regularisation method for full-waveform inversion | https://doi.org/10.1093/jge/gxab010 | Highlights that structural preservation metrics are expected when claiming better inversion. |
| 2019 | Multi-source multi-scale source-independent full waveform inversion | https://doi.org/10.1093/jge/gxz013 | Supports adding multi-scale or source-independent extensions if the current FWI is upgraded. |

### RTM and LSRTM-related JGE papers

| Year | Article | DOI | Relevance to this project |
|---:|---|---|---|
| 2026 | Adaptive Wasserstein Distance-driven Least-squares Reverse Time Migration with Mini-batch Strategy | https://doi.org/10.1093/jge/gxag083 | JGE accepts LSRTM work with optimization and objective-function innovation; our current RTM is diagnostic, not LSRTM. |
| 2024 | Reverse time migration surface offset gathers by attribute migration and constrained least-squares inversion | https://doi.org/10.1093/jge/gxad097 | Shows the need for gather/attribute or constrained inversion evidence for stronger RTM claims. |
| 2022 | Efficient least-squares reverse time migration using local cross-correlation imaging condition | https://doi.org/10.1093/jge/gxac027 | Directly relevant to our imaging-condition comparison; stronger submission would implement local cross-correlation/LSRTM baseline. |
| 2017 | Cross-correlation least-squares reverse time migration in the pseudo-time domain | https://doi.org/10.1088/1742-2140/aa6b33 | Supports the importance of cross-correlation imaging conditions beyond simple display comparison. |
| 2016 | Improving the gradient in least-squares reverse time migration | https://doi.org/10.1088/1742-2132/13/2/172 | Relevant to our finding that update direction and gradient conditioning matter. |
| 2013 | Wavefield reconstruction methods for reverse time migration | https://doi.org/10.1088/1742-2132/10/1/015004 | Relevant to computational framework and wavefield storage/reconstruction discussion. |
| 2012 | Reverse time migration with source wavefield reconstruction strategy | https://doi.org/10.1088/1742-2132/9/1/008 | Supports wavefield-memory/reconstruction discussion in our RTM implementation. |

## 3. Content ranking for the revised paper

| Rank | Content block | Keep as main text? | Why |
|---:|---|---|---|
| 1 | FWI-RTM diagnostic framework and reproducible program pipeline | Yes | Most defensible contribution with current code/results. |
| 2 | Full-model CG vs P-CG FWI misfit ranking | Yes, but carefully | Useful optimization evidence; not enough alone for inversion-quality claim. |
| 3 | Local-window adaptive line-search ranking | Yes | Strongest mechanistic result: line search beats lightweight illumination preconditioning. |
| 4 | RTM imaging-condition quantitative ranking | Yes | Converts image comparison into measurable evidence. |
| 5 | Lightweight illumination preconditioner negative result | Yes | Honest limitation; helps avoid overclaiming. |
| 6 | ML/OpenFWI/foundation-model extension | No, discussion only | Current branch is smoke/small-sample; not ready for JGE core claim. |

## 4. Innovation content after JGE filtering

### Innovation A: diagnostic coupling rather than standalone FWI

The paper should claim a coupled diagnostic workflow, not a new high-performance FWI algorithm. This aligns with the current evidence: full FWI reduces misfit, RTM condition metrics expose illumination/imaging-condition behaviour, and local experiments explain why line search matters.

### Innovation B: evidence-based separation of RTM illumination normalization and FWI preconditioning

Current results support this separation:

- RTM normalization acts on the migration image.
- FWI preconditioning acts on the model update direction.
- The lightweight illumination preconditioner does not outperform baseline with adaptive line search.

### Innovation C: negative ablation as methodological boundary

JGE reviewers may accept a negative ablation if it is used to clarify method boundaries. The current result should be phrased as:

> In this implementation, source-illumination scaling is insufficient as a primary FWI optimizer; adaptive step selection dominates the local-window convergence behaviour.

### Innovation D: quantitative RTM condition ranking

The RTM part should not be written as "the figure looks better". Use:

- correlation between source-only and source-receiver normalization;
- 99th percentile amplitude;
- low-illumination fraction;
- Laplacian spectral/background change.

## 5. Program framework required before serious submission

| Priority | New module | Purpose | Status |
|---:|---|---|---|
| 1 | `evaluate_fwi_model_quality.py` | Compute velocity MAE/RMSE, salt-edge error, gradient error, update localization | Required next |
| 2 | `run_rtm_before_after_fwi.py` | Run RTM using initial model and FWI-updated model with identical acquisition | Required next |
| 3 | `build_jge_core_results.py` | Generate result rankings from existing outputs | Added |
| 4 | `build_jge_submission_package.py` | Collect figures, captions, tables, data statement and checklist | Later |
| 5 | `run_lsrtm_baseline.py` | Stronger RTM contribution if time allows | Optional but high value |

## 6. Immediate next experiments

1. Do not run more long FWI blindly. First compute model-quality metrics for current runs.
2. If the current FWI model update worsens velocity MAE/RMSE or salt-edge error, do not present it as improved velocity recovery.
3. Run RTM with the initial smoothed velocity and with the FWI-updated velocity under identical RTM parameters.
4. Compare RTM focus/continuity/objective image metrics before and after FWI.
5. Only then decide whether to extend FWI iterations or redesign the objective/regularization.

## 7. Current core outputs

Generated files:

- `full_fwi_ranking.csv`
- `local_fwi_strategy_ranking.csv`
- `rtm_imaging_condition_metrics.csv`
- `innovation_ranking.csv`
- `jge_upgrade_plan.md`

These are generated by:

```powershell
python -m rtm_acoustic.build_jge_core_results
```
