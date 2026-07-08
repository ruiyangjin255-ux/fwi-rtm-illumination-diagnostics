# Core innovation selection from FWI/RTM illumination references

## 1. Literature signals most relevant to the current project

### 1.1 OpenFWI benchmark logic

`OPENFWI.pdf` emphasizes diversified, rigorous and reproducible FWI benchmarking rather than a single visually attractive inversion image. Its most useful lesson for this project is not to add an underpowered deep-learning claim, but to structure the current SEG/Salt workflow as a reproducible benchmark-style diagnostic experiment with fixed data, fixed scripts, quantitative metrics and explicit failure boundaries.

### 1.2 RTM illumination compensation

Chen Shengchang, Ma Zaitian and Wu Ru-Shan show that wave-equation migration lacks the inverse Hessian action of linearized inversion, so conventional migration images can contain illumination shadows caused by acquisition aperture and complex wave-propagation paths. For the current salt model, this directly supports using source illumination, receiver illumination, source-receiver illumination and Laplacian imaging-condition comparisons as a main RTM-side contribution.

### 1.3 Two-way illumination optimized FWI

Chen Yongrui et al. connect wave-equation two-way illumination analysis with acquisition-system optimization and FWI energy compensation. The relevant idea for this project is that illumination should be treated in two layers: acquisition/RTM illumination distribution and FWI gradient/update energy distribution. This supports a bidirectional illumination diagnostic, but the current results do not yet prove that illumination preconditioning is a superior optimizer.

### 1.4 Illumination-preconditioned stepped multi-parameter FWI

Zhang Guangzhi et al. show that stepped multi-parameter inversion and illumination-preconditioned L-BFGS can reduce parameter crosstalk and balance gradient energy. The current project is still acoustic single-parameter velocity inversion, so the direct transferable idea is not multi-parameter inversion itself, but a staged workflow: first stabilize velocity and model-quality gates, then use illumination preconditioning or L-BFGS as a controlled extension.

### 1.5 FWI optimization-method comparison

Liu Yuhang et al. motivate comparing CG, L-BFGS and preconditioned variants instead of relying on one optimizer. This matches the current local-window result: adaptive line search currently contributes more than the lightweight illumination preconditioner. Therefore, the strongest FWI-side claim should be optimization diagnosis, not high-quality velocity recovery.

### 1.6 Shallow/complex-medium FWI review

Pan Yudi and Gao Lingli summarize the common FWI difficulties: strong nonlinearity, heavy computation, initial-model dependence, uncertainty and robust objective-function design. This supports conservative wording: the present FWI result is a controlled numerical diagnostic result, not a production-quality inversion method.

## 2. Best-fit core innovation for this project

### Core innovation

**A literature-guided FWI-RTM illumination diagnostic framework for salt imaging, coupling RTM illumination compensation, FWI update-quality gating, and local optimization ablation.**

This is stronger and more defensible than claiming a new FWI algorithm. The framework answers three linked questions:

1. Does FWI data-misfit reduction actually improve velocity-model structure?
2. Does a quality-gated FWI update improve downstream RTM imaging relative to an initial smoothed model?
3. Are illumination effects image-side effects, gradient/update-side effects, or acquisition/geometry effects?

The novelty is the integration and boundary control:

- RTM illumination compensation is evaluated as image regularization.
- FWI illumination/preconditioning is evaluated as update regularization.
- Two-way/source-receiver illumination is evaluated as a geometry and target-zone diagnostic.
- FWI updates are accepted only if model-structure and RTM validation metrics pass.
- Negative ablation is treated as evidence: current illumination scaling is weaker than adaptive line search.

## 3. Recommended final innovation hierarchy

### Innovation 1: Bidirectional illumination-diagnostic FWI-RTM coupling

Build a unified workflow in which source illumination, receiver illumination and source-receiver illumination are not only used for RTM display normalization but also used to explain where FWI updates are likely unreliable. This extends illumination analysis from "make RTM image brighter" to "decide whether FWI-to-RTM coupling is trustworthy."

### Innovation 2: Model-quality-gated FWI-to-RTM transfer

The current full FWI result reduces data misfit but has weak edge improvement. The proposed contribution is a rejection/acceptance mechanism: only damped updates that improve MAE/RMSE and do not degrade salt-edge metrics are passed to RTM. This converts a poor FWI image into a methodological result: residual reduction alone is insufficient.

### Innovation 3: Separation of image-side illumination compensation and inversion-side gradient preconditioning

RTM illumination normalization acts on migrated image amplitudes, while FWI illumination preconditioning acts on model update directions. The project can claim this separation clearly because it has both RTM imaging-condition diagnostics and local FWI preconditioning ablation.

### Innovation 4: Optimization-dominance ablation

The local salt-window experiments show that adaptive line search outperforms the current lightweight illumination preconditioner. This supports a practical conclusion: in the present implementation, step-length control is the primary FWI stabilizer, while illumination preconditioning should be treated as a secondary or future Hessian/L-BFGS extension.

### Innovation 5: Benchmark-style reproducibility and claim-boundary reporting

Inspired by OpenFWI, the workflow provides fixed scripts, fixed outputs, package-level readiness checks and explicit claim boundaries. This is suitable for a JGE-style engineering paper because it emphasizes reproducibility and quantitative diagnostics.

## 4. Claims that should not be used as core innovation

- Do not claim high-quality FWI velocity recovery.
- Do not claim the current illumination preconditioner is better than baseline optimization.
- Do not claim LSRTM-level imaging improvement unless LSRTM is implemented.
- Do not make DINOv2/OpenFWI or foundation-model priors a main result with only smoke-level evidence.
- Do not claim severe illumination deficit as the main limitation if full-aperture low-illumination fraction remains low.

## 5. Most suitable paper title direction

Recommended title:

**A quality-gated FWI-RTM illumination diagnostic workflow for salt-model imaging**

Alternative title:

**Coupling illumination-compensated RTM and quality-gated FWI updates for reproducible salt-model imaging diagnostics**

## 6. Minimal next experiments to strengthen the core innovation

1. Add target-zone metrics: salt-top, salt-flank and subsalt windows for illumination, RTM amplitude and FWI update energy.
2. Add gradient-energy maps before and after illumination preconditioning in the local FWI window.
3. Add one optimizer comparison table for CG, adaptive line search and illumination-preconditioned variants.
4. Add a "rejected update" panel showing why full alpha=1.0 is unsafe for RTM transfer.
5. Keep ML/OpenFWI as a future low-wavenumber-prior route unless a controlled experiment is added.
