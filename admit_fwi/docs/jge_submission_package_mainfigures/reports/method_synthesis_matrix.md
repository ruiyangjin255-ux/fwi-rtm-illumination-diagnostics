# Literature-guided method synthesis matrix

This matrix reframes the paper away from a weak standalone FWI claim and toward a defensible integrated FWI-RTM diagnostic workflow.

| Literature direction | Current problem | Implemented response | Paper role |
|---|---|---|---|
| Low-frequency and low-wavenumber FWI recovery | The full salt FWI reduces data misfit but does not recover a sharper salt boundary. | Do not claim high-quality FWI; add model-quality and edge-error gates before RTM. | claim boundary and quality-control mechanism |
| Regularized or constrained FWI | Unregularized updates mainly improve non-edge MAE and can degrade gradient/edge metrics. | Rank alpha candidates by MAE, RMSE, edge MAE, gradient MAE and reject harmful updates. | diagnostic update-scale selection |
| Optimization and step-length control | Simple illumination scaling is weaker than adaptive step selection in the local salt window. | Use local-window line-search ablation as mechanistic evidence. | strongest FWI-side experimental evidence |
| RTM/LSRTM imaging-condition improvement | The current code is RTM diagnostics, not LSRTM or high-resolution reflectivity inversion. | Report source, source-receiver, Laplacian and illumination metrics without claiming LSRTM. | RTM-side quantitative imaging-condition comparison |
| Reproducible benchmark and data-driven FWI | The ML/foundation-model branch is smoke-level and cannot support the main result. | Keep ML priors as a future extension; keep this paper physics-first and reproducible. | future work only |

## Revised innovation statement

The manuscript should not claim a new high-performance FWI algorithm. Its defensible contribution is a literature-guided, reproducible diagnostic workflow that combines FWI misfit reduction, model-structure quality gates, update-scale rejection, RTM before/after validation, imaging-condition diagnostics and local optimizer ablation.

## Source links used for positioning

- JGE SDCI FWI: https://academic.oup.com/jge/article/21/6/1594/7762962
- JGE LSRTM local cross-correlation: https://academic.oup.com/jge/article/19/3/376/6597025
- JGE LSRTM gradient improvement: https://academic.oup.com/jge/article/13/2/172/5113404
- JGE adaptive Wasserstein LSRTM: https://academic.oup.com/jge/advance-article/doi/10.1093/jge/gxag083/8708462
- Virieux and Operto FWI overview: https://doi.org/10.1190/1.3238367
- OpenFWI benchmark: https://proceedings.neurips.cc/paper_files/paper/2022/hash/27d3ef263c7cb8d542c4f9815a49b69b-Abstract-Datasets_and_Benchmarks.html
- DINOv2: https://arxiv.org/abs/2304.07193
