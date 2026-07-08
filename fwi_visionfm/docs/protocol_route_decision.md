# Protocol Route Decision

## Project Route Decision

The project now keeps two routes with different roles:

- The legacy npz matrix route is retained as an engineering validation asset.
- The raw OpenFWI route is the formal research mainline.

The practical rule is simple: legacy outputs can document historical engineering coverage, but they should not be mixed into the main protocol comparison unless the split, normalization statistics, target test set, and metric definitions are verified to be identical.

## Legacy Route Summary

The legacy route is:

```text
convert_openfwi -> npz -> run_experiment_matrix
```

It includes these assets:

- `convert_openfwi`
- `convert_openfwi_multi`
- `validate_npz_dataset`
- `make_split_manifest`
- `validate_split_manifest`
- `run_experiment_matrix`
- `torch_cnn_baseline`
- `dummy_dinov2_frozen`
- `dummy_dinov2_lora`
- `subset500`, `subset2k`, `tiny16`, and cross-family matrix experiments
- Protocol v1 post-processing
- shape guard and auto-shape inference

This route was useful for early validation because it proved data conversion, small matrix execution, metric collection, shape checking, and report generation could run on CPU.

## Legacy Route Limitations

The legacy route is not the formal conclusion route for the current research stage:

- Split definitions are not fully unified across all historical runs.
- After conversion to per-sample `.npz`, sample provenance and global index tracing are weaker than in the raw route.
- Some results are smoke or tiny engineering checks.
- Some results cannot be directly compared with main route results because data splits, normalization statistics, test sets, epochs, or metric definitions differ.
- Legacy results are not used as primary research conclusions.

## Main Route Summary

The main route is:

```text
raw OpenFWI .npy -> manifest/split/stats -> run_foundation_experiment -> checkpoint-only evaluation -> report/figures/comparison
```

It includes:

- raw OpenFWI `.npy` data
- `openfwi_manifest.csv`
- `train.csv`, `val.csv`, `test_in_family.csv`, and `test_cross_family.csv`
- `train_stats.json`
- `run_foundation_experiment`
- checkpoint-only evaluation
- qualitative prediction grids
- final stage report
- real DINOv2 smoke and full fixed-test checkpoint-only evaluation
- bridge ablation
- bridge by transfer interaction
- LoRA `raw_repeat3` versus `raw_spectrogram`

This route is preferred because each sample keeps direct `data_file`, `model_file`, `local_index`, `global_index`, and `family` metadata. It also enforces train-only normalization statistics and fixed in-family/cross-family evaluation splits.

## Current Main Route Gaps

The main route is not yet a full-scale OpenFWI benchmark. Remaining gaps are:

- full OpenFWI family training
- GPU-scale real DINOv2, MAE, and SAM training
- complete cross-family benchmark
- multi-seed, large-sample, multi-epoch unified result table
- physics consistency loss
- SSIM, PSNR, and gradient metrics as first-class evaluation fields across all final reports
- attention, UMAP, and feature similarity representation analysis

## Recommended Next Priority

The recommended sequence is:

1. Complete unified metrics across the main route.
2. Run `train=500`, `val=100`, and fixed `test=100` experiments under one protocol.
3. Scale to `train=2000`.
4. When GPU is available, run real DINOv2 LoRA and Adapter training.
5. Add OOD evaluation and representation analysis.

This order keeps the protocol auditable before increasing scale or adding new foundation models.
