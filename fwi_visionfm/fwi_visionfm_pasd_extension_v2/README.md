# PASD-FWI v0.2 — Protocol-Ready Extension for `fwi_visionfm`

This package is a **non-destructive extension** for the existing `fwi_visionfm` project. It implements a reproducible, CPU-oriented foundation for the proposed **Physics-Aligned Attribute–Structure Decoupled FWI (PASD-FWI)** study.

The package is intentionally isolated under `fwi_visionfm.pasd`; it does not replace the existing V2/V3/V4 code, bridge registry, decoder registry, manifests, historical results, or reporting scripts.

## What v0.2 adds

### Method modules

- **B1–B4 ablation registry** with genuinely distinct configurations:
  - `B1_raw_unet`: raw gather + mean pooling + plain decoder + L1;
  - `B2_hybrid_unet`: raw/envelope/band bridge + mean pooling + plain decoder + L1;
  - `B3_raw_bed`: raw gather + mean pooling + background-edge decoder/loss;
  - `B4_pasd_fwi`: hybrid bridge + geometry-aware attention + background-edge decoder/loss.
- **Physics-aligned hybrid bridge**: robust raw gather, Hilbert envelope, lightweight low/high temporal band-energy proxy.
- **Geometry-aware shot aggregation**: uses source coordinate, shot index, and mean normalized offset. Optional `.npy` source/receiver coordinates are supported.
- **Background-edge decoder**: separately predicts smoothed background velocity and bounded structural residual.
- **Background-edge loss**: global velocity L1 + background target fidelity + target-edge-gated gradient consistency + non-edge smoothness.

### Protocol and evidence modules

- Frozen JSON **source/in-family/cross-family manifests** with explicit split leakage checks.
- Strict **train-only target scaling**: target-family velocity labels are never used to fit the velocity scaler.
- Multi-seed B1–B4 matrix runner.
- Per-sample `sample_id` prediction archives for both in-family and cross-family evaluation.
- Aligned paired bootstrap comparison of B4 versus B1 from saved archives, with no re-training.
- Protocol aggregation, mean ± standard deviation reports, and metric plots.
- Common-sample comparison renderer using a fixed `median_mae`, `best_mae`, or `worst_mae` selection policy after strict `sample_id` alignment.

## Data contract

```text
records:    [N, S, T, R]      e.g. [N, 5, 1000, 70]
velocities: [N, H, W] or [N, 1, H, W]   e.g. [N, 70, 70]
```

Optional acquisition geometry arrays:

```text
source_positions:   [S] or [N, S]
receiver_positions: [R], [N, R], or [N, S, R]
```

When geometry arrays are unavailable, the bridge uses a deterministic normalized shot index and receiver grid. This preserves a valid control experiment while making the fallback explicit in the experiment metadata.

## Installation into the local project

Copy the extension package into the existing repository:

```text
fwi_visionfm_pasd_extension_v2/src/fwi_visionfm/pasd/
    -> D:\ryjin\fwi_visionfm\fwi_visionfm\pasd\
```

Then, from the project root:

```powershell
$env:PYTHONPATH = "$PWD"
python -m pytest -q tests -k pasd
```

## 1. CPU synthetic smoke

```powershell
python -m fwi_visionfm.pasd.run_experiment `
  --synthetic `
  --synthetic-shots 3 --synthetic-time 64 --synthetic-receivers 24 --synthetic-model-size 32 `
  --max-samples 24 --train-size 14 --val-size 5 --test-size 5 `
  --epochs 1 --batch-size 2 `
  --base-channels 4 --latent-channels 16 --latent-height 4 --latent-width 4 `
  --lowpass-kernel 7 --torch-threads 1 `
  --variant B4_pasd_fwi `
  --output outputs\pasd_v2_smoke
```

## 2. Build a fixed OpenFWI cross-family protocol

```powershell
python -m fwi_visionfm.pasd.make_protocol `
  --source-records D:\data\openfwi\flatvel_a_subset500\records.npy `
  --source-models D:\data\openfwi\flatvel_a_subset500\velocity.npy `
  --source-family FlatVel-A `
  --target-records D:\data\openfwi\curvevel_a_subset500\records.npy `
  --target-models D:\data\openfwi\curvevel_a_subset500\velocity.npy `
  --target-family CurveVel-A `
  --train-size 350 --val-size 75 --in-family-test-size 75 --cross-family-test-size 75 `
  --seed 0 `
  --output protocols\pasd_flat_to_curve_seed0.json
```

If acquisition coordinates are available, append:

```powershell
  --source-positions D:\data\openfwi\flatvel_a_subset500\source_positions.npy `
  --receiver-positions D:\data\openfwi\flatvel_a_subset500\receiver_positions.npy `
  --target-source-positions D:\data\openfwi\curvevel_a_subset500\source_positions.npy `
  --target-receiver-positions D:\data\openfwi\curvevel_a_subset500\receiver_positions.npy
```

## 3. Run B1–B4 under the same protocol

```powershell
python -m fwi_visionfm.pasd.run_protocol `
  --protocol protocols\pasd_flat_to_curve_seed0.json `
  --output outputs\pasd_flat_to_curve_v2 `
  --variants B1_raw_unet B2_hybrid_unet B3_raw_bed B4_pasd_fwi `
  --seeds 0 1 2 `
  --epochs 3 --batch-size 4 `
  --base-channels 16 --latent-channels 96 `
  --torch-threads 1 `
  --bootstrap-resamples 2000
```

## 4. Render a fixed aligned publication comparison

```powershell
python -m fwi_visionfm.pasd.compare_variants `
  --protocol-root outputs\pasd_flat_to_curve_v2 `
  --variants B1_raw_unet B2_hybrid_unet B3_raw_bed B4_pasd_fwi `
  --seed 0 --split cross_family --selection median_mae
```

## Key outputs

```text
outputs/pasd_flat_to_curve_v2/
├── protocol_manifest.json
├── B1_raw_unet/seed_0/
│   ├── history.csv
│   ├── metrics_summary.json
│   ├── metrics_in_family_per_sample.csv
│   ├── metrics_cross_family_per_sample.csv
│   ├── predictions_in_family.npz
│   ├── predictions_cross_family.npz
│   ├── checkpoint.pt
│   └── figures/
├── ... B2/B3/B4 ...
├── bootstrap/
│   └── B4_vs_B1_seed*_cross_family_*.json
├── protocol_runs.csv
├── protocol_summary.csv
├── PROTOCOL_REPORT.md
├── figures/
└── comparison/
```

## Result-interpretation guardrails

1. The synthetic smoke only verifies execution; it is **not evidence of scientific advantage**.
2. A final PASD claim requires fixed manifests, at least three seeds, and in-family plus cross-family evidence.
3. Numerical MAE/RMSE must be interpreted with SSIM, edge MAE, gradient error, and common-sample figures.
4. Do not select visually favorable samples manually. Use `median_mae`, `best_mae`, `worst_mae`, or an archived explicit `sample_id` rule.
5. The current version does not include PDE consistency loss, a foundation-model backbone, or target-family tuning. Those are follow-up modules after B1–B4 stabilize.
