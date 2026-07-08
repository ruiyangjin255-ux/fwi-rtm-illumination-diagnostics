# Integration Guide — PASD-FWI v0.2 into Existing `fwi_visionfm`

## Integration boundary

The v0.2 package is additive:

```text
fwi_visionfm/pasd/
```

It should initially remain separate from the existing V2/V3/V4 runners. PASD uses the same scientific principles already established in the project: fixed manifests, train-only statistics, strict source/target isolation, saved prediction IDs, and paired bootstrap.

## Recommended merge order

### 1. Add the package and collect tests

```powershell
Copy-Item -Recurse .\src\fwi_visionfm\pasd D:\ryjin\fwi_visionfm\fwi_visionfm\pasd
cd D:\ryjin\fwi_visionfm
$env:PYTHONPATH = "$PWD"
python -m pytest -q tests -k pasd
```

### 2. Treat the PASD protocol JSON as the authoritative split source

Use `make_protocol.py` once to produce a committed JSON manifest. Do not use ad-hoc random splits for the scientific matrix after the smoke stage.

Mapping to the existing project:

| Existing capability | PASD v0.2 hook |
|---|---|
| raw `.npy` OpenFWI arrays | `DatasetRef` / `ArrayBundle` |
| fixed manifest and split | `ProtocolManifest` |
| train-only stats | `VelocityScaler.fit(source[train])` |
| bridge registry | `B1/B2/B4` bridge mode selection |
| decoder registry | `PlainVelocityDecoder` and `BackgroundEdgeDecoder` |
| experiment matrix | `run_protocol.py` |
| prediction alignment | `predictions_{split}.npz` with `sample_id` |
| paired bootstrap | `bootstrap/` B4-vs-B1 JSON outputs |
| visualization layer | `compare_variants.py` common-sample renderer |

### 3. Preserve existing source and target split semantics

PASD’s source split uses `train`, `val`, and `in_family_test`. The target family contributes only `cross_family_test`. Target data must not affect:

- scaler fitting;
- training batches;
- validation batches;
- checkpoint selection;
- bridge/hyperparameter selection.

### 4. First formal matrix

Start with the fixed four rows only:

```text
B1_raw_unet
B2_hybrid_unet
B3_raw_bed
B4_pasd_fwi
```

Do not add DINOv2, MAE, SAM, NCS, PDE consistency, or a large search grid until this B1–B4 matrix completes with aligned predictions and reports.

### 5. Link formal reports

After the PASD runs complete, the existing visual-score / reporting process can consume `predictions_in_family.npz` and `predictions_cross_family.npz` directly. The per-sample IDs allow the existing paired-bootstrap convention to be reused without retraining.

## Deliberate v0.2 scope

Implemented:

- physical attribute bridge;
- optional source/receiver coordinate ingestion;
- shared shot encoder;
- mean and geometry-aware attention fusion;
- plain and decoupled decoders;
- plain and structure-aware losses;
- fixed protocol construction;
- cross-family isolation;
- multi-seed matrix runner;
- bootstrap/report/figure generation.

Deferred:

- differentiable acoustic forward modeling;
- full OpenFWI scale training;
- real DINOv2/MAE/NCS backbone integration;
- target-family model selection;
- new acquisition geometry beyond the supplied array conventions.
