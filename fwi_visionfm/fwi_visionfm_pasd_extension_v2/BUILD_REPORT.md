# Cloud Build Report — PASD-FWI v0.2

## Build status

- Extension package created under `src/fwi_visionfm/pasd/`.
- Existing v0.1 extension left unchanged; v0.2 is delivered as a separate package.
- Unit tests: **8 passed**.
- Python compilation: passed.
- Single-run CPU smoke: **SUCCESS** for `B4_pasd_fwi` on an intentionally tiny synthetic set.
- Fixed source/cross-family protocol smoke: **SUCCESS** for `B1_raw_unet` and `B4_pasd_fwi`, including report, aligned archives, paired bootstrap JSON files, and common-sample figures.
- No local files from `D:\ryjin\fwi_visionfm` were modified because that project is not mounted in this cloud runtime.

## New v0.2 modules

```text
fwi_visionfm/pasd/
├── bridge.py               # raw/envelope/band attributes and geometry vectors
├── model.py                # raw/hybrid controls, plain/decoupled decoders, shot fusion
├── losses.py               # L1 and background-edge objectives
├── data.py                 # NPY data, stable sample IDs, optional geometry arrays
├── protocol.py             # strict fixed source/cross-family manifest
├── make_protocol.py        # manifest creation CLI
├── experiment.py           # shared train/evaluate engine
├── run_experiment.py       # single-variant runner
├── run_protocol.py         # B1–B4 multi-seed runner
├── bootstrap.py            # aligned paired-bootstrap implementation
├── reporting.py            # protocol aggregation and report generation
├── compare_variants.py     # common-sample paper figure renderer
├── metrics.py              # physical-unit numerical/structural metrics
└── plotting.py             # bridge, velocity, profile, gradient, and report plots
```

## Smoke-run boundary

The smoke outputs verify only that all code paths execute and produce consistent artifacts. Their numerical values are not scientific results and must not be used as evidence of PASD superiority.

## Definition of done for the next local phase

1. Copy v0.2 package into the project without replacing existing protocol modules.
2. Create one committed FlatVel-A → CurveVel-A protocol JSON from the project’s frozen manifests.
3. Execute B1–B4 for seeds 0/1/2 under identical CPU settings.
4. Inspect `PROTOCOL_REPORT.md`, paired bootstrap files, and common-sample figures.
5. Expand to FlatFault-A/CurveFault-A only after B1–B4 produces stable source/cross-family artifacts.
