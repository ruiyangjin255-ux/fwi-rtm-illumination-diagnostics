# Release Manifest

## Project Identity

- Current repository: `ruiyangjin255-ux/fwi-rtm-illumination-diagnostics`
- Recommended repository name: `admit-fwi-update-admissibility-audit`
- Current project: ADMIT-FWI update admissibility audit framework
- Primary implementation path: `admit_fwi/`

## Included in Git

- ADMIT-FWI/RTM/FWI source code under `admit_fwi/`.
- Audit and production configurations under `admit_fwi/configs/`.
- Diagnostic modules under `admit_fwi/diagnostics/`.
- Reproducible run scripts under `admit_fwi/scripts/`.
- Manuscript-facing documentation and figure metadata under `docs/` and `admit_fwi/docs/`.

## Excluded from Git

- `admit_fwi/outputs/`
- `paper_rewriting_output/`
- `.external/`
- legacy local project directories such as `learning-based FWI extension/` and `external_wavefield_models/`
- generated `*.npy`, `*.bin`, `*.dat`, `*.tiff`, logs, wavefield memmaps, Python caches, and compiled local binaries

These files are local experiment products. They should be regenerated from the committed code/configs or stored in a dedicated data release, not mixed into the source repository.

## Key Commands

```powershell
python -m py_compile admit_fwi\run_full_salt_fwi.py admit_fwi\scripts\run_deep_wavefield_smoke.py
python admit_fwi\scripts\run_deep_wavefield_smoke.py --config admit_fwi\configs\deep_time_preflight_pml_pad_v1.yaml --shots 3
powershell -ExecutionPolicy Bypass -File admit_fwi\scripts\run_deep_time_multiscale_fwi_production.ps1
```

## Current Evidence Status

The framework supports a conservative interpretation: shallow and upper-salt full-FWI updates must be audited before being accepted as geologically meaningful, and strongly suppressed ECG-gated updates should be reported as evidence of insufficient admissibility rather than hidden or over-claimed.
