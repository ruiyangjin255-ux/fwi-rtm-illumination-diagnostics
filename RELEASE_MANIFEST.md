# Release Manifest

## Project Identity

- Repository: `ruiyangjin255-ux/fwi-rtm-illumination-diagnostics`
- Current project: ADMIT-FWI update admissibility audit framework
- Primary implementation path: `rtm_acoustic/`

## Included in Git

- ADMIT-FWI/RTM/FWI source code under `rtm_acoustic/`.
- Audit and production configurations under `rtm_acoustic/configs/`.
- Diagnostic modules under `rtm_acoustic/diagnostics/`.
- Reproducible run scripts under `rtm_acoustic/scripts/`.
- Manuscript-facing documentation and figure metadata under `docs/` and `rtm_acoustic/docs/`.

## Excluded from Git

- `rtm_acoustic/outputs/`
- `paper_rewriting_output/`
- `.external/`
- generated `*.npy`, `*.bin`, `*.dat`, `*.tiff`, logs, wavefield memmaps, Python caches, and compiled local binaries

These files are local experiment products. They should be regenerated from the committed code/configs or stored in a dedicated data release, not mixed into the source repository.

## Key Commands

```powershell
python -m py_compile rtm_acoustic\run_full_salt_fwi.py rtm_acoustic\scripts\run_deep_wavefield_smoke.py
python rtm_acoustic\scripts\run_deep_wavefield_smoke.py --config rtm_acoustic\configs\deep_time_preflight_pml_pad_v1.yaml --shots 3
powershell -ExecutionPolicy Bypass -File rtm_acoustic\scripts\run_deep_time_multiscale_fwi_production.ps1
```

## Current Evidence Status

The framework supports a conservative interpretation: shallow and upper-salt full-FWI updates must be audited before being accepted as geologically meaningful, and strongly suppressed ECG-gated updates should be reported as evidence of insufficient admissibility rather than hidden or over-claimed.
