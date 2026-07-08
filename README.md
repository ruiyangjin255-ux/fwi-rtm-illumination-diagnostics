# Quality-gated FWI-RTM illumination diagnostics for salt-model imaging

This repository contains a lightweight, publication-oriented code and result package extracted from a larger local SEG/Salt FWI-RTM experiment.

The core contribution is not a claim of high-quality FWI velocity recovery. The defensible contribution is a reproducible diagnostic framework that couples:

- FWI model-quality gates;
- RTM before/after validation;
- source, receiver and source-receiver illumination diagnostics;
- target-zone salt-top, salt-flank and subsalt metrics;
- local FWI optimizer and illumination-preconditioner ablation;
- conservative claim-boundary reporting for a JGE-style manuscript.

## Repository layout

```text
rtm_acoustic/                         Python modules and tests
docs/jge_revision/                    Result tables and manuscript-support reports
docs/jge_main_figures/                Lightweight paper figures: PNG/PDF/SVG
```

Large local simulation outputs are intentionally excluded:

- `outputs/`
- raw SEG/Salt velocity binaries;
- `.npy`, `.dat`, `.bin`, checkpoint files;
- submission TIFF figures.

The excluded files are required to rerun the full local experiment, but the code and included CSV/figure outputs are sufficient to inspect the framework and manuscript-supporting results.

## Main framework modules

```powershell
python -m rtm_acoustic.build_jge_method_synthesis
python -m rtm_acoustic.build_target_zone_illumination_diagnostics
python -m rtm_acoustic.make_jge_main_figures
python -m rtm_acoustic.build_jge_submission_package
```

The most important new module is:

```text
rtm_acoustic/build_target_zone_illumination_diagnostics.py
```

It derives salt-top, salt-flank and subsalt-shadow zones from the SEG/Salt high-velocity body, then evaluates illumination, RTM image response and FWI update energy over the same target zones.

## Key result

The current FWI update should be treated conservatively. The target-zone diagnostics show that the subsalt zone has weaker source-receiver illumination and weaker RTM response, while the current gated FWI update contributes negligible update energy there. This supports a quality-gated illumination diagnostic paper, not an overclaim of production-grade FWI velocity recovery.

## Verification used in the source workspace

The source workspace passed the focused FWI/RTM/JGE checks:

```text
14 passed
```

Some tests in `rtm_acoustic/tests` expect the excluded local outputs to exist. They are retained to document the validation surface used in the full workspace.

## Suggested paper title

```text
A quality-gated FWI-RTM illumination diagnostic workflow for salt-model imaging
```

