"""PASD Phase-4 paper freeze and reproducibility package."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .diagnostics import sha256_file
from .phase3_utils import write_json


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _git(cwd: Path, args: list[str]) -> str:
    completed = subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)
    return (completed.stdout or completed.stderr).strip()


def _hash_files(root: Path, patterns: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    for pattern in patterns:
        for path in sorted(root.rglob(pattern)):
            if path.is_file():
                rows.append({"path": str(path), "relative_path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return rows


def _copy_tree_files(source: Path, dest: Path, suffixes: tuple[str, ...]) -> list[dict[str, Any]]:
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in sorted(source.iterdir()):
        if path.is_file() and path.suffix.lower() in suffixes:
            target = dest / path.name
            shutil.copyfile(path, target)
            copied.append({"source": str(path), "dest": str(target), "sha256": sha256_file(target)})
    return copied


def _improvement(b1: float, pasd: float, higher_is_better: bool = False) -> float:
    if higher_is_better:
        return pasd - b1
    return b1 - pasd


def _build_results_summary(phase3r_root: Path, out: Path) -> list[dict[str, Any]]:
    rows = _read_csv(phase3r_root / "corrected_metrics" / "corrected_summary_across_seeds.csv")
    by_key = {(row["dataset"], row["variant"]): row for row in rows}
    summary = []
    for dataset in ("in_family", "cross_curvevel_a", "cross_flatfault_a"):
        b1 = by_key[(dataset, "B1_raw_unet")]
        pasd = by_key[(dataset, "PASD_Core_locked")]
        summary.append(
            {
                "dataset": dataset,
                "B1_MAE": b1["MAE"],
                "PASD_Core_MAE": pasd["MAE"],
                "MAE_reduction": _improvement(float(b1["MAE"]), float(pasd["MAE"])),
                "B1_RMSE": b1["RMSE"],
                "PASD_Core_RMSE": pasd["RMSE"],
                "RMSE_reduction": _improvement(float(b1["RMSE"]), float(pasd["RMSE"])),
                "B1_SSIM": b1["SSIM"],
                "PASD_Core_SSIM": pasd["SSIM"],
                "SSIM_gain": _improvement(float(b1["SSIM"]), float(pasd["SSIM"]), higher_is_better=True),
                "B1_source_threshold_edge_MAE": b1["source_threshold_edge_MAE"],
                "PASD_Core_source_threshold_edge_MAE": pasd["source_threshold_edge_MAE"],
                "source_threshold_edge_MAE_reduction": _improvement(float(b1["source_threshold_edge_MAE"]), float(pasd["source_threshold_edge_MAE"])),
                "B1_gradient_l1_edge": b1["gradient_l1_edge"],
                "PASD_Core_gradient_l1_edge": pasd["gradient_l1_edge"],
                "gradient_l1_edge_reduction": _improvement(float(b1["gradient_l1_edge"]), float(pasd["gradient_l1_edge"])),
                "B1_edge_F1": b1["edge_F1"],
                "PASD_Core_edge_F1": pasd["edge_F1"],
                "edge_F1_gain": _improvement(float(b1["edge_F1"]), float(pasd["edge_F1"]), higher_is_better=True),
            }
        )
    _write_csv(out, summary)
    return summary


def _write_manuscript_draft(root: Path, summary: list[dict[str, Any]]) -> None:
    methods = """# Methods Draft

## PASD-Core architecture and training protocol

PASD-Core was evaluated as a frozen paper model after source-only aggregation selection. The model combines a physical-attribute hybrid gather bridge with geometry-aware aggregation and a background-edge decoupled velocity decoder. All model-selection decisions were made on the FlatVel-A source domain before target evaluation. CurveVel-A and FlatFault-A were reserved as target-family evaluation sets and were not used for training, validation, threshold fitting, scaler fitting, or model selection.

## Source-only locking and corrected structural metrics

The source velocity scaler was fitted only on FlatVel-A source-train samples. Edge-region metrics used a fixed source-trained threshold applied as a strict mask, `gradient_magnitude(v_true_physical) > tau_source`, after inverse transformation to physical velocity space. Prediction edge masks used a source-validation threshold and one-pixel tolerance for edge precision, recall, and F1. Deprecated Phase-3 archive fields, including legacy `edge_MAE` and `gradient_error`, were excluded from all Phase-4 tables and figures.

## Statistical comparison

B1_raw_unet and PASD-Core were trained from scratch for three seeds under the locked protocol. Prediction archives were aligned by sample identifier before paired bootstrap analysis. Bootstrap comparisons were performed separately for CurveVel-A and FlatFault-A and were not pooled across target families.
"""
    experiments = """# Experiments Draft

## Cross-family target evaluation

The main experiment trained both B1_raw_unet and PASD-Core on FlatVel-A and evaluated them on two independent target families, CurveVel-A and FlatFault-A. This design tested whether the PASD-Core inductive bias could transfer across target families without target-driven model selection.

## Corrected metric repair and provenance control

Phase-3R repaired the metric provenance by recomputing all formal metrics directly from fresh prediction archives. The repair verified archive alignment, rejected deprecated structural fields, and enforced a source-threshold strict edge mask. All masked-MAE identity checks passed, supporting the consistency of the corrected edge and non-edge summaries.

## Baseline and statistical controls

The B1_raw_unet baseline and PASD-Core were compared at matched seeds and matched sample identifiers. Paired bootstrap intervals were computed per target and seed for MAE, RMSE, SSIM, source-threshold edge MAE, edge-gradient error, and edge-F1.
"""
    result_lines = [
        "# Results Draft",
        "",
        "## PASD-Core reduced cross-family velocity error",
        "",
        "PASD-Core reduced overall velocity error relative to B1_raw_unet across FlatVel-A in-family evaluation and both cross-family targets. The corrected Phase-3R summary showed lower MAE and RMSE for PASD-Core on CurveVel-A and FlatFault-A, together with higher SSIM.",
        "",
        "## Source-threshold edge metrics supported target-dependent regional gains",
        "",
        "Using the source-trained fixed threshold and strict edge mask, PASD-Core reduced source-threshold edge MAE on both CurveVel-A and FlatFault-A. On CurveVel-A, edge-gradient error improved modestly, whereas edge-F1 increased from a low baseline. These results support improved boundary-region numerical consistency and boundary localization, but not complete high-wavenumber recovery.",
        "",
        "## Corrected target summaries",
        "",
        "| dataset | MAE reduction | RMSE reduction | SSIM gain | edge MAE reduction | edge-gradient reduction | edge-F1 gain |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        result_lines.append(
            f"| {row['dataset']} | {float(row['MAE_reduction']):.6g} | {float(row['RMSE_reduction']):.6g} | "
            f"{float(row['SSIM_gain']):.6g} | {float(row['source_threshold_edge_MAE_reduction']):.6g} | "
            f"{float(row['gradient_l1_edge_reduction']):.6g} | {float(row['edge_F1_gain']):.6g} |"
        )
    results = "\n".join(result_lines) + "\n"
    (root / "manuscript_draft").mkdir(parents=True, exist_ok=True)
    (root / "manuscript_draft" / "Methods_draft.md").write_text(methods, encoding="utf-8")
    (root / "manuscript_draft" / "Experiments_draft.md").write_text(experiments, encoding="utf-8")
    (root / "manuscript_draft" / "Results_draft.md").write_text(results, encoding="utf-8")
    (root / "manuscript_draft" / "MANUSCRIPT_CORE_CLAIM.md").write_text(
        "PASD-Core demonstrates preliminary, target-dependent cross-family numerical and regional-edge structural gains under a locked FlatVel-A source-training protocol, but the evidence does not support universal OOD generalization or complete high-wavenumber recovery.\n",
        encoding="utf-8",
    )


def _write_limitations(root: Path) -> None:
    text = """# Limitations and Negative Results

1. Geometry attention is not claimed as an independently stable source of gain. Phase-4 treats it as part of the locked PASD-Core package rather than as a universally beneficial isolated module.
2. CurveVel-A edge-gradient improvement is small and should be discussed with bootstrap intervals and seed-level consistency, not as large gradient recovery.
3. CurveVel-A edge-F1 improves but remains low in absolute value. The correct wording is boundary localization improvement, not accurate interface reconstruction.
4. The evidence does not support universal OOD generalization, full high-wavenumber structural recovery, natural-image foundation-model superiority, or target-agnostic performance guarantees.
5. The paper should not add new target families, new foundation backbones, PDE consistency losses, or additional model-selection experiments in this freeze stage.
"""
    (root / "LIMITATIONS_AND_NEGATIVE_RESULTS.md").write_text(text, encoding="utf-8")


def _write_repro(root: Path, phase3r_root: Path) -> None:
    commands = """# Reproducibility Commands

All commands assume:

```powershell
cd D:\\ryjin\\fwi_visionfm
$env:PYTHONPATH='D:\\ryjin'
```

## Phase-3R metric repair

```powershell
python -m fwi_visionfm.pasd.freeze_phase3r --phase3-root outputs\\pasd_phase3_paper --locked-config configs\\pasd_phase3_pasd_core_locked.json --dual-target-protocol protocols\\pasd_phase3_dual_target_locked.json --output outputs\\pasd_phase3r_metric_repair\\freeze_manifest.json
python -m fwi_visionfm.pasd.audit_phase3_archives --phase3-root outputs\\pasd_phase3_paper --locked-config configs\\pasd_phase3_pasd_core_locked.json --dual-target-protocol protocols\\pasd_phase3_dual_target_locked.json --output outputs\\pasd_phase3r_metric_repair\\archive_audit
python -m fwi_visionfm.pasd.recompute_phase3_metrics --phase3-root outputs\\pasd_phase3_paper --locked-config configs\\pasd_phase3_pasd_core_locked.json --dual-target-protocol protocols\\pasd_phase3_dual_target_locked.json --output outputs\\pasd_phase3r_metric_repair\\corrected_metrics --use-fresh-prediction-archives-only --edge-mask source_threshold_strict_gt --prediction-edge-threshold source_val_locked --dx auto --dz auto
python -m fwi_visionfm.pasd.bootstrap_corrected_metrics --corrected-metrics-root outputs\\pasd_phase3r_metric_repair\\corrected_metrics --variants B1_raw_unet PASD_Core_locked --targets cross_curvevel_a cross_flatfault_a --seeds 0 1 2 --metrics MAE RMSE SSIM source_threshold_edge_MAE gradient_l1_edge edge_F1 --bootstrap-resamples 2000 --output outputs\\pasd_phase3r_metric_repair\\bootstrap
python -m fwi_visionfm.pasd.corrected_metric_plotting --phase3-root outputs\\pasd_phase3_paper --output-root outputs\\pasd_phase3r_metric_repair
```

## Phase-4 freeze package

```powershell
python -m fwi_visionfm.pasd.phase4_paper_freeze --phase3-root outputs\\pasd_phase3_paper --phase3r-root outputs\\pasd_phase3r_metric_repair --output outputs\\pasd_phase4_paper_freeze
```

## Tests

```powershell
python -m pytest -q tests -k pasd
python -m pytest -q tests -k \"pasd or phase3r\"
```
"""
    (root / "REPRODUCIBILITY_COMMANDS.md").write_text(commands, encoding="utf-8")
    readme = f"""# PASD Phase-4 Paper Freeze

This package freezes the PASD paper evidence after Phase-3R metric repair. It is a writing and reproducibility package only. It does not retrain, reselect, or alter Phase-3 prediction archives.

Source evidence root: `{phase3r_root}`

Main files:

- `freeze_manifest.json`
- `results_summary.csv`
- `manuscript_draft/`
- `paper_figures/`
- `paper_tables/`
- `LIMITATIONS_AND_NEGATIVE_RESULTS.md`
- `REPRODUCIBILITY_COMMANDS.md`
"""
    (root / "README.md").write_text(readme, encoding="utf-8")


def phase4_freeze(phase3_root: Path, phase3r_root: Path, output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    figure_copy = _copy_tree_files(phase3r_root / "figures", output / "paper_figures", (".png", ".pdf"))
    table_copy = _copy_tree_files(phase3r_root / "tables", output / "paper_tables", (".csv",))
    summary = _build_results_summary(phase3r_root, output / "results_summary.csv")
    _write_manuscript_draft(output, summary)
    _write_limitations(output)
    _write_repro(output, phase3r_root)
    manifest = {
        "status": "FROZEN",
        "phase": "PASD Phase-4: Paper Freeze, Reproducibility Package, and Manuscript Preparation",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "no_more_model_iteration": True,
        "phase3_root": str(phase3_root),
        "phase3r_root": str(phase3r_root),
        "git": {"commit": _git(Path.cwd(), ["git", "rev-parse", "HEAD"]), "status": _git(Path.cwd(), ["git", "status", "--short"])},
        "locked_inputs": {
            "phase3r_report": {"path": str(phase3r_root / "PASD_PHASE3R_METRIC_REPAIR_REPORT.md"), "sha256": sha256_file(phase3r_root / "PASD_PHASE3R_METRIC_REPAIR_REPORT.md")},
            "phase3r_corrected_summary": {"path": str(phase3r_root / "corrected_metrics" / "corrected_summary_across_seeds.csv"), "sha256": sha256_file(phase3r_root / "corrected_metrics" / "corrected_summary_across_seeds.csv")},
            "phase3r_identity_check": {"path": str(phase3r_root / "corrected_metrics" / "masked_mae_identity_check.csv"), "sha256": sha256_file(phase3r_root / "corrected_metrics" / "masked_mae_identity_check.csv")},
        },
        "phase3_archives": _hash_files(phase3_root / "dual_target_formal" / "prediction_archives", ("*.npz",)),
        "phase3r_figures": figure_copy,
        "phase3r_tables": table_copy,
        "phase4_outputs": _hash_files(output, ("*.md", "*.csv", "*.json", "*.png", "*.pdf")),
        "conclusion_boundary": "preliminary target-dependent cross-family numerical and regional-edge structural gain; no universal OOD or full high-wavenumber recovery claim",
    }
    write_json(output / "freeze_manifest.json", manifest)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase3-root", required=True, type=Path)
    parser.add_argument("--phase3r-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    out = phase4_freeze(args.phase3_root, args.phase3r_root, args.output)
    print(json.dumps({"status": "SUCCESS", "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
