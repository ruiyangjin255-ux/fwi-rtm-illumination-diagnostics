from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from rtm_acoustic.evaluate_fwi_model_quality import evaluate_run_dir


ROOT = Path(__file__).resolve().parents[1]
RTM_ROOT = ROOT / "rtm_acoustic"
OUTPUTS = RTM_ROOT / "outputs"
DOCS = RTM_ROOT / "docs"

FULL_FWI_RUN_DIRS = {
    "CG_allshots_v2": OUTPUTS / "FWI" / "full_salt_fwi_cg_allshots_v2",
    "P-CG_allshots_v2": OUTPUTS / "FWI" / "full_salt_fwi_pcg_allshots_v2",
    "CG_nt2500_2iter": OUTPUTS / "FWI" / "full_salt_fwi_cg_nt2500_2iter",
    "P-CG_nt2500_2iter": OUTPUTS / "FWI" / "full_salt_fwi_pcg_nt2500_2iter",
    "CG_nt4000_continue": OUTPUTS / "FWI" / "full_salt_fwi_cg_nt4000_from_nt2500_iter2",
    "CG_f15_outerpad_continue": OUTPUTS
    / "FWI"
    / "full_salt_fwi_cg_f15_nt2500_outerpad_x40_top40_bottom60_continue_step2_1iter",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown_table(path: Path, title: str, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(f"# {title}\n\nNo rows.\n", encoding="utf-8")
        return
    headers = list(rows[0])
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pct(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value * 100:.4f}"


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float64).ravel()
    bv = np.asarray(b, dtype=np.float64).ravel()
    if av.size != bv.size or av.size == 0:
        return float("nan")
    if np.std(av) == 0.0 or np.std(bv) == 0.0:
        return float("nan")
    return float(np.corrcoef(av, bv)[0, 1])


def collect_full_fwi_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, run_dir in FULL_FWI_RUN_DIRS.items():
        summary_path = run_dir / "full_salt_fwi_summary.json"
        if not summary_path.exists():
            continue
        summary = _load_json(summary_path)
        config = summary.get("config", {})
        rows.append(
            {
                "rank_metric": "full_model_misfit_reduction",
                "case": label,
                "optimizer": config.get("optimizer", ""),
                "shots": summary.get("shot_count", ""),
                "iterations": summary.get("iterations", ""),
                "nt": config.get("nt", ""),
                "f0_hz": config.get("f0", ""),
                "initial_misfit": f"{summary.get('initial_misfit', 0.0):.8e}",
                "final_misfit": f"{summary.get('final_misfit', 0.0):.8e}",
                "misfit_reduction_pct": _pct(summary.get("misfit_reduction_fraction")),
                "evidence_level": "core" if "allshots_v2" in label else "supporting",
            }
        )
    return sorted(rows, key=lambda row: float(row["misfit_reduction_pct"]), reverse=True)


def collect_fwi_quality_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    required = ("full_salt_true_model.npy", "full_salt_initial_model.npy", "full_salt_inverted_model.npy")
    for label, run_dir in FULL_FWI_RUN_DIRS.items():
        if not all((run_dir / name).exists() for name in required):
            continue
        metrics = evaluate_run_dir(run_dir)
        rows.append(
            {
                "rank_metric": "model_quality_improvement",
                "case": label,
                "optimizer": metrics.get("optimizer", ""),
                "shots": metrics.get("shot_count", ""),
                "iterations": metrics.get("iterations", ""),
                "mae_improvement_pct": _pct(metrics.get("mae_improvement_fraction")),
                "rmse_improvement_pct": _pct(metrics.get("rmse_improvement_fraction")),
                "edge_mae_improvement_pct": _pct(metrics.get("edge_mae_improvement_fraction")),
                "gradient_mae_improvement_pct": _pct(metrics.get("gradient_mae_improvement_fraction")),
                "update_l1_edge_fraction": f"{metrics.get('update_l1_edge_fraction', 0.0):.6f}",
                "update_true_error_correlation": f"{metrics.get('update_true_error_correlation', 0.0):.6f}",
                "verdict": metrics.get("verdict", ""),
            }
        )
    return sorted(rows, key=lambda row: float(row["mae_improvement_pct"]), reverse=True)


def collect_update_scale_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, run_dir in FULL_FWI_RUN_DIRS.items():
        path = run_dir / "update_scale_optimization" / "update_scale_optimization.json"
        if not path.exists():
            continue
        data = _load_json(path)
        selected_alpha = data.get("selected_alpha")
        for candidate in data.get("candidates", []):
            rows.append(
                {
                    "case": label,
                    "alpha": candidate.get("alpha", ""),
                    "accepted": candidate.get("accepted", ""),
                    "selected": candidate.get("alpha") == selected_alpha,
                    "score": f"{candidate.get('score', 0.0):.8f}",
                    "mae_improvement_pct": _pct(candidate.get("mae_improvement_fraction")),
                    "rmse_improvement_pct": _pct(candidate.get("rmse_improvement_fraction")),
                    "edge_mae_improvement_pct": _pct(candidate.get("edge_mae_improvement_fraction")),
                    "gradient_mae_improvement_pct": _pct(candidate.get("gradient_mae_improvement_fraction")),
                    "verdict": candidate.get("verdict", ""),
                }
            )
    return rows


def collect_local_fwi_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    fixed = _load_json(OUTPUTS / "FWI影响因素" / "small_salt_fwi_illumination_scan" / "summary_scan.json")
    rows.append(
        {
            "rank_metric": "local_window_misfit_reduction",
            "case": "fixed_step_baseline",
            "epsilon": "",
            "max_update": "",
            "line_search": "none",
            "selected_steps": "",
            "initial_misfit": f"{fixed['baseline']['initial_misfit']:.10f}",
            "final_misfit": f"{fixed['baseline']['final_misfit']:.10f}",
            "misfit_reduction_pct": _pct(fixed["baseline"]["misfit_reduction_fraction"]),
            "claim_role": "baseline reference",
        }
    )
    best_fixed_pre = fixed["best_preconditioned"]
    rows.append(
        {
            "rank_metric": "local_window_misfit_reduction",
            "case": "fixed_step_illumination_preconditioned_best_1d",
            "epsilon": best_fixed_pre.get("epsilon", ""),
            "max_update": "",
            "line_search": "none",
            "selected_steps": "",
            "initial_misfit": f"{best_fixed_pre['initial_misfit']:.10f}",
            "final_misfit": f"{best_fixed_pre['final_misfit']:.10f}",
            "misfit_reduction_pct": _pct(best_fixed_pre["misfit_reduction_fraction"]),
            "claim_role": "negative ablation",
        }
    )

    scan2d = _load_json(OUTPUTS / "FWI影响因素" / "small_salt_fwi_illumination_2d_scan" / "summary_2d_scan.json")
    best_2d = scan2d["best_preconditioned"]
    rows.append(
        {
            "rank_metric": "local_window_misfit_reduction",
            "case": "fixed_step_illumination_preconditioned_best_2d",
            "epsilon": best_2d.get("epsilon", ""),
            "max_update": best_2d.get("max_update", ""),
            "line_search": "none",
            "selected_steps": "",
            "initial_misfit": f"{best_2d['initial_misfit']:.10f}",
            "final_misfit": f"{best_2d['final_misfit']:.10f}",
            "misfit_reduction_pct": _pct(best_2d["misfit_reduction_fraction"]),
            "claim_role": "negative ablation",
        }
    )

    line = _load_json(OUTPUTS / "FWI影响因素" / "small_salt_fwi_line_search" / "line_search_summary.json")
    for key, label in (("baseline", "line_search_baseline"), ("illumination_preconditioned", "line_search_illumination_preconditioned")):
        item = line[key]
        rows.append(
            {
                "rank_metric": "local_window_misfit_reduction",
                "case": label,
                "epsilon": item.get("preconditioner_epsilon") or "",
                "max_update": "",
                "line_search": "fixed candidates",
                "selected_steps": ",".join(str(x) for x in item.get("selected_step_scales", [])),
                "initial_misfit": f"{item['initial_misfit']:.10f}",
                "final_misfit": f"{item['final_misfit']:.10f}",
                "misfit_reduction_pct": _pct(item["misfit_reduction_fraction"]),
                "claim_role": "optimizer evidence",
            }
        )

    adaptive = _load_json(
        OUTPUTS / "FWI影响因素" / "small_salt_fwi_adaptive_line_search" / "adaptive_line_search_summary.json"
    )
    for key, label in (
        ("baseline", "adaptive_line_search_baseline"),
        ("illumination_preconditioned", "adaptive_line_search_illumination_preconditioned"),
    ):
        item = adaptive[key]
        rows.append(
            {
                "rank_metric": "local_window_misfit_reduction",
                "case": label,
                "epsilon": item.get("preconditioner_epsilon") or "",
                "max_update": "",
                "line_search": "adaptive extended",
                "selected_steps": ",".join(str(x) for x in item.get("selected_step_scales", [])),
                "initial_misfit": f"{item['initial_misfit']:.10f}",
                "final_misfit": f"{item['final_misfit']:.10f}",
                "misfit_reduction_pct": _pct(item["misfit_reduction_fraction"]),
                "claim_role": "core optimizer evidence" if key == "baseline" else "negative ablation",
            }
        )

    return sorted(rows, key=lambda row: float(row["misfit_reduction_pct"]), reverse=True)


def collect_rtm_condition_rows() -> list[dict[str, Any]]:
    scheme_dir = OUTPUTS / "RTM" / "seg_salt_scheme2_full30m_nt4001_workers4"
    raw = np.load(scheme_dir / "scheme2_raw.npy")
    source_norm = np.load(scheme_dir / "scheme2_source_normalized.npy")
    sr_norm = np.load(scheme_dir / "scheme2_source_receiver_normalized.npy")
    lap = np.load(scheme_dir / "scheme2_laplacian_image.npy")
    lap_source = np.load(scheme_dir / "scheme2_laplacian_source_normalized.npy")
    source_illum = np.load(scheme_dir / "scheme2_source_illumination.npy")
    receiver_illum = np.load(scheme_dir / "scheme2_receiver_illumination.npy")
    geom = np.sqrt(np.maximum(source_illum, 0.0) * np.maximum(receiver_illum, 0.0))
    low_fraction = float(np.mean(geom < 0.01 * float(np.max(geom))))

    return [
        {
            "case": "source_receiver_vs_source_normalized",
            "correlation": f"{_safe_corr(sr_norm, source_norm):.4f}",
            "abs_p99": f"{np.percentile(np.abs(sr_norm), 99):.6e}",
            "low_illumination_fraction": f"{low_fraction:.6f}",
            "claim_role": "source-receiver normalization close to source-only under full aperture",
        },
        {
            "case": "laplacian_source_vs_source_normalized",
            "correlation": f"{_safe_corr(lap_source, source_norm):.4f}",
            "abs_p99": f"{np.percentile(np.abs(lap_source), 99):.6e}",
            "low_illumination_fraction": f"{low_fraction:.6f}",
            "claim_role": "laplacian changes spectral content and low-wavenumber background",
        },
        {
            "case": "raw_cross_correlation",
            "correlation": f"{_safe_corr(raw, source_norm):.4f}",
            "abs_p99": f"{np.percentile(np.abs(raw), 99):.6e}",
            "low_illumination_fraction": f"{low_fraction:.6f}",
            "claim_role": "amplitude reference",
        },
        {
            "case": "laplacian_raw",
            "correlation": f"{_safe_corr(lap, raw):.4f}",
            "abs_p99": f"{np.percentile(np.abs(lap), 99):.6e}",
            "low_illumination_fraction": f"{low_fraction:.6f}",
            "claim_role": "edge-enhancement reference",
        },
    ]


def build_innovation_rows() -> list[dict[str, Any]]:
    return [
        {
            "rank": 1,
            "innovation": "JGE-core: reproducible FWI-RTM diagnostic workflow for salt imaging",
            "evidence": "Full-model FWI, full-aperture RTM, local-window optimizer ablation",
            "risk": "FWI image quality still preliminary; must frame as diagnostic workflow",
            "paper_position": "main contribution",
        },
        {
            "rank": 2,
            "innovation": "Separation of RTM image normalization and FWI update preconditioning",
            "evidence": "RTM condition correlations plus local preconditioner negative ablations",
            "risk": "Needs careful language; no claim of superior preconditioned FWI",
            "paper_position": "methodological contribution",
        },
        {
            "rank": 3,
            "innovation": "Adaptive line-search dominance over lightweight illumination scaling",
            "evidence": "6.3690% local baseline vs 4.1386% local preconditioned reduction",
            "risk": "Local-window result only; should motivate full-model follow-up",
            "paper_position": "results and discussion",
        },
        {
            "rank": 4,
            "innovation": "Quantitative imaging-condition ranking instead of visual-only RTM comparison",
            "evidence": "Correlation, p99 amplitude, low-illumination fraction",
            "risk": "Needs stronger structural target metrics for higher-tier submission",
            "paper_position": "supporting result",
        },
        {
            "rank": 5,
            "innovation": "Optional ML low-wavenumber prior linked to FWI-RTM loop",
            "evidence": "Existing fwi_visionfm smoke and small-sample OpenFWI results",
            "risk": "Not ready as a JGE core claim without new controlled experiments",
            "paper_position": "future work only",
        },
    ]


def write_upgrade_plan(path: Path) -> None:
    text = """# JGE-oriented Upgrade Plan

## Journal fit

Journal of Geophysics and Engineering papers should be written as focused geophysical or engineering-geophysics studies, with a clear method, reproducible data processing path, quantitative results, and concise discussion. For a regular Paper, keep the manuscript within roughly 8000 words, 250-word abstract, up to 5 keywords, up to 10 figures/tables, and a compact reference list.

## Current readiness verdict

The current FWI effect is not yet strong enough to claim a high-quality inversion result. The most defensible JGE positioning is therefore not "a superior FWI algorithm", but "a reproducible diagnostic framework showing how FWI updates, RTM illumination normalization, imaging-condition selection, and line-search behavior interact in a salt model".

## Required result upgrades before submission

1. Use the structural FWI model-quality table as a gate: velocity MAE/RMSE, salt-boundary edge error, gradient error, update localization, and update/error correlation must be reported with the misfit curve.
2. Run the RTM-before/after-FWI comparison as a formal result: RTM with initial smoothed velocity versus RTM with inverted velocity, measured by reflector continuity, image focus, or objective image metrics.
3. Extend full-model FWI beyond 3 iterations only if misfit and model metrics improve together; otherwise keep the short run as a diagnostic case.
4. Convert smoke-level scheme2 results into a formal run or explicitly label it as imaging-condition diagnostic.
5. Keep ML/OpenFWI results out of the main claim unless a controlled initialization experiment is added.

## Recommended paper title

Diagnostic coupling of acoustic full-waveform inversion and reverse-time migration illumination analysis for salt-model imaging

## Recommended core figures

1. Workflow and reproducibility map.
2. Full-model FWI misfit and velocity update.
3. Initial-velocity RTM versus FWI-updated-velocity RTM.
4. RTM imaging-condition comparison.
5. Local-window line-search and illumination-preconditioner ranking.
6. Failure/limitation figure: where lightweight illumination preconditioning underperforms.

## Recommended claims

- Claim: The workflow provides a reproducible diagnostic bridge between FWI velocity updating and RTM imaging-condition assessment.
- Claim: For the current full-aperture salt RTM setup, source-receiver normalization is close to source-only normalization, while Laplacian filtering changes the spectral content more strongly.
- Claim: In the local-window FWI ablation, adaptive line search contributes more to misfit reduction than the current lightweight source-illumination preconditioner.
- Do not claim: The current FWI result has reached production-quality velocity recovery.
- Do not claim: The current illumination preconditioner improves FWI over baseline.
- Do not claim: The current ML foundation-model branch is a validated JGE-level contribution.

## Program framework

```text
rtm_acoustic/
  run_full_salt_fwi.py                  existing
  run_multishot_rtm.py                  existing
  run_scheme2_imaging_condition_compare.py existing
  build_jge_core_results.py             added summary/ranking generator
  evaluate_fwi_model_quality.py         added model MAE/RMSE/edge metrics
  optimize_fwi_update_scale.py          added damped update gate before RTM
  run_rtm_before_after_fwi.py           added RTM with initial vs inverted velocity
  run_optimized_fwi_rtm_pipeline.py     added recommended one-command pipeline
  build_jge_submission_package.py       added figures, captions, tables, checklist
```
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build(output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    full_fwi = collect_full_fwi_rows()
    fwi_quality = collect_fwi_quality_rows()
    update_scales = collect_update_scale_rows()
    local_fwi = collect_local_fwi_rows()
    rtm_conditions = collect_rtm_condition_rows()
    innovations = build_innovation_rows()

    written = {
        "full_fwi_csv": output_dir / "full_fwi_ranking.csv",
        "fwi_quality_csv": output_dir / "fwi_model_quality_ranking.csv",
        "update_scale_csv": output_dir / "fwi_update_scale_optimization.csv",
        "local_fwi_csv": output_dir / "local_fwi_strategy_ranking.csv",
        "rtm_csv": output_dir / "rtm_imaging_condition_metrics.csv",
        "innovation_csv": output_dir / "innovation_ranking.csv",
        "full_fwi_md": output_dir / "full_fwi_ranking.md",
        "fwi_quality_md": output_dir / "fwi_model_quality_ranking.md",
        "update_scale_md": output_dir / "fwi_update_scale_optimization.md",
        "local_fwi_md": output_dir / "local_fwi_strategy_ranking.md",
        "rtm_md": output_dir / "rtm_imaging_condition_metrics.md",
        "innovation_md": output_dir / "innovation_ranking.md",
        "plan_md": output_dir / "jge_upgrade_plan.md",
    }
    _write_csv(written["full_fwi_csv"], full_fwi)
    _write_csv(written["fwi_quality_csv"], fwi_quality)
    _write_csv(written["update_scale_csv"], update_scales)
    _write_csv(written["local_fwi_csv"], local_fwi)
    _write_csv(written["rtm_csv"], rtm_conditions)
    _write_csv(written["innovation_csv"], innovations)
    _write_markdown_table(written["full_fwi_md"], "Full-model FWI ranking", full_fwi)
    _write_markdown_table(written["fwi_quality_md"], "FWI model quality ranking", fwi_quality)
    _write_markdown_table(written["update_scale_md"], "FWI update-scale optimization", update_scales)
    _write_markdown_table(written["local_fwi_md"], "Local-window FWI strategy ranking", local_fwi)
    _write_markdown_table(written["rtm_md"], "RTM imaging-condition metrics", rtm_conditions)
    _write_markdown_table(written["innovation_md"], "Innovation ranking", innovations)
    write_upgrade_plan(written["plan_md"])
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Build JGE-oriented core result and ranking tables.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DOCS / "jge_revision",
        help="Directory for generated CSV and Markdown result tables.",
    )
    args = parser.parse_args()
    written = build(args.output_dir)
    for label, path in written.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
