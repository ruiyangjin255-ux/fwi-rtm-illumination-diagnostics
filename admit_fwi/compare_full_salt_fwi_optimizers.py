from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_CG_DIR = ROOT / "outputs" / "FWI" / "full_salt_fwi_cg_nt2500_2iter"
DEFAULT_PCG_DIR = ROOT / "outputs" / "FWI" / "full_salt_fwi_pcg_nt2500_2iter"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "FWI" / "full_salt_fwi_optimizer_compare_report"


def _read_history(history_path: Path) -> list[dict[str, Any]]:
    if not history_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with history_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            parsed = dict(row)
            for key in (
                "iteration",
                "completed_shots",
            ):
                if parsed.get(key) not in (None, ""):
                    parsed[key] = int(float(parsed[key]))
            for key in (
                "mean_misfit",
                "step_scale",
                "cg_beta",
                "max_abs_update",
                "model_min",
                "model_max",
            ):
                if parsed.get(key) not in (None, ""):
                    parsed[key] = float(parsed[key])
            rows.append(parsed)
    return rows


def _checkpoint_iterations(run_dir: Path) -> list[int]:
    checkpoint_dir = run_dir / "checkpoint"
    if not checkpoint_dir.exists():
        return []
    iterations: list[int] = []
    for path in checkpoint_dir.glob("iteration_*.json"):
        stem = path.stem.removeprefix("iteration_")
        try:
            iterations.append(int(stem))
        except ValueError:
            continue
    return sorted(iterations)


def summarize_optimizer_run(run_dir: str | Path) -> dict[str, Any]:
    """Summarize an existing FWI optimizer output directory without modifying it."""
    run_dir = Path(run_dir)
    history = _read_history(run_dir / "fwi_iteration_history.csv")
    misfits = [float(row["mean_misfit"]) for row in history if "mean_misfit" in row]
    optimizer = str(history[-1].get("optimizer", "")) if history else ""
    initial_misfit = misfits[0] if misfits else None
    final_misfit = misfits[-1] if misfits else None
    reduction = None
    if initial_misfit is not None and final_misfit is not None and initial_misfit > 0.0:
        reduction = (initial_misfit - final_misfit) / initial_misfit

    return {
        "name": run_dir.name,
        "path": str(run_dir),
        "exists": run_dir.exists(),
        "optimizer": optimizer,
        "history_exists": (run_dir / "fwi_iteration_history.csv").exists(),
        "iterations_completed": len(history),
        "last_iteration": int(history[-1]["iteration"]) if history else None,
        "completed_shots_last": int(history[-1]["completed_shots"]) if history else None,
        "initial_misfit": initial_misfit,
        "final_misfit": final_misfit,
        "misfit_reduction_fraction": reduction,
        "misfit_history": misfits,
        "checkpoint_iterations": _checkpoint_iterations(run_dir),
        "has_initial_model": (run_dir / "full_salt_initial_model.npy").exists(),
        "has_inverted_model": (run_dir / "full_salt_inverted_model.npy").exists(),
        "has_model_update": (run_dir / "full_salt_model_update.npy").exists(),
        "has_summary_json": (run_dir / "full_salt_fwi_summary.json").exists(),
        "pid_file": str(run_dir / "run.pid") if (run_dir / "run.pid").exists() else None,
    }


def compare_optimizer_runs(*, cg_dir: str | Path, pcg_dir: str | Path) -> dict[str, Any]:
    """Compare existing CG and P-CG output folders without launching FWI."""
    cg = summarize_optimizer_run(cg_dir)
    pcg = summarize_optimizer_run(pcg_dir)
    missing_iterations_vs_cg = max(0, int(cg["iterations_completed"]) - int(pcg["iterations_completed"]))
    pcg["missing_iterations_vs_cg"] = missing_iterations_vs_cg

    if not cg["exists"] or not pcg["exists"]:
        status = "missing-input"
    elif cg["iterations_completed"] == 0 or pcg["iterations_completed"] == 0:
        status = "no-history"
    elif missing_iterations_vs_cg > 0:
        status = "partial"
    else:
        status = "ready"

    final_misfit_delta_pcg_minus_cg = None
    if cg["final_misfit"] is not None and pcg["final_misfit"] is not None:
        final_misfit_delta_pcg_minus_cg = float(pcg["final_misfit"] - cg["final_misfit"])

    return {
        "status": status,
        "cg": cg,
        "pcg": pcg,
        "final_misfit_delta_pcg_minus_cg": final_misfit_delta_pcg_minus_cg,
        "notes": [
            "This report only reads existing output files and does not run forward modeling or inversion.",
            "Use status=ready only when both optimizers have comparable completed iteration counts.",
        ],
    }


def _format_float(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.6e}"


def _format_percent(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{100.0 * float(value):.3f}%"


def _render_markdown(comparison: dict[str, Any]) -> str:
    cg = comparison["cg"]
    pcg = comparison["pcg"]
    lines = [
        "# CG/P-CG 优化器对照报告",
        "",
        "本报告只读取已有 FWI 输出文件，不启动正演或反演计算。",
        "",
        f"- 状态: `{comparison['status']}`",
        f"- CG 目录: `{cg['path']}`",
        f"- P-CG 目录: `{pcg['path']}`",
        "",
        "| 方法 | 已完成迭代 | 最后迭代 | 最终误差 | 误差下降 | 最后完成炮数 | 反演模型 | 更新量 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        (
            f"| CG | {cg['iterations_completed']} | {cg['last_iteration']} | "
            f"{_format_float(cg['final_misfit'])} | {_format_percent(cg['misfit_reduction_fraction'])} | "
            f"{cg['completed_shots_last']} | {cg['has_inverted_model']} | {cg['has_model_update']} |"
        ),
        (
            f"| P-CG | {pcg['iterations_completed']} | {pcg['last_iteration']} | "
            f"{_format_float(pcg['final_misfit'])} | {_format_percent(pcg['misfit_reduction_fraction'])} | "
            f"{pcg['completed_shots_last']} | {pcg['has_inverted_model']} | {pcg['has_model_update']} |"
        ),
        "",
        f"- P-CG 相对 CG 尚缺迭代数: {pcg.get('missing_iterations_vs_cg', 0)}",
        f"- P-CG 最终误差减 CG 最终误差: {_format_float(comparison['final_misfit_delta_pcg_minus_cg'])}",
        "",
        "## 判读建议",
        "",
        "- `partial` 表示两组迭代数尚不一致，暂不应直接比较最终收敛优劣。",
        "- `ready` 表示两组已有相同数量的历史迭代，可继续生成误差曲线和模型差异图。",
        "- 若 P-CG 的历史文件仍在增长，等当前运行结束后重新执行本脚本即可刷新报告。",
        "",
    ]
    return "\n".join(lines)


def write_optimizer_comparison(comparison: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    """Write comparison JSON and Markdown to a dedicated report directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "full_salt_fwi_optimizer_compare.json"
    markdown_path = output_dir / "full_salt_fwi_optimizer_compare.md"
    json_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text(_render_markdown(comparison), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总已有 full-salt FWI CG/P-CG 输出，不运行反演")
    parser.add_argument("--cg-dir", type=Path, default=DEFAULT_CG_DIR)
    parser.add_argument("--pcg-dir", type=Path, default=DEFAULT_PCG_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison = compare_optimizer_runs(cg_dir=args.cg_dir, pcg_dir=args.pcg_dir)
    written = write_optimizer_comparison(comparison, args.output_dir)
    print(json.dumps({"status": comparison["status"], "written": {k: str(v) for k, v in written.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
