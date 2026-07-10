from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from admit_fwi.evaluate_fwi_model_quality import evaluate_run_dir, write_outputs
from admit_fwi.optimize_fwi_update_scale import DEFAULT_ALPHAS, optimize_run_dir
from admit_fwi.run_rtm_before_after_fwi import run_before_after_rtm


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FWI_DIR = ROOT / "admit_fwi" / "outputs" / "FWI" / "full_salt_fwi_cg_allshots_v2"
DEFAULT_TRUE_MODEL = ROOT / "admit_fwi" / "outputs" / "generated_inputs" / "seg676x230_from_fwi_true.bin"
DEFAULT_RTM_DIR = ROOT / "admit_fwi" / "outputs" / "RTM" / "before_after_fwi_alpha010_nt1200_shots12"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rtm_status(rtm_dir: Path) -> dict[str, Any]:
    summary_path = rtm_dir / "rtm_before_after_summary.json"
    if not summary_path.exists():
        return {"available": False, "summary_path": str(summary_path)}
    summary = _load_json(summary_path)
    before = summary["cases"]["before_initial_velocity"]["filtered"]
    after = summary["cases"]["after_fwi_velocity"]["filtered"]
    return {
        "available": True,
        "summary_path": str(summary_path),
        "verdict": summary.get("verdict"),
        "shot_count": summary.get("shot_count"),
        "filtered_rmse_before": before.get("reference_rmse"),
        "filtered_rmse_after": after.get("reference_rmse"),
        "filtered_rmse_improvement_fraction": summary.get("filtered_reference_rmse_improvement_fraction"),
        "filtered_corr_before": before.get("reference_corr"),
        "filtered_corr_after": after.get("reference_corr"),
    }


def write_pipeline_report(report: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "optimized_fwi_rtm_pipeline_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path = output_dir / "optimized_fwi_rtm_pipeline_report.md"
    lines = [
        "# Optimized FWI-RTM pipeline report",
        "",
        "## FWI quality gate",
        "",
        f"- `full_update_verdict`: {report['full_update_quality'].get('verdict')}",
        f"- `full_update_mae_improvement_pct`: {report['full_update_quality'].get('mae_improvement_fraction', 0.0) * 100:.4f}",
        f"- `full_update_edge_mae_improvement_pct`: {report['full_update_quality'].get('edge_mae_improvement_fraction', 0.0) * 100:.4f}",
        f"- `selected_alpha`: {report['update_scale'].get('selected_alpha')}",
        f"- `selected_model`: {report['selected_model']}",
        "",
        "## RTM validation",
        "",
    ]
    rtm = report["rtm_validation"]
    if rtm.get("available"):
        lines.extend(
            [
                f"- `verdict`: {rtm.get('verdict')}",
                f"- `shot_count`: {rtm.get('shot_count')}",
                f"- `filtered_rmse_before`: {rtm.get('filtered_rmse_before'):.8f}",
                f"- `filtered_rmse_after`: {rtm.get('filtered_rmse_after'):.8f}",
                f"- `filtered_rmse_improvement_pct`: {rtm.get('filtered_rmse_improvement_fraction', 0.0) * 100:.4f}",
                f"- `summary_path`: {rtm.get('summary_path')}",
            ]
        )
    else:
        lines.append(f"- `available`: False; run RTM or provide existing summary at `{rtm.get('summary_path')}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def run_pipeline(
    *,
    fwi_dir: Path,
    true_model: Path,
    output_dir: Path,
    rtm_dir: Path,
    run_rtm: bool,
    alphas: list[float],
    edge_tolerance: float = 0.0,
    gradient_tolerance: float = 0.05,
    nt: int = 1200,
    f0: float = 20.0,
    max_shots: int = 12,
    pad_x: int = 60,
    pad_bottom: int = 60,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    quality = evaluate_run_dir(fwi_dir)
    quality_paths = write_outputs(quality, output_dir / "full_update_model_quality.json")
    scale_paths = optimize_run_dir(
        fwi_dir,
        output_dir=output_dir / "update_scale_optimization",
        alphas=alphas,
        edge_tolerance=edge_tolerance,
        gradient_tolerance=gradient_tolerance,
    )
    scale_summary = _load_json(scale_paths["json"])
    selected_model = scale_paths["model"]

    if run_rtm:
        run_before_after_rtm(
            true_model_path=true_model,
            initial_model_path=fwi_dir / "full_salt_initial_model.npy",
            inverted_model_path=selected_model,
            output_dir=rtm_dir,
            nt=nt,
            f0=f0,
            max_shots=max_shots,
            pad_x=pad_x,
            pad_bottom=pad_bottom,
        )

    report = {
        "fwi_dir": str(fwi_dir),
        "true_model": str(true_model),
        "output_dir": str(output_dir),
        "full_update_quality": quality,
        "full_update_quality_paths": {key: str(value) for key, value in quality_paths.items()},
        "update_scale": scale_summary,
        "update_scale_paths": {key: str(value) for key, value in scale_paths.items()},
        "selected_model": str(selected_model),
        "rtm_validation": _rtm_status(rtm_dir),
    }
    written = write_pipeline_report(report, output_dir)
    report["pipeline_report_paths"] = {key: str(value) for key, value in written.items()}
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the gated FWI-to-RTM optimization pipeline.")
    parser.add_argument("--fwi-dir", type=Path, default=DEFAULT_FWI_DIR)
    parser.add_argument("--true-model", type=Path, default=DEFAULT_TRUE_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FWI_DIR / "optimized_fwi_rtm_pipeline")
    parser.add_argument("--rtm-dir", type=Path, default=DEFAULT_RTM_DIR)
    parser.add_argument("--run-rtm", action="store_true", help="Run the expensive RTM validation instead of only reading it.")
    parser.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    parser.add_argument("--edge-tolerance", type=float, default=0.0)
    parser.add_argument("--gradient-tolerance", type=float, default=0.05)
    parser.add_argument("--nt", type=int, default=1200)
    parser.add_argument("--f0", type=float, default=20.0)
    parser.add_argument("--max-shots", type=int, default=12)
    parser.add_argument("--pad-x", type=int, default=60)
    parser.add_argument("--pad-bottom", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_pipeline(
        fwi_dir=args.fwi_dir,
        true_model=args.true_model,
        output_dir=args.output_dir,
        rtm_dir=args.rtm_dir,
        run_rtm=args.run_rtm,
        alphas=list(args.alphas),
        edge_tolerance=args.edge_tolerance,
        gradient_tolerance=args.gradient_tolerance,
        nt=args.nt,
        f0=args.f0,
        max_shots=args.max_shots,
        pad_x=args.pad_x,
        pad_bottom=args.pad_bottom,
    )
    for label, path in report["pipeline_report_paths"].items():
        print(f"{label}: {path}")
    print(f"selected_alpha: {report['update_scale']['selected_alpha']}")
    print(f"rtm_available: {report['rtm_validation']['available']}")


if __name__ == "__main__":
    main()
