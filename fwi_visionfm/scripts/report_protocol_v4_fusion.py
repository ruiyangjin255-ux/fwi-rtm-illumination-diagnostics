from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Any

from fwi_visionfm.scripts.summarize_protocol_v4_fusion import write_fusion_summary


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _best(rows: list[dict[str, Any]], field: str, *, reverse: bool = False) -> dict[str, Any] | None:
    usable = [row for row in rows if row.get(field) != ""]
    if not usable:
        return None
    return sorted(usable, key=lambda row: float(row[field]), reverse=reverse)[0]


def _load_integrated_rows(output_root: Path) -> list[dict[str, Any]]:
    candidates = [
        output_root.parent / "protocol_v4_integrated_bridge_visual_search" / "protocol_v4_integrated_summary.csv",
        Path("outputs") / "protocol_v4_integrated_bridge_visual_search" / "protocol_v4_integrated_summary.csv",
    ]
    for path in candidates:
        if path.exists():
            return _read_rows(path)
    return []


def _improves(best_fusion: dict[str, Any] | None, best_single: dict[str, Any] | None, field: str, *, higher: bool = False) -> str:
    if best_fusion is None or best_single is None:
        return "unavailable"
    fusion_value = float(best_fusion[field])
    single_value = float(best_single[field])
    improved = fusion_value > single_value if higher else fusion_value < single_value
    return f"{improved} (fusion={fusion_value}, best_single={single_value})"


def _method_best(rows: list[dict[str, Any]]) -> str:
    best = _best(rows, "visual_score", reverse=True)
    return best.get("method", "unavailable") if best else "unavailable"


def write_fusion_report(root: str | Path) -> Path:
    output_root = Path(root)
    paths = write_fusion_summary(output_root)
    rows = _read_rows(paths["summary"])
    success_rows = [row for row in rows if row.get("status", "SUCCESS") == "SUCCESS"]
    skipped_rows = [row for row in rows if row.get("status", "SUCCESS") != "SUCCESS"]
    primary = [row for row in success_rows if str(row.get("reference_only", "")).lower() != "true"]
    reference = [row for row in success_rows if str(row.get("reference_only", "")).lower() == "true"]
    integrated = [row for row in _load_integrated_rows(output_root) if row.get("status") == "SUCCESS"]
    best_visual = _best(primary, "visual_score", reverse=True)
    best_mae = _best(primary, "MAE")
    best_rmse = _best(primary, "RMSE")
    best_grad = _best(primary, "gradient_error")
    best_edge = _best(primary, "edge_MAE")
    single_visual = _best(integrated, "visual_score", reverse=True)
    single_mae = _best(integrated, "MAE")
    single_rmse = _best(integrated, "RMSE")
    single_grad = _best(integrated, "gradient_error")
    single_edge = _best(integrated, "edge_MAE")
    if best_visual and best_visual.get("run_dir"):
        run_dir = Path(best_visual["run_dir"])
        if (run_dir / "fused_prediction_grid.png").exists():
            shutil.copyfile(run_dir / "fused_prediction_grid.png", output_root / "best_fused_prediction_grid.png")
        if (run_dir / "fused_gradient_grid.png").exists():
            shutil.copyfile(run_dir / "fused_gradient_grid.png", output_root / "best_fused_gradient_grid.png")
    report_path = output_root / "protocol_v4_fusion_report.md"
    lines = [
        "# Protocol V4 Prediction Fusion Report",
        "",
        "Protocol V4 prediction fusion tests whether numerical and structural strengths from different model-bridge configurations can be combined under CPU-limited settings. It provides a low-cost post-training strategy to improve final velocity-map quality, while keeping application-level claims conservative.",
        "",
        "## Fusion Questions",
        f"- Best fused visual_score: {best_visual.get('fusion_name') if best_visual else 'unavailable'} ({best_visual.get('visual_score') if best_visual else ''}).",
        f"- Best fused MAE/RMSE: {best_mae.get('fusion_name') if best_mae else 'unavailable'} / {best_rmse.get('fusion_name') if best_rmse else 'unavailable'}.",
        f"- Best fused gradient_error/edge_MAE: {best_grad.get('fusion_name') if best_grad else 'unavailable'} / {best_edge.get('fusion_name') if best_edge else 'unavailable'}.",
        f"- Best fusion method by visual_score: {_method_best(primary)}.",
        f"- Fusion improves visual_score over best single model: {_improves(best_visual, single_visual, 'visual_score', higher=True)}.",
        f"- Fusion improves MAE/RMSE over best single model: MAE={_improves(best_mae, single_mae, 'MAE')}; RMSE={_improves(best_rmse, single_rmse, 'RMSE')}.",
        f"- Fusion improves gradient_error/edge_MAE over best single model: gradient_error={_improves(best_grad, single_grad, 'gradient_error')}; edge_MAE={_improves(best_edge, single_edge, 'edge_MAE')}.",
        "- The numerical-structural trade-off remains an active criterion: MAE/RMSE and gradient/edge metrics can select different fusion rows.",
        f"- DINOv2 probe rows are reference-only: {len(reference)} rows; not benchmark evidence.",
        f"- Skipped incompatible fusion rows: {len(skipped_rows)}; split/target mismatch rows are not fused.",
        "- Current outputs are not application-level and do not establish complex FWI deployment quality.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report Protocol V4 fusion results.")
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    print(f"Wrote {write_fusion_report(parse_args().root)}")


if __name__ == "__main__":
    main()
