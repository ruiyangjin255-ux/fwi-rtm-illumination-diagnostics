from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from fwi_visionfm.evaluation.visual_score import add_visual_scores


FIELDS = [
    "fusion_name",
    "method",
    "source_a",
    "source_b",
    "best_param",
    "optimize_requested",
    "optimize_actual",
    "MAE",
    "RMSE",
    "SSIM",
    "gradient_error",
    "edge_MAE",
    "visual_score",
    "visual_rank",
    "status",
    "skip_reason",
    "reference_only",
    "run_dir",
]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def collect_fusion_rows(root: str | Path) -> list[dict[str, Any]]:
    rows = []
    for config_path in sorted(Path(root).rglob("fusion_config.json")):
        config = _read_json(config_path)
        metrics = _read_json(config_path.parent / "fused_metrics_cross_family_test.json")
        rows.append(
            {
                "fusion_name": config_path.parent.name,
                "method": config.get("method", ""),
                "source_a": Path(config.get("run_a", "")).parts[-5] if config.get("run_a") else "",
                "source_b": Path(config.get("run_b", "")).parts[-5] if config.get("run_b") else "",
                "best_param": config.get("best_param", ""),
                "optimize_requested": config.get("optimize_requested", ""),
                "optimize_actual": config.get("optimize_actual", ""),
                "MAE": metrics.get("mae", ""),
                "RMSE": metrics.get("rmse", ""),
                "SSIM": metrics.get("ssim", ""),
                "gradient_error": metrics.get("gradient_error", ""),
                "edge_MAE": metrics.get("edge_mae", ""),
                "reference_only": bool(config.get("reference_only", False)),
                "status": config.get("status", "SUCCESS"),
                "skip_reason": config.get("skip_reason", ""),
                "run_dir": str(config_path.parent),
            }
        )
    scored = add_visual_scores(rows)
    ranked = sorted(range(len(scored)), key=lambda idx: float(scored[idx]["visual_score"]), reverse=True)
    ranks = {idx: rank for rank, idx in enumerate(ranked, start=1)}
    for idx, row in enumerate(scored):
        row["visual_rank"] = ranks[idx]
    return scored


def write_fusion_summary(root: str | Path) -> dict[str, Path]:
    output_root = Path(root)
    rows = collect_fusion_rows(output_root)
    summary_path = output_root / "protocol_v4_fusion_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in FIELDS} for row in rows])
    ranking_path = output_root / "protocol_v4_fusion_ranking.csv"
    ranked = sorted(rows, key=lambda row: float(row["visual_score"]), reverse=True)
    with ranking_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in FIELDS} for row in ranked])
    return {"summary": summary_path, "ranking": ranking_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Protocol V4 fusion results.")
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    paths = write_fusion_summary(parse_args().root)
    for path in paths.values():
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
