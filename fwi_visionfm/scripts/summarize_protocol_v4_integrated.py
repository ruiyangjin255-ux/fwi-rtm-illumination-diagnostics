from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from fwi_visionfm.evaluation.visual_score import add_visual_scores


FIELDS = [
    "model_name",
    "bridge",
    "decoder_name",
    "loss_name",
    "seed",
    "MAE",
    "RMSE",
    "SSIM",
    "gradient_error",
    "edge_MAE",
    "visual_score",
    "visual_rank",
    "numerical_rank",
    "structural_rank",
    "status",
    "skip_reason",
    "is_probe",
    "is_structural_control",
    "run_dir",
]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for config_path in sorted(root.rglob("config.json")):
        if "manifests" in config_path.parts:
            continue
        config = _read_json(config_path)
        if config.get("protocol") != "protocol_v4_integrated_bridge_visual_search":
            continue
        metrics = _read_json(config_path.parent / "metrics_cross_family_test.json")
        rows.append(
            {
                "model_name": config.get("model_name", ""),
                "bridge": config.get("bridge", ""),
                "decoder_name": config.get("decoder_name", ""),
                "loss_name": config.get("loss_name", ""),
                "seed": config.get("seed", ""),
                "MAE": metrics.get("mae", ""),
                "RMSE": metrics.get("rmse", ""),
                "SSIM": metrics.get("ssim", ""),
                "gradient_error": metrics.get("gradient_error", ""),
                "edge_MAE": metrics.get("edge_mae", ""),
                "status": config.get("status", ""),
                "skip_reason": config.get("skip_reason", ""),
                "is_probe": bool(config.get("is_probe", False)),
                "is_structural_control": config.get("bridge") == "raw_repeat3",
                "run_dir": str(config_path.parent),
            }
        )
    return rows


def _rank(rows: list[dict[str, Any]], key: str, *, reverse: bool = False) -> dict[int, int]:
    ranked = sorted(
        [(index, row) for index, row in enumerate(rows) if row.get(key) != ""],
        key=lambda item: float(item[1][key]),
        reverse=reverse,
    )
    return {index: rank for rank, (index, _row) in enumerate(ranked, start=1)}


def write_integrated_summary(root: str | Path) -> Path:
    output_root = Path(root)
    rows = _collect_rows(output_root)
    success_indexes = [index for index, row in enumerate(rows) if row.get("status") == "SUCCESS"]
    scored_success = add_visual_scores([rows[index] for index in success_indexes])
    for source_index, scored in zip(success_indexes, scored_success):
        rows[source_index]["visual_score"] = scored["visual_score"]
    for row in rows:
        row.setdefault("visual_score", "")
    visual_ranks = _rank(rows, "visual_score", reverse=True)
    numerical_scores = []
    structural_scores = []
    for row in rows:
        try:
            numerical_scores.append((float(row["MAE"]) + float(row["RMSE"])) / 2.0)
            structural_scores.append((float(row["gradient_error"]) + float(row["edge_MAE"])) / 2.0)
        except (TypeError, ValueError):
            numerical_scores.append("")
            structural_scores.append("")
    for row, score in zip(rows, numerical_scores):
        row["_numerical_score"] = score
    for row, score in zip(rows, structural_scores):
        row["_structural_score"] = score
    numerical_ranks = _rank(rows, "_numerical_score")
    structural_ranks = _rank(rows, "_structural_score")
    for index, row in enumerate(rows):
        row["visual_rank"] = visual_ranks.get(index, "")
        row["numerical_rank"] = numerical_ranks.get(index, "")
        row["structural_rank"] = structural_ranks.get(index, "")
    summary_path = output_root / "protocol_v4_integrated_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize integrated Protocol V4 results.")
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    print(f"Wrote {write_integrated_summary(parse_args().root)}")


if __name__ == "__main__":
    main()
