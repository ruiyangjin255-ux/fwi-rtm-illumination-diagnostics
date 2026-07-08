from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from fwi_visionfm.evaluation.visual_score import add_visual_scores


SUMMARY_FIELDS = [
    "bridge_name",
    "model_name",
    "decoder_name",
    "loss_name",
    "seed",
    "status",
    "MAE",
    "RMSE",
    "SSIM",
    "gradient_error",
    "edge_MAE",
    "visual_score",
    "skip_reason",
    "run_dir",
]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def collect_bridge_rows(root: str | Path) -> list[dict[str, Any]]:
    rows = []
    for config_path in sorted(Path(root).rglob("config.json")):
        if "manifests" in config_path.parts:
            continue
        config = _read_json(config_path)
        if config.get("protocol") != "protocol_v4_bridge_selection" and "bridge_name" not in config:
            continue
        metrics = _read_json(config_path.parent / "metrics_cross_family_test.json")
        rows.append(
            {
                "bridge_name": config.get("bridge_name") or config.get("bridge", ""),
                "model_name": config.get("model_name", ""),
                "decoder_name": config.get("decoder_name", ""),
                "loss_name": config.get("loss_name", ""),
                "seed": config.get("seed", ""),
                "status": config.get("status", ""),
                "MAE": metrics.get("mae", ""),
                "RMSE": metrics.get("rmse", ""),
                "SSIM": metrics.get("ssim", ""),
                "gradient_error": metrics.get("gradient_error", ""),
                "edge_MAE": metrics.get("edge_mae", ""),
                "skip_reason": config.get("skip_reason", ""),
                "run_dir": str(config_path.parent),
            }
        )
    success = [row for row in rows if row.get("status") == "SUCCESS"]
    scored = add_visual_scores(success)
    by_bridge = {row["bridge_name"]: row for row in scored}
    for row in rows:
        row["visual_score"] = by_bridge.get(row["bridge_name"], {}).get("visual_score", "")
    return rows


def write_bridge_selection_summary(root: str | Path) -> dict[str, Path]:
    output_root = Path(root)
    rows = collect_bridge_rows(output_root)
    summary_path = output_root / "bridge_selection_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    ranking_path = output_root / "bridge_selection_ranking.csv"
    ranked = sorted(
        [row for row in rows if row.get("status") == "SUCCESS"],
        key=lambda row: float(row["visual_score"]) if row.get("visual_score") != "" else -1.0,
        reverse=True,
    )
    with ranking_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(ranked)
    return {"summary": summary_path, "ranking": ranking_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Protocol V4 bridge selection.")
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    paths = write_bridge_selection_summary(parse_args().root)
    for path in paths.values():
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
