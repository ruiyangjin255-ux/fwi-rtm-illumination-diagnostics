from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from fwi_visionfm.evaluation.visual_score import add_visual_scores


FIELDS = [
    "model_type",
    "bridge",
    "mask_type",
    "decoder_name",
    "loss_name",
    "seed",
    "pretrain_epochs",
    "decoder_epochs",
    "reconstruction_loss",
    "MAE",
    "RMSE",
    "SSIM",
    "gradient_error",
    "edge_MAE",
    "visual_score",
    "status",
    "skip_reason",
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for config_path in (root / "decoder_runs").rglob("config.json"):
        config = _read_json(config_path)
        run_dir = config_path.parent
        model_type = config.get("model_type", "pretrained_local_mae")
        bridge = config.get("bridge", "")
        mask_type = config.get("mask_type", "random_patch")
        seed = str(config.get("seed", 0))
        pre_epochs = ""
        reconstruction_loss = ""
        pretrain_dir = root / "pretrain" / bridge / mask_type / f"seed_{seed}"
        if model_type == "pretrained_local_mae" and pretrain_dir.exists():
            pre_epochs = str(_read_json(pretrain_dir / "config.json").get("epochs", ""))
            reconstruction_loss = str(_read_json(pretrain_dir / "pretrain_val_metrics.json").get("reconstruction_loss", ""))
        cross = _read_json(run_dir / "metrics_cross_family_test.json")
        rows.append(
            {
                "model_type": model_type,
                "bridge": bridge,
                "mask_type": mask_type,
                "decoder_name": config.get("decoder_name", ""),
                "loss_name": config.get("loss_name", ""),
                "seed": seed,
                "pretrain_epochs": pre_epochs,
                "decoder_epochs": str(config.get("epochs", "")),
                "reconstruction_loss": reconstruction_loss,
                "MAE": str(cross.get("mae", "")),
                "RMSE": str(cross.get("rmse", "")),
                "SSIM": str(cross.get("ssim", "")),
                "gradient_error": str(cross.get("gradient_error", "")),
                "edge_MAE": str(cross.get("edge_mae", "")),
                "status": config.get("status", ""),
                "skip_reason": config.get("skip_reason", ""),
            }
        )
    return add_visual_scores(rows)


def write_local_mae_ablation_summary(root: str | Path) -> Path:
    output_root = Path(root)
    rows = _collect_rows(output_root)
    path = output_root / "local_mae_ablation_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Protocol V5 local MAE ablation results.")
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    print(f"Wrote {write_local_mae_ablation_summary(parse_args().root)}")


if __name__ == "__main__":
    main()
