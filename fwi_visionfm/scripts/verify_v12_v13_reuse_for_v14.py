# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml


def _load_run(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        json.loads((run_dir / "config.json").read_text(encoding="utf-8")),
        json.loads((run_dir / "model_card.json").read_text(encoding="utf-8")),
    )


def _expected_contract(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "shot_count": int(config["shot_count"]),
        "image_size": int(config["image_size"]),
        "decoder": str(config["decoder"]),
        "loss": str(config["loss"]),
        "epochs": int(config["epochs"]),
        "metric_space": str(config["metric_space"]),
    }


def _actual_contract(run: dict[str, Any], card: dict[str, Any]) -> dict[str, Any]:
    return {
        "shot_count": int(run["shot_count"]),
        "image_size": int(run["image_size"]),
        "decoder": str(run["decoder"]),
        "loss": str(run["loss"]),
        "epochs": int(run["epochs"]),
        "metric_space": str(run["metric_space"]),
        "decoder_fully_registered": card.get("decoder_fully_registered") is True,
        "optimizer_parameters": int(card.get("optimizer_parameters", 0)),
        "trainable_parameters": int(card.get("trainable_parameters", 0)),
        "is_real_feature": bool(card.get("is_real_feature", False)),
    }


def _rows_from(root: Path, *, protocol_source: str, method_key: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.glob(f"runs/*/{method_key}/seed_*/config.json")):
        run_dir = path.parent
        run, card = _load_run(run_dir)
        rows.append(
            {
                "protocol_source": protocol_source,
                "run_id": run["run_id"],
                "source_run_dir": str(run_dir),
                "actual_contract": _actual_contract(run, card),
                "config": run,
            }
        )
    return rows


def verify_v12_v13_reuse_for_v14(*, v12_root: str | Path, v13_root: str | Path, config_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    expected = _expected_contract(config)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = _rows_from(Path(v12_root), protocol_source="protocol_v12", method_key="dinov2_frozen")
    rows.extend(_rows_from(Path(v13_root), protocol_source="protocol_v13", method_key="ncs2d_frozen"))
    result_rows = []
    for row in rows:
        reasons = []
        actual = row["actual_contract"]
        for field, value in expected.items():
            if actual.get(field) != value:
                reasons.append(field)
        if actual.get("decoder_fully_registered") is not True:
            reasons.append("decoder_fully_registered")
        if actual.get("optimizer_parameters") != actual.get("trainable_parameters"):
            reasons.append("optimizer_registration")
        if row["config"]["bridge"] != "raw_envelope_spectrum3":
            reasons.append("bridge")
        if row["config"]["method_key"] == "ncs2d_frozen" and actual.get("is_real_feature") is not True:
            reasons.append("is_real_feature")
        result_rows.append(
            {
                "run_id": row["run_id"].replace("__seed", "__B0__seed"),
                "source_run_dir": row["source_run_dir"],
                "protocol_source": row["protocol_source"],
                "reusable": not reasons,
                "reasons": reasons,
                "method_key": row["config"]["method_key"],
                "transfer_id": row["config"]["transfer_id"],
                "seed": int(row["config"]["seed"]),
            }
        )
    payload = {
        "total_runs": len(result_rows),
        "reusable_count": sum(row["reusable"] for row in result_rows),
        "rows": result_rows,
    }
    (out / "reuse_verification.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out / "reusable_runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_id", "method_key", "transfer_id", "seed", "protocol_source", "source_run_dir", "reusable", "reasons"])
        writer.writeheader()
        for row in result_rows:
            if row["reusable"]:
                writer.writerow({**row, "reasons": ";".join(row["reasons"])})
    with (out / "nonreusable_runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_id", "method_key", "transfer_id", "seed", "protocol_source", "source_run_dir", "reusable", "reasons"])
        writer.writeheader()
        for row in result_rows:
            if not row["reusable"]:
                writer.writerow({**row, "reasons": ";".join(row["reasons"])})
    lines = [
        "# V14 B0 严格复用门禁",
        "",
        f"- total runs: {len(result_rows)}",
        f"- reusable: {sum(row['reusable'] for row in result_rows)}",
        "",
        "仅允许 V12 的 M3 DINOv2 frozen 与 V13 的 M6 NCS2D frozen 进入 B0 复用候选。",
    ]
    (out / "reuse_verification.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v12-root", required=True)
    parser.add_argument("--v13-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(verify_v12_v13_reuse_for_v14(v12_root=args.v12_root, v13_root=args.v13_root, config_path=args.config, output_dir=args.output_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
