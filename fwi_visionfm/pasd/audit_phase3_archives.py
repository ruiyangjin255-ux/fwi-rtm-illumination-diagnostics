"""Audit Phase-3 fresh prediction archives and historical metric provenance."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .corrected_metrics import archive_sha, stable_id_hash
from .metric_provenance import classify_metric_field, deprecated_metric_payload
from .phase3_utils import load_json, write_json


EVAL_SETS = ("in_family", "cross_curvevel_a", "cross_flatfault_a")
VARIANTS = ("B1_raw_unet", "PASD_Core_locked")
SEEDS = (0, 1, 2)


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


def _archive_path(root: Path, variant: str, seed: int, dataset: str) -> Path:
    return root / "dual_target_formal" / "prediction_archives" / f"{variant}_seed{seed}_{dataset}.npz"


def audit_phase3_archives(phase3_root: Path, locked_config: Path, dual_target_protocol: Path, output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    config = load_json(locked_config)
    protocol = load_json(dual_target_protocol)
    inventory: list[dict[str, Any]] = []
    schema: dict[str, Any] = {}
    alignment: list[dict[str, Any]] = []
    for dataset in EVAL_SETS:
        for seed in SEEDS:
            ids_by_variant: dict[str, np.ndarray] = {}
            for variant in VARIANTS:
                path = _archive_path(phase3_root, variant, seed, dataset)
                with np.load(path) as payload:
                    sample_id = np.asarray(payload["sample_id"])
                    prediction = np.asarray(payload["prediction"])
                    target = np.asarray(payload["target"])
                    ids_by_variant[variant] = sample_id.astype(np.int64)
                    inventory.append(
                        {
                            "dataset": dataset,
                            "variant": variant,
                            "seed": seed,
                            "archive_path": str(path),
                            "archive_sha256": archive_sha(path),
                            "sample_id_presence": "present",
                            "sample_id_unique": bool(np.unique(sample_id).size == sample_id.size),
                            "sample_id_count": int(sample_id.size),
                            "sample_id_hash": stable_id_hash(sample_id),
                            "prediction_shape": list(prediction.shape),
                            "target_shape": list(target.shape),
                            "prediction_dtype": str(prediction.dtype),
                            "target_dtype": str(target.dtype),
                            "velocity_scaler_metadata": "not stored in archive; checkpoint frozen separately",
                            "inverse_transform_metadata": "archive stores physical velocity predictions saved by run_single_experiment",
                            "normalization_metadata": "physical velocity archive; no normalized values used for Phase-3R metrics",
                            "records_layout_metadata": "not required for archive-only metric recomputation",
                            "dx": 1.0,
                            "dz": 1.0,
                            "edge_threshold_metadata": config.get("edge_threshold_source", "source tau parsed from locked config"),
                            "strict_edge_condition_metadata": "gradient_magnitude > tau_source",
                            "variant_config_hash": config.get("selection_decision_sha256", config.get("config_hash", "")),
                            "training_seed": seed,
                            "evaluation_target": dataset,
                        }
                    )
                    schema[str(path)] = {key: str(np.asarray(payload[key]).shape) for key in payload.files}
            same = set(ids_by_variant["B1_raw_unet"].tolist()) == set(ids_by_variant["PASD_Core_locked"].tolist())
            alignment.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "b1_count": int(ids_by_variant["B1_raw_unet"].size),
                    "pasd_count": int(ids_by_variant["PASD_Core_locked"].size),
                    "sample_id_set_identical": bool(same),
                    "sample_order_identical": bool(np.array_equal(ids_by_variant["B1_raw_unet"], ids_by_variant["PASD_Core_locked"])),
                    "sample_id_hash_b1": stable_id_hash(ids_by_variant["B1_raw_unet"]),
                    "sample_id_hash_pasd": stable_id_hash(ids_by_variant["PASD_Core_locked"]),
                }
            )
    metric_rows: list[dict[str, Any]] = []
    for csv_path in sorted((phase3_root / "dual_target_formal").glob("*.csv")) + sorted((phase3_root / "dual_target_formal" / "tables").glob("*.csv")):
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            fields = next(reader, [])
        for field in fields:
            metric_rows.append({"file": str(csv_path), "field": field, "classification": classify_metric_field(field)})
    _write_csv(output / "archive_inventory.csv", inventory)
    write_json(output / "archive_schema.json", schema)
    _write_csv(output / "metric_field_provenance.csv", metric_rows)
    _write_csv(output / "archive_alignment_check.csv", alignment)
    write_json(output / "deprecated_metric_fields.json", deprecated_metric_payload())
    report = [
        "# Phase-3R Archive Audit",
        "",
        "Phase-3 prediction archives are present for all variants, seeds, and evaluation sets.",
        "Historical `edge_MAE` and `gradient_error` fields are classified as deprecated unless recomputed in Phase-3R.",
        "CurveVel-A Phase-3 summary contained old archive metric fields for edge/gradient quantities; these are not used in Phase-3R main tables or figures.",
        "",
        f"Protocol targets: {', '.join(protocol['targets'].keys())}",
    ]
    (output / "ARCHIVE_AUDIT_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase3-root", required=True, type=Path)
    parser.add_argument("--locked-config", required=True, type=Path)
    parser.add_argument("--dual-target-protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    out = audit_phase3_archives(args.phase3_root, args.locked_config, args.dual_target_protocol, args.output)
    print(json.dumps({"status": "SUCCESS", "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
