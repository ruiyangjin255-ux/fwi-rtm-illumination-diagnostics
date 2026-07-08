"""Validate a fixed PASD protocol before running Phase-1 experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .protocol import load_protocol, load_protocol_bundles


def _unique(values: tuple[int, ...]) -> bool:
    return len(set(values)) == len(values)


def validate_protocol(protocol: str | Path) -> dict[str, Any]:
    manifest = load_protocol(protocol)
    source, target = load_protocol_bundles(manifest)
    train = set(manifest.train_indices)
    val = set(manifest.val_indices)
    in_family = set(manifest.in_family_test_indices)
    overlaps = {
        "train_val": sorted(train & val),
        "train_in_family": sorted(train & in_family),
        "val_in_family": sorted(val & in_family),
    }
    metadata = manifest.metadata or {}
    checks = {
        "source_split_disjoint": not any(overlaps.values()),
        "source_sample_id_unique": bool(np.unique(source.sample_ids).size == source.sample_ids.size),
        "target_sample_id_unique": True if target is None else bool(np.unique(target.sample_ids).size == target.sample_ids.size),
        "target_isolation": target is not None and metadata.get("target_role") == "cross_family_test_only",
        "scaler_source": metadata.get("scaler_fit_split") == "source.train",
        "shape_consistency": source.records.shape[0] == source.velocities.shape[0] and (target is None or target.records.shape[0] == target.velocities.shape[0]),
        "geometry_source_recorded": bool(metadata.get("geometry_mode")),
        "source_only_edge_threshold": metadata.get("edge_mask_mode") in {None, "source_threshold_strict_gt"} or "source" in str(metadata.get("edge_mask_mode")),
        "source_only_locked_config": bool(metadata.get("locked_config_hash")) if metadata.get("target_family") == "FlatFault-A" else True,
        "strict_edge_mask_condition": metadata.get("edge_mask_mode") == "source_threshold_strict_gt" if metadata.get("target_family") == "FlatFault-A" else True,
        "target_selection_role": metadata.get("target_selection_role") in {None, "evaluation_only"},
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "protocol": str(Path(protocol).resolve()),
        "checks": checks,
        "overlaps": overlaps,
        "source_records_path": manifest.source.records,
        "source_models_path": manifest.source.models,
        "target_records_path": manifest.target.records if manifest.target else None,
        "target_models_path": manifest.target.models if manifest.target else None,
        "records_shape": list(source.records.shape),
        "velocity_shape": list(source.velocities.shape),
        "target_records_shape": None if target is None else list(target.records.shape),
        "target_velocity_shape": None if target is None else list(target.velocities.shape),
        "records_layout_detected": metadata.get("source", {}).get("records_layout_detected") if isinstance(metadata.get("source"), dict) else None,
        "velocity_layout_detected": metadata.get("source", {}).get("velocity_layout_detected") if isinstance(metadata.get("source"), dict) else None,
        "geometry_mode": metadata.get("geometry_mode"),
        "metadata": metadata,
        "strict_edge_mask_condition": metadata.get("edge_mask_mode"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate PASD fixed protocol integrity.")
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = validate_protocol(args.protocol)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(output)}, ensure_ascii=False))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
