"""Generate Phase-3 locked PASD-Core config and dual-target protocol."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .diagnostics import sha256_file
from .phase3_utils import candidate_to_variant, load_json, write_json


def lock_assets(
    phase1b_protocol: Path,
    phase2_protocol: Path,
    selection_decision: Path,
    locked_config: Path,
    output_config: Path,
    output_protocol: Path,
) -> tuple[Path, Path]:
    p1 = load_json(phase1b_protocol)
    p2 = load_json(phase2_protocol)
    decision = load_json(selection_decision)
    base_config = load_json(locked_config)
    selected = str(decision["selected_candidate"])
    phase3_config = {
        **base_config,
        "phase": "PASD-FWI Phase-3",
        "selected_candidate": selected,
        "selected_variant": candidate_to_variant(selected),
        "selection_decision": str(selection_decision),
        "selection_decision_sha256": sha256_file(selection_decision),
        "phase1b_locked_config": str(locked_config),
        "phase1b_locked_config_sha256": sha256_file(locked_config),
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_access_policy": "source aggregation selection used source_val only; CurveVel-A and FlatFault-A are evaluation-only targets.",
    }
    write_json(output_config, phase3_config)
    protocol = {
        "version": "pasd_phase3_dual_target_locked_v1",
        "source": p1["source"],
        "targets": {
            "CurveVel-A": {
                **p1["target"],
                "cross_family_test_indices": p1["split"]["cross_family_test"],
                "role": "evaluation_only",
            },
            "FlatFault-A": {
                **p2["target"],
                "cross_family_test_indices": p2["split"]["cross_family_test"],
                "role": "evaluation_only",
            },
        },
        "split": {
            "train": p1["split"]["train"],
            "val": p1["split"]["val"],
            "in_family_test": p1["split"]["in_family_test"],
        },
        "seed": int(p1.get("seed", 0)),
        "notes": "PASD Phase-3 locked dual target protocol. Model and aggregation selection are source-only; both targets are evaluation-only.",
        "metadata": {
            "phase1b_protocol": str(phase1b_protocol),
            "phase1b_protocol_sha256": sha256_file(phase1b_protocol),
            "phase2_protocol": str(phase2_protocol),
            "phase2_protocol_sha256": sha256_file(phase2_protocol),
            "pasd_core_config": str(output_config),
            "pasd_core_config_sha256": sha256_file(output_config),
            "selection_decision": str(selection_decision),
            "selection_decision_sha256": sha256_file(selection_decision),
            "scaler_fit_split": "source.train",
            "target_role": "evaluation_only",
            "historical_output_policy": "read-only Phase-1b/Phase-2 inputs; Phase-3 outputs written under outputs/pasd_phase3_paper.",
        },
    }
    write_json(output_protocol, protocol)
    return output_config, output_protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1b-protocol", required=True, type=Path)
    parser.add_argument("--phase2-protocol", required=True, type=Path)
    parser.add_argument("--selection-decision", required=True, type=Path)
    parser.add_argument("--locked-config", required=True, type=Path)
    parser.add_argument("--output-config", required=True, type=Path)
    parser.add_argument("--output-protocol", required=True, type=Path)
    args = parser.parse_args()
    config, protocol = lock_assets(
        args.phase1b_protocol,
        args.phase2_protocol,
        args.selection_decision,
        args.locked_config,
        args.output_config,
        args.output_protocol,
    )
    print(json.dumps({"status": "SUCCESS", "config": str(config), "protocol": str(protocol)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
