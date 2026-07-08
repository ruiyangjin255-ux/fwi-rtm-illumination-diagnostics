"""Audit FlatFault-A target data and create the locked Phase-2 protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .data import inspect_array_source, load_arrays
from .protocol import DatasetRef, ProtocolManifest, load_protocol


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_target_dir(root: Path, target_family: str) -> Path | None:
    normalized = target_family.lower().replace("-", "_")
    candidates = sorted(path for path in (root / "data").iterdir() if path.is_dir() and normalized in path.name.lower())
    return candidates[0] if candidates else None


def audit_target_data(source_protocol: str | Path, target_family: str, output: str | Path) -> dict[str, Any]:
    project = Path.cwd()
    source_protocol = Path(source_protocol)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    source = load_protocol(source_protocol)
    locked_config = project / "configs" / "pasd_phase1b_locked_config.json"
    target_dir = _find_target_dir(project, target_family)
    report: dict[str, Any] = {
        "status": "FAIL",
        "target_family": target_family,
        "source_protocol_reference": str(source_protocol),
        "source_protocol_sha256": _sha256(source_protocol),
        "locked_config_reference": str(locked_config),
        "locked_config_hash": json.loads(locked_config.read_text(encoding="utf-8")).get("config_hash") if locked_config.exists() else None,
    }
    if target_dir is None:
        report["failure_reason"] = f"No real data directory found for {target_family} under {project / 'data'}."
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return report
    info = inspect_array_source(target_dir, target_dir)
    report.update(
        {
            "FlatFault-A records path": str(target_dir),
            "FlatFault-A velocity path": str(target_dir),
            "records shape": info["records_shape"],
            "velocity shape": info["velocity_shape"],
            "records dtype": info["dtype_records"],
            "velocity dtype": info["dtype_velocity"],
            "records layout detected": info["records_layout_detected"],
            "velocity layout detected": info["velocity_layout_detected"],
            "available sample count": info["sample_count"],
            "target sample_id source": "generated 0..N-1 from sorted npz files",
            "target geometry availability": "not provided; deterministic_fallback",
            "target normalization before PASD processing": "raw records are robust-normalized inside bridge; velocity uses FlatVel-A source-train scaler only",
        }
    )
    if int(info["sample_count"]) < 75:
        report["failure_reason"] = "FlatFault-A has fewer than 75 samples."
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return report
    try:
        src_bundle = load_arrays(source.source.records, source.source.models, max_samples=max(source.train_indices) + 1, family=source.source.family)
        tgt_bundle = load_arrays(target_dir, target_dir, max_samples=75, family=target_family)
        if src_bundle.velocities.shape[-2:] != tgt_bundle.velocities.shape[-2:]:
            report["failure_reason"] = "Target velocity grid shape is incompatible with source velocity grid."
            out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            return report
        report["source_trained_scaler_compatibility"] = True
    except Exception as exc:
        report["failure_reason"] = f"Compatibility check failed: {exc}"
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return report

    target_ref = DatasetRef(str(target_dir.resolve()), str(target_dir.resolve()), target_family)
    protocol = ProtocolManifest(
        version="pasd_phase2_locked_protocol_v1",
        source=source.source,
        target=target_ref,
        train_indices=source.train_indices,
        val_indices=source.val_indices,
        in_family_test_indices=source.in_family_test_indices,
        cross_family_test_indices=tuple(range(75)),
        seed=0,
        notes="Phase-2 FlatVel-A to FlatFault-A locked external validation; FlatFault-A evaluation only.",
        metadata={
            **(source.metadata or {}),
            "source_protocol_reference": str(source_protocol),
            "source_protocol_sha256": _sha256(source_protocol),
            "locked_config_reference": str(locked_config),
            "locked_config_hash": report["locked_config_hash"],
            "target_family": target_family,
            "target_selection_role": "evaluation_only",
            "target_role": "cross_family_test_only",
            "edge_mask_mode": "source_threshold_strict_gt",
            "gradient_metric_definition": "inverse-transformed physical velocity; dx=dz=1 grid_cell; strict edge mask gradient_magnitude > tau",
            "dx": 1.0,
            "dz": 1.0,
            "velocity_scaler_source": "FlatVel-A source-train only",
            "scaler_fit_split": "source.train",
            "geometry_mode": "deterministic_fallback",
            "target": info,
        },
    )
    protocol_path = project / "protocols" / "pasd_phase2_locked_flatvel_a_to_flatfault_a.json"
    protocol.save(protocol_path)
    report["status"] = "PASS"
    report["created_protocol"] = str(protocol_path)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out.parent / "repository_audit.md").write_text(
        "# PASD Phase-2 Repository Audit\n\n"
        f"- Project root: `{project}`\n"
        f"- Source protocol: `{source_protocol}`\n"
        f"- Target data: `{target_dir}`\n"
        "- Phase-2 code remains isolated in `fwi_visionfm.pasd`.\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit target data and create Phase-2 protocol.")
    parser.add_argument("--source-protocol", required=True)
    parser.add_argument("--target-family", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit_target_data(args.source_protocol, args.target_family, args.output)
    print(json.dumps({"status": result["status"], "output": args.output}, ensure_ascii=False))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
