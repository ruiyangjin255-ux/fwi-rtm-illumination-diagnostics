# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml


TRANSFERS = [("flatvel_a", "curvevel_a"), ("flatvel_a", "flatfault_a"), ("curvevel_a", "flatfault_a")]


def _method_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    backbones = config.get("backbones", {})
    dino = backbones.get("dinov2_frozen", {"method_name": "DINOv2 frozen"})
    ncs = backbones.get("ncs2d_frozen", {"method_name": "NCS2D frozen"})
    return [
        {"method_id": "M3", "method_key": "dinov2_frozen", "method_name": dino["method_name"]},
        {"method_id": "M6", "method_key": "ncs2d_frozen", "method_name": ncs["method_name"]},
    ]


def _bridge_specs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    bridges = config.get("bridges")
    if bridges:
        return dict(bridges)
    return {
        "B0": {"bridge_name": "raw_envelope_spectrum3", "geometry_mode": "none", "trace_context_radius": 0, "use_shot_global_context": False, "use_multiscale_context": False},
        "B1": {"bridge_name": "geometry_aware_trace_bridge_geometry", "geometry_mode": "trace_geometry_only", "trace_context_radius": 0, "use_shot_global_context": False, "use_multiscale_context": False},
        "B2": {"bridge_name": "geometry_aware_trace_bridge_context", "geometry_mode": "trace_geometry_context", "trace_context_radius": 2, "use_shot_global_context": True, "use_multiscale_context": False},
        "B3": {"bridge_name": "geometry_aware_trace_bridge_multiscale", "geometry_mode": "trace_geometry_context_multiscale", "trace_context_radius": 2, "use_shot_global_context": True, "use_multiscale_context": True},
    }


def build_protocol_v14_matrix(*, repo_root: str | Path, config_path: str | Path, geometry_audit_path: str | Path, reuse_gate_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    geometry_audit = json.loads(Path(geometry_audit_path).read_text(encoding="utf-8"))
    reuse_gate = json.loads(Path(reuse_gate_path).read_text(encoding="utf-8"))
    reusable = {row["run_id"]: row for row in reuse_gate.get("rows", []) if row.get("reusable")}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    geometry_provenance = geometry_audit.get("geometry_provenance", "UNAVAILABLE")
    for source, target in TRANSFERS:
        transfer_id = f"{source}_to_{target}"
        for seed in config["seeds"]:
            for method in _method_specs(config):
                for bridge_id, bridge in _bridge_specs(config).items():
                    run_id = f"{transfer_id}__{method['method_key']}__{bridge_id}__seed{seed}"
                    status = "PENDING"
                    skip_reason = ""
                    reused_from = ""
                    if geometry_provenance == "UNAVAILABLE":
                        status = "SKIPPED_GEOMETRY_UNAVAILABLE"
                        skip_reason = "geometry provenance unavailable"
                    elif bridge_id == "B0":
                        reuse_id = run_id
                        if reuse_id in reusable:
                            status = "REUSE_GATE_PASSED"
                            reused_from = reusable[reuse_id]["source_run_dir"]
                        else:
                            status = "REQUIRES_RERUN"
                            skip_reason = "strict reuse gate did not pass"
                    rows.append(
                        {
                            "run_id": run_id,
                            "transfer_id": transfer_id,
                            "source_family": source,
                            "target_family": target,
                            "seed": int(seed),
                            "method_id": method["method_id"],
                            "method_key": method["method_key"],
                            "method_name": method["method_name"],
                            "bridge_id": bridge_id,
                            "bridge_name": bridge["bridge_name"],
                            "geometry_mode": bridge["geometry_mode"],
                            "trace_context_radius": int(bridge["trace_context_radius"]),
                            "use_shot_global_context": bool(bridge["use_shot_global_context"]),
                            "use_multiscale_context": bool(bridge["use_multiscale_context"]),
                            "shot_count": int(config["shot_count"]),
                            "image_size": int(config["image_size"]),
                            "decoder": str(config["decoder"]),
                            "loss": str(config["loss"]),
                            "epochs": int(config["epochs"]),
                            "metric_space": str(config["metric_space"]),
                            "geometry_provenance": geometry_provenance,
                            "status": status,
                            "skip_reason": skip_reason,
                            "reused_from": reused_from,
                        }
                    )
    with (out / "protocol_v14_run_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    payload = {"run_count": len(rows), "geometry_provenance": geometry_provenance, "rows": rows}
    (out / "protocol_v14_matrix.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    pre_registered = [
        "# Protocol V14 预注册规范",
        "",
        "A. M3 B1 vs M3 B0",
        "B. M3 B2 vs M3 B1",
        "C. M3 B3 vs M3 B2",
        "D. M6 B1 vs M6 B0",
        "E. M6 B2 vs M6 B1",
        "F. M6 B3 vs M6 B2",
        "G. geometry gain comparison: (M6 B3 - M6 B0) vs (M3 B3 - M3 B0)",
    ]
    (out / "protocol_v14_pre_registered_spec.md").write_text("\n".join(pre_registered) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--geometry-audit", required=True)
    parser.add_argument("--reuse-gate", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(build_protocol_v14_matrix(repo_root=args.repo_root, config_path=args.config, geometry_audit_path=args.geometry_audit, reuse_gate_path=args.reuse_gate, output_dir=args.output_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
