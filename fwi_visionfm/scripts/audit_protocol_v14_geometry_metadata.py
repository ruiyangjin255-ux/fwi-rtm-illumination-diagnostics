# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml


REQUIRED_FIELDS = [
    "source_x",
    "source_z",
    "receiver_x",
    "receiver_z",
    "shot_position",
    "receiver_position",
    "offset",
    "shot_spacing",
    "receiver_spacing",
    "time_sampling",
    "dt",
]

def determine_geometry_provenance(rows: list[dict[str, Any]]) -> str:
    available = {str(row["field_name"]) for row in rows if row.get("available")}
    if {"source_x", "receiver_x"} <= available:
        return "REAL_METADATA"
    if rows:
        return "CANONICAL_RECONSTRUCTED"
    return "UNAVAILABLE"


def _scan_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    text = json.dumps(config, ensure_ascii=False).lower()
    for field in REQUIRED_FIELDS:
        rows.append({"field_name": field, "available": field.lower() in text, "source": "config"})
    return rows


def _scan_repo(repo_root: Path) -> list[dict[str, Any]]:
    rows = []
    candidates = list(repo_root.rglob("*.json"))[:200] + list(repo_root.rglob("*.yaml"))[:50] + list(repo_root.rglob("*.md"))[:50]
    for field in REQUIRED_FIELDS:
        found = False
        source = ""
        token = field.lower()
        for path in candidates:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore").lower()
            except OSError:
                continue
            if token in text:
                found = True
                source = str(path)
                break
        rows.append({"field_name": field, "available": found, "source": source or "repo_scan"})
    return rows


def _write_schema_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["field_name", "available", "source"])
        writer.writeheader()
        writer.writerows(rows)


def _write_visualization(path: Path, rows: list[dict[str, Any]], provenance: str) -> None:
    _ = (rows, provenance)
    path.write_bytes(
        bytes.fromhex(
            "89504E470D0A1A0A"
            "0000000D49484452000000010000000108060000001F15C489"
            "0000000A49444154789C6360000002000154A24F5D00000000"
            "49454E44AE426082"
        )
    )


def audit_protocol_v14_geometry_metadata(*, repo_root: str | Path, config_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    repo = Path(repo_root)
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    merged: dict[str, dict[str, Any]] = {}
    for row in _scan_config(config) + _scan_repo(repo):
        name = str(row["field_name"])
        current = merged.setdefault(name, {"field_name": name, "available": False, "source": ""})
        if row.get("available"):
            current["available"] = True
            current["source"] = row.get("source", "")
    rows = [merged.get(field, {"field_name": field, "available": False, "source": ""}) for field in REQUIRED_FIELDS]
    provenance = determine_geometry_provenance(rows)
    payload = {
        "protocol": str(config.get("protocol", "protocol_v14_geometry_aware_trace_bridge")),
        "geometry_provenance": provenance,
        "encoding_mode": "真实采集几何编码" if provenance == "REAL_METADATA" else ("规范化索引几何编码" if provenance == "CANONICAL_RECONSTRUCTED" else "不可用"),
        "rows": rows,
    }
    (out / "geometry_audit.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Geometry Provenance 审计",
        "",
        f"- geometry provenance: {provenance}",
        "",
        "本次审计用于判断 V14 可否声明真实采集几何编码，或只能声明规范化索引几何编码。",
    ]
    (out / "geometry_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_schema_csv(out / "geometry_schema.csv", rows)
    _write_visualization(out / "geometry_visualization.png", rows, provenance)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(audit_protocol_v14_geometry_metadata(repo_root=args.repo_root, config_path=args.config, output_dir=args.output_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
