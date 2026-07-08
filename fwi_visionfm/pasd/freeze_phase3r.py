"""Freeze Phase-3 inputs for Phase-3R metric repair."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .diagnostics import sha256_file
from .phase3_utils import write_json


def _git(cwd: Path, args: list[str]) -> str:
    completed = subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)
    return (completed.stdout or completed.stderr).strip()


def _files(root: Path, pattern: str) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    rows = []
    for path in sorted(root.rglob(pattern)):
        if path.is_file():
            rows.append({"path": str(path), "relative_path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return rows


def freeze_phase3r(phase3_root: Path, locked_config: Path, dual_target_protocol: Path, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "FROZEN",
        "phase": "PASD Phase-3R: Result Archive Repair and Corrected Structural Metric Recalculation",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase3_root": str(phase3_root),
        "locked_config": {"path": str(locked_config), "sha256": sha256_file(locked_config)},
        "dual_target_protocol": {"path": str(dual_target_protocol), "sha256": sha256_file(dual_target_protocol)},
        "checkpoints": _files(phase3_root / "dual_target_formal" / "checkpoints", "*.pt"),
        "prediction_archives": _files(phase3_root / "dual_target_formal" / "prediction_archives", "*.npz"),
        "historical_metrics_csv": _files(phase3_root, "*.csv"),
        "historical_bootstrap_json": _files(phase3_root / "dual_target_formal" / "bootstrap", "*.json"),
        "git": {
            "commit": _git(Path.cwd(), ["git", "rev-parse", "HEAD"]),
            "status": _git(Path.cwd(), ["git", "status", "--short"]),
        },
        "immutability_policy": "Phase-3R reads Phase-3 artifacts only and writes exclusively to outputs/pasd_phase3r_metric_repair.",
    }
    write_json(output, manifest)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase3-root", required=True, type=Path)
    parser.add_argument("--locked-config", required=True, type=Path)
    parser.add_argument("--dual-target-protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    path = freeze_phase3r(args.phase3_root, args.locked_config, args.dual_target_protocol, args.output)
    print(json.dumps({"status": "SUCCESS", "output": str(path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
