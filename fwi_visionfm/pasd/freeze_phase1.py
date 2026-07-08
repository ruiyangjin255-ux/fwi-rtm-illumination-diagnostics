"""Freeze Phase-1 outputs by hashing archives and recording run state without modifying them."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .diagnostics import sha256_file


def _git_status(cwd: Path) -> dict[str, Any]:
    try:
        result = subprocess.run(["git", "status", "--short"], cwd=str(cwd), text=True, capture_output=True, check=False)
        return {"git_available": result.returncode == 0, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except Exception as exc:
        return {"git_available": False, "error": str(exc)}


def freeze_phase1(phase1_root: str | Path, output: str | Path) -> dict[str, Any]:
    root = Path(phase1_root)
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            item = {"path": str(path), "relative_path": str(path.relative_to(root)), "size": path.stat().st_size}
            if path.suffix.lower() in {".json", ".csv", ".npz", ".md", ".pt"}:
                item["sha256"] = sha256_file(path)
            files.append(item)
    runs = []
    for summary in sorted(root.glob("*/seed_*/metrics_summary.json")):
        payload = json.loads(summary.read_text(encoding="utf-8"))
        runs.append({
            "variant": payload.get("variant"),
            "seed": payload.get("seed"),
            "status": payload.get("status"),
            "run_dir": str(summary.parent),
            "config_hash": sha256_file(summary),
            "prediction_archives": [
                {"path": str(p), "sha256": sha256_file(p)}
                for p in sorted(summary.parent.glob("predictions_*.npz"))
            ],
            "metrics_summary_path": str(summary),
        })
    protocol = root / "protocol_manifest.json"
    manifest = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "phase1_root": str(root.resolve()),
        "file_count": len(files),
        "files": files,
        "runs": runs,
        "protocol": {"path": str(protocol), "sha256": sha256_file(protocol)} if protocol.exists() else None,
        "metrics_files": [str(p) for p in sorted(root.rglob("metrics*.csv"))] + [str(p) for p in sorted(root.rglob("metrics_summary.json"))],
        "bootstrap_files": [str(p) for p in sorted((root / "bootstrap").glob("*"))] if (root / "bootstrap").exists() else [],
        "code_version": _git_status(Path.cwd()),
        "git_status_before_phase1b": _git_status(Path.cwd()),
    }
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze PASD Phase-1 outputs.")
    parser.add_argument("--phase1-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = freeze_phase1(args.phase1_root, args.output)
    print(json.dumps({"status": "SUCCESS", "runs": len(manifest["runs"]), "output": args.output}, ensure_ascii=False))


if __name__ == "__main__":
    main()
