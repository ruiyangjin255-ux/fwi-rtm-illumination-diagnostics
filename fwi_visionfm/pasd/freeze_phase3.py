"""Freeze Phase-3 input state without mutating historical PASD outputs."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .diagnostics import sha256_file
from .phase3_utils import write_json


def _hash_tree(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return [{"path": str(root), "exists": False}]
    rows: list[dict[str, Any]] = []
    suffixes = {".json", ".csv", ".md", ".npz", ".pt", ".png", ".pdf"}
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes):
        rows.append(
            {
                "path": str(path),
                "relative_path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def _git_audit(cwd: Path) -> dict[str, Any]:
    def run(args: list[str]) -> str:
        completed = subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)
        return (completed.stdout or completed.stderr).strip()

    return {
        "rev_parse_head": run(["git", "rev-parse", "HEAD"]),
        "status_short": run(["git", "status", "--short"]),
        "diff_stat": run(["git", "diff", "--stat"]),
    }


def freeze_phase3(phase1b_root: Path, phase2_root: Path, locked_config: Path, output: Path) -> Path:
    if any(part in {"pasd_phase1", "pasd_phase1b", "pasd_phase2_flatfault"} for part in output.parts):
        raise ValueError("Phase-3 freeze output must not be written into historical PASD output roots.")
    cwd = Path.cwd()
    manifest = {
        "status": "FROZEN",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "PASD-FWI Phase-3",
        "output_root": str(output.parent),
        "locked_config": {
            "path": str(locked_config),
            "sha256": sha256_file(locked_config),
        },
        "phase1b_root": str(phase1b_root),
        "phase1b_artifacts": _hash_tree(phase1b_root),
        "phase2_root": str(phase2_root),
        "phase2_artifacts": _hash_tree(phase2_root),
        "repository_audit": _git_audit(cwd),
        "historical_output_policy": "read-only reference; Phase-3 writes only under outputs/pasd_phase3_paper plus locked config/protocol files requested by the Phase-3 document.",
    }
    write_json(output, manifest)
    audit = output.parent / "repository_audit.md"
    audit.write_text(
        "\n".join(
            [
                "# PASD Phase-3 Repository Audit",
                "",
                f"- Created UTC: {manifest['created_utc']}",
                f"- Locked config: `{locked_config}`",
                f"- Locked config sha256: `{manifest['locked_config']['sha256']}`",
                f"- Phase-1b root: `{phase1b_root}` ({len(manifest['phase1b_artifacts'])} hashed files)",
                f"- Phase-2 root: `{phase2_root}` ({len(manifest['phase2_artifacts'])} hashed files)",
                f"- Git HEAD: `{manifest['repository_audit']['rev_parse_head']}`",
                "",
                "## Git Status",
                "```",
                str(manifest["repository_audit"]["status_short"]),
                "```",
                "",
                "## Diff Stat",
                "```",
                str(manifest["repository_audit"]["diff_stat"]),
                "```",
            ]
        ),
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1b-root", required=True, type=Path)
    parser.add_argument("--phase2-root", required=True, type=Path)
    parser.add_argument("--locked-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    path = freeze_phase3(args.phase1b_root, args.phase2_root, args.locked_config, args.output)
    print(json.dumps({"status": "SUCCESS", "output": str(path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
