from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from admit_fwi.scripts._common import ensure_output_tree, read_simple_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Write reliability-gate smoke report.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    config = read_simple_yaml(args.config)
    output_dir = root / config.get("output_dir", "admit_fwi/outputs/salt_reliability_gate_v1")
    ensure_output_tree(output_dir)
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"status": "BLOCKED_MISSING_REPLAY_MANIFEST"}
    lines = [
        "# Reliability Gate Smoke Report",
        "",
        f"- `status`: {manifest.get('status')}",
        f"- `smoke`: {bool(args.smoke)}",
        "",
        "## Interpretation",
        "",
        "The current repository does not yet contain the per-shot-group FWI diagnostics required to compute ECG reliability gates." if manifest.get("status") != "READY" else "Required diagnostics are present; gate ablation can proceed.",
    ]
    report = output_dir / "report" / "reliability_gate_smoke_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
