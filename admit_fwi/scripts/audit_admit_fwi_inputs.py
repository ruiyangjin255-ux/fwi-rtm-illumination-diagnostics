from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from admit_fwi.diagnostics.admit_common import markdown_table, write_csv, write_json
from admit_fwi.diagnostics.admit_input_audit import audit_inputs


def run(output_dir: Path) -> dict:
    root = Path(__file__).resolve().parents[1]
    audit = audit_inputs(root, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "input_audit.json", audit)
    write_csv(
        output_dir / "discovered_models.csv",
        audit["discovered_external_models"],
        fieldnames=["benchmark", "path", "extension", "bytes", "status"],
    )
    missing_rows = [{"model": name, "status": "MISSING_EXTERNAL_MODEL"} for name in audit["missing_external_models"]]
    write_csv(output_dir / "missing_required_inputs.csv", missing_rows, fieldnames=["model", "status"])
    md = [
        "# ADMIT-FWI Input Audit",
        "",
        f"- status: `{audit['status']}`",
        f"- git_commit: `{audit['git_commit']}`",
        f"- SEG/Salt shape: `{audit['seg_salt']['current_model_shape']}`",
        f"- can_enter_p1_seg_salt: `{audit['can_enter_p1_seg_salt']}`",
        f"- can_enter_p2_model_staircase: `{audit['can_enter_p2_model_staircase']}`",
        "",
        "## Gate Models",
        markdown_table(audit["gate_models"], ["method", "exists", "shape", "path"]),
        "## External Model Candidates",
        markdown_table(audit["discovered_external_models"], ["benchmark", "status", "path", "extension", "bytes"]),
        "## Missing External Models",
        markdown_table(missing_rows, ["model", "status"]),
    ]
    (output_dir / "input_audit.md").write_text("\n".join(md), encoding="utf-8")
    (output_dir / "missing_required_inputs.md").write_text(markdown_table(missing_rows, ["model", "status"]), encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit existing ADMIT-FWI inputs without running new FWI/RTM.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/admit_fwi_v1/input_audit"))
    args = parser.parse_args()
    audit = run(args.output_dir)
    print(f"input audit written to {args.output_dir} (P1={audit['can_enter_p1_seg_salt']}, P2={audit['can_enter_p2_model_staircase']})")


if __name__ == "__main__":
    main()
