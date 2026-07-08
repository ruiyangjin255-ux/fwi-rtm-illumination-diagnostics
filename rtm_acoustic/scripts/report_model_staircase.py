from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from rtm_acoustic.diagnostics.admit_common import file_hash, git_commit, markdown_table, now_utc, text_hash, write_csv, write_json
from rtm_acoustic.diagnostics.model_staircase_report import summarize_model_staircase


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ADMIT-FWI model staircase outputs.")
    parser.add_argument("--root", type=Path, default=Path("outputs/admit_fwi_v1/model_staircase"))
    args = parser.parse_args()
    rows = summarize_model_staircase(args.root)
    if not rows:
        raise FileNotFoundError(f"no model staircase manifests found in {args.root}")
    write_csv(args.root / "model_staircase_summary.csv", rows)
    write_csv(args.root / "failure_mode_matrix.csv", rows)
    (args.root / "model_staircase_summary.md").write_text("# Model Staircase Summary\n\n" + markdown_table(rows), encoding="utf-8")
    (args.root / "failure_mode_matrix.md").write_text("# Failure Mode Matrix\n\n" + markdown_table(rows), encoding="utf-8")
    write_json(
        args.root / "model_staircase_manifest.json",
        {
            "status": "READY",
            "timestamp_utc": now_utc(),
            "git_commit": git_commit(),
            "command": "report_model_staircase.py",
            "config_hash": text_hash("model_staircase_report_v1"),
            "input_hash": file_hash(args.root / rows[0]["model_name"] / "manifest.json"),
            "models": [row["model_name"] for row in rows],
        },
    )
    print(f"model staircase summary written to {args.root}")


if __name__ == "__main__":
    main()
