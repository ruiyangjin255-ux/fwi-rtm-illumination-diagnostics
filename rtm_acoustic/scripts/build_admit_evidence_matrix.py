from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from rtm_acoustic.diagnostics.admit_common import file_hash, git_commit, markdown_table, now_utc, text_hash, write_csv, write_json
from rtm_acoustic.diagnostics.admit_evidence_matrix import build_evidence_matrix


def run(output_dir: Path, smoke: bool = False) -> dict:
    root = Path(__file__).resolve().parents[1]
    split_dir = root / "outputs" / "admit_fwi_v1" / "seg_salt_main_case" / "split_consistency"
    roi_dir = root / "outputs" / "admit_fwi_v1" / "seg_salt_main_case" / "roi_diagnostics"
    if not (split_dir / "split_metrics.csv").exists():
        raise FileNotFoundError(f"missing split metrics: {split_dir / 'split_metrics.csv'}")
    if not (roi_dir / "roi_metrics.csv").exists():
        raise FileNotFoundError(f"missing ROI metrics: {roi_dir / 'roi_metrics.csv'}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    rows = build_evidence_matrix(root, split_dir, roi_dir)
    fields = list(rows[0])
    write_csv(output_dir / "admit_evidence_matrix.csv", rows, fields)
    (output_dir / "admit_evidence_matrix.md").write_text("# ADMIT-FWI Main Case Evidence Matrix\n\n" + markdown_table(rows, fields), encoding="utf-8")
    claim_rows = [{"method": r["method"], "allowed_claim": r["allowed_claim"], "forbidden_claim": r["forbidden_claim"], "verdict": r["overall_admissibility_verdict"]} for r in rows]
    (output_dir / "claim_evidence_limitation_matrix.md").write_text("# Claim-Evidence-Limitation Matrix\n\n" + markdown_table(claim_rows), encoding="utf-8")
    manifest = {
        "status": "READY",
        "timestamp_utc": now_utc(),
        "git_commit": git_commit(),
        "command": "build_admit_evidence_matrix.py --smoke" if smoke else "build_admit_evidence_matrix.py",
        "config_hash": text_hash("admit_evidence_matrix_v1"),
        "input_hash": file_hash(split_dir / "split_metrics.csv") + ":" + file_hash(roi_dir / "roi_metrics.csv"),
        "methods": [row["method"] for row in rows],
    }
    write_json(output_dir / "admit_main_case_summary.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SEG/Salt ADMIT-FWI evidence matrix.")
    parser.add_argument("--config", type=Path, default=Path("configs/admit_seg_salt_main_case.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/admit_fwi_v1/seg_salt_main_case/evidence_matrix"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    manifest = run(args.output_dir, args.smoke)
    print(f"evidence matrix written to {args.output_dir} ({manifest['status']})")


if __name__ == "__main__":
    main()
