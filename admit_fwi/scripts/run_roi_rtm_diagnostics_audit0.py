from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from admit_fwi.diagnostics.admit_common import file_hash, git_commit, markdown_table, now_utc, text_hash, write_csv, write_json
from admit_fwi.diagnostics.admit_common import read_csv_dicts
from admit_fwi.diagnostics.roi_rtm_diagnostics import compute_roi_rows


METHODS = ["initial", "full_fwi", "global", "illumination", "consensus", "depth", "inverse", "ecg", "random_seed_4", "random_seed_0"]


def run(output_dir: Path, smoke: bool = False) -> dict:
    root = Path(__file__).resolve().parents[1]
    fwi_dir = root / "outputs" / "FWI" / "full_salt_fwi_cg_audit0_train_ecg_v1"
    gate_root = root / "outputs" / "salt_reliability_gate_audit0_v1"
    rtm_dir = root / "outputs" / "RTM" / "audit0_gate_rtm_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    for name in ["roi_metrics.csv", "roi_metrics.md"]:
        path = output_dir / name
        if path.exists():
            path.replace(output_dir / name.replace(".", ".previous.", 1))
    rows = compute_roi_rows(fwi_dir=fwi_dir, gate_root=gate_root, rtm_dir=rtm_dir, methods=METHODS)
    split_path = root / "outputs" / "admit_fwi_v1" / "seg_salt_main_case" / "split_consistency" / "split_metrics.csv"
    split_by_method = {row["method"]: row for row in read_csv_dicts(split_path)} if split_path.exists() else {}
    for row in rows:
        split = split_by_method.get(row["method"], {})
        row["rtm_split_correlation"] = split.get("rtm_split_correlation", row.get("rtm_split_correlation", ""))
        row["rtm_split_laplacian_correlation"] = split.get("rtm_split_laplacian_correlation", "")
        row["local_structure_tensor_coherence"] = split.get("local_structure_tensor_coherence", "")
    fields = list(rows[0])
    write_csv(output_dir / "roi_long_table.csv", rows, fields)
    write_csv(output_dir / "roi_metrics.csv", rows, fields)
    md = [
        "# ROI-based Gate RTM Diagnostics",
        "",
        "1. `salt_top`, `salt_flanks`, and `subsalt_shadow` are synthetic benchmark posterior evaluation regions.",
        "2. Truth-free proxy regions are the only regions with future field-data transfer meaning.",
        "3. Short-record subsalt values must not be written as deep subsalt imaging improvement.",
        "",
        markdown_table(rows[:80], fields),
    ]
    (output_dir / "roi_metrics.md").write_text("\n".join(md), encoding="utf-8")
    manifest = {
        "status": "READY",
        "timestamp_utc": now_utc(),
        "git_commit": git_commit(),
        "command": "run_roi_rtm_diagnostics_audit0.py --smoke" if smoke else "run_roi_rtm_diagnostics_audit0.py",
        "config_hash": text_hash("admit_seg_salt_roi_v1"),
        "input_hash": file_hash(gate_root / "audit" / "audit_method_summary.csv"),
        "model_source": "SEG/Salt synthetic benchmark; truth-aware masks are TRUTH_AWARE_BENCHMARK_ONLY",
        "methods": METHODS,
    }
    write_json(output_dir / "roi_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build audit0 SEG/Salt ROI diagnostics from existing gate RTM outputs.")
    parser.add_argument("--config", type=Path, default=Path("configs/admit_seg_salt_main_case.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/admit_fwi_v1/seg_salt_main_case/roi_diagnostics"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    manifest = run(args.output_dir, args.smoke)
    print(f"ROI diagnostics written to {args.output_dir} ({manifest['status']})")


if __name__ == "__main__":
    main()
