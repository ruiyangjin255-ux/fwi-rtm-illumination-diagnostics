from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from rtm_acoustic.diagnostics.admit_common import git_commit, now_utc, text_hash, write_json


ASSETS = {
    "Table3_SEG_Salt_evidence_matrix.csv": Path("outputs/admit_fwi_v1/seg_salt_main_case/evidence_matrix/admit_evidence_matrix.csv"),
    "Table4_ROI_diagnostics.csv": Path("outputs/admit_fwi_v1/seg_salt_main_case/roi_diagnostics/roi_metrics.csv"),
    "Table5_model_staircase.csv": Path("outputs/admit_fwi_v1/model_staircase/model_staircase_summary.csv"),
    "Table6_claim_evidence_limitation.md": Path("outputs/admit_fwi_v1/seg_salt_main_case/evidence_matrix/claim_evidence_limitation_matrix.md"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect ADMIT-FWI paper tables and notes.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/admit_fwi_v1/paper_assets"))
    args = parser.parse_args()
    for child in ["figures", "tables", "manuscript_notes"]:
        (args.output_dir / child).mkdir(parents=True, exist_ok=True)
    copied = []
    for name, src in ASSETS.items():
        if src.exists():
            dst = args.output_dir / ("tables" if name.lower().endswith(".csv") else "manuscript_notes") / name
            shutil.copy2(src, dst)
            copied.append({"name": name, "source": str(src), "dest": str(dst)})
    notes = [
        "# ADMIT-FWI Paper Asset Notes",
        "",
        "Required figure plan:",
        "",
        "1. ADMIT-FWI framework workflow.",
        "2. SEG/Salt data/model/image/deep-time evidence matrix.",
        "3. SEG/Salt matched-budget gate maps.",
        "4. SEG/Salt held-out residual comparison.",
        "5. SEG/Salt RTM split consistency and ROI diagnostics.",
        "6. Deep-time preflight decision.",
        "7. Model staircase.",
        "8. Failure mode matrix.",
        "",
        "Do not claim ECG significantly improves deep subsalt imaging.",
        "",
        "Can write:",
        "",
        "- ADMIT-FWI includes true split-RTM image consistency for the SEG/Salt short-record case when split metrics are READY.",
        "- Spatial selective gates can be audited against global, inverse, and random controls.",
        "- Illumination-only remains a strong baseline.",
        "- ECG is an evidence-calibrated candidate, but its superiority over illumination must be supported by split and ROI metrics.",
        "",
        "Forbidden:",
        "",
        "- ECG significantly improves FWI/RTM imaging.",
        "- ADMIT-FWI solves subsalt velocity building.",
        "- Short-record split RTM proves deep imaging quality.",
        "- Full FWI is most reliable only because residual is lowest.",
        "",
        "Split consistency limits:",
        "",
        "- Split consistency is image-domain stability, not velocity accuracy.",
        "- Current split RTM uses short-record nt=600 and cannot be extrapolated to deep subsalt.",
        "",
        "External model status:",
        "",
        "- Marmousi/Sigsbee/BP remain missing unless local files are provided; do not report them as completed.",
    ]
    (args.output_dir / "manuscript_notes" / "admit_paper_asset_notes.md").write_text("\n".join(notes), encoding="utf-8")
    write_json(
        args.output_dir / "paper_assets_manifest.json",
        {
            "status": "READY",
            "timestamp_utc": now_utc(),
            "git_commit": git_commit(),
            "command": "build_admit_paper_assets.py",
            "config_hash": text_hash("admit_paper_assets_v1"),
            "copied": copied,
        },
    )
    print(f"paper assets written to {args.output_dir}")


if __name__ == "__main__":
    main()
