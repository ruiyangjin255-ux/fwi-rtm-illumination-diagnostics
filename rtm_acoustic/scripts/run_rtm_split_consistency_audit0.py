from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from rtm_acoustic.diagnostics.admit_common import file_hash, git_commit, markdown_table, now_utc, safe_corr, text_hash, write_csv, write_json
from rtm_acoustic.diagnostics.roi_rtm_diagnostics import build_regions
from rtm_acoustic.diagnostics.rtm_split_consistency import split_consistency
from rtm_acoustic.scripts.run_holdout_gate_audit import MODEL_FILES


AUDIT_RTM_SHOTS = [4, 64, 124, 184, 244, 304, 364, 424, 484, 544, 604, 664]


def _find_subset_file(method_dir: Path, subset: str, kind: str) -> Path | None:
    candidates = [
        method_dir / subset / f"rtm_{kind}_physical.npy",
        method_dir / subset / f"{kind}.npy",
        method_dir / f"{subset}_{kind}.npy",
        method_dir / f"rtm_{subset}_{kind}.npy",
    ]
    return next((path for path in candidates if path.exists()), None)


def _new_layout_file(rtm_dir: Path, subset: str, method: str, kind: str) -> Path | None:
    candidates = [
        rtm_dir / subset / method / f"rtm_{kind}_physical.npy",
        rtm_dir / subset / method / f"{kind}.npy",
    ]
    return next((path for path in candidates if path.exists()), None)


def _metadata_shots(rtm_dir: Path, subset: str, method: str, fallback: list[int]) -> list[int]:
    meta_path = rtm_dir / subset / method / "rtm_metadata.json"
    if not meta_path.exists():
        return fallback
    try:
        import json

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return [int(value) for value in meta.get("shot_indices", fallback)]
    except Exception:
        return fallback


def run(output_dir: Path, rtm_dir: Path, smoke: bool = False) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    rows: list[dict[str, Any]] = []
    for method in MODEL_FILES:
        method_dir = rtm_dir / method
        a_path = _new_layout_file(rtm_dir, "subset_A", method, "laplacian_filtered") or _find_subset_file(method_dir, "subset_A", "laplacian_filtered")
        b_path = _new_layout_file(rtm_dir, "subset_B", method, "laplacian_filtered") or _find_subset_file(method_dir, "subset_B", "laplacian_filtered")
        raw_a_path = _new_layout_file(rtm_dir, "subset_A", method, "source_normalized") or _find_subset_file(method_dir, "subset_A", "source_normalized")
        raw_b_path = _new_layout_file(rtm_dir, "subset_B", method, "source_normalized") or _find_subset_file(method_dir, "subset_B", "source_normalized")
        row: dict[str, Any] = {
            "method": method,
            "status": "READY",
            "subset_A_shots": ";".join(str(s) for s in _metadata_shots(rtm_dir, "subset_A", method, AUDIT_RTM_SHOTS[0::2])),
            "subset_B_shots": ";".join(str(s) for s in _metadata_shots(rtm_dir, "subset_B", method, AUDIT_RTM_SHOTS[1::2])),
        }
        if not a_path or not b_path:
            row.update(
                {
                    "status": "MISSING_SPLIT_RTM_INPUT" if method_dir.exists() else "MISSING_RTM_CASE",
                    "rtm_split_correlation": "",
                    "rtm_split_ssim": "",
                    "rtm_split_laplacian_correlation": "",
                    "local_structure_tensor_coherence": "",
                    "salt_top_split_correlation": "",
                    "salt_flank_split_correlation": "",
                    "subsalt_shadow_split_correlation": "",
                    "deep_roi_split_correlation": "",
                    "low_illumination_split_correlation": "",
                    "low_consensus_split_correlation": "",
                }
            )
        else:
            a = np.load(a_path)
            b = np.load(b_path)
            raw_a = np.load(raw_a_path) if raw_a_path else None
            raw_b = np.load(raw_b_path) if raw_b_path else None
            metrics = split_consistency(raw_a if raw_a is not None else a, raw_b if raw_b is not None else b, a, b)
            roi_metrics = {
                "salt_top_split_correlation": "",
                "salt_flank_split_correlation": "",
                "subsalt_shadow_split_correlation": "",
                "deep_roi_split_correlation": "",
                "low_illumination_split_correlation": "",
                "low_consensus_split_correlation": "",
            }
            root = Path(__file__).resolve().parents[1]
            true_path = root / "outputs" / "FWI" / "full_salt_fwi_cg_audit0_train_ecg_v1" / "full_salt_true_model.npy"
            illum_path = root / "outputs" / "salt_reliability_gate_audit0_v1" / "diagnostics" / "illumination_score.npy"
            cons_path = root / "outputs" / "salt_reliability_gate_audit0_v1" / "diagnostics" / "gradient_consensus.npy"
            if true_path.exists() and illum_path.exists():
                regions = build_regions(np.load(true_path), np.load(illum_path), np.load(cons_path) if cons_path.exists() else None)
                region_key_map = {
                    "salt_top": "salt_top_split_correlation",
                    "salt_flanks": "salt_flank_split_correlation",
                    "subsalt_shadow": "subsalt_shadow_split_correlation",
                    "deep_roi": "deep_roi_split_correlation",
                    "low_illumination": "low_illumination_split_correlation",
                    "low_consensus": "low_consensus_split_correlation",
                }
                for region_name, key in region_key_map.items():
                    if region_name in regions:
                        mask = regions[region_name]["mask"]
                        roi_metrics[key] = safe_corr(a[mask], b[mask]) if mask.shape == a.shape and np.any(mask) else ""
            row.update(
                {
                    "rtm_split_correlation": metrics["rtm_split_correlation"],
                    "rtm_split_ssim": metrics["rtm_split_ssim"],
                    "rtm_split_laplacian_correlation": metrics["laplacian_rtm_split_correlation"],
                    "local_structure_tensor_coherence": metrics["local_structure_tensor_coherence"],
                    **roi_metrics,
                }
            )
        rows.append(row)

    fields = [
        "method",
        "status",
        "rtm_split_correlation",
        "rtm_split_ssim",
        "rtm_split_laplacian_correlation",
        "local_structure_tensor_coherence",
        "salt_top_split_correlation",
        "salt_flank_split_correlation",
        "subsalt_shadow_split_correlation",
        "deep_roi_split_correlation",
        "low_illumination_split_correlation",
        "low_consensus_split_correlation",
        "subset_A_shots",
        "subset_B_shots",
    ]
    write_csv(output_dir / "split_metrics.csv", rows, fields)
    for baseline in ["illumination", "global", "random_seed_4"]:
        base = next((r for r in rows if r["method"] == baseline), None)
        delta_rows = []
        for row in rows:
            value = row.get("rtm_split_laplacian_correlation")
            base_value = base.get("rtm_split_laplacian_correlation") if base else ""
            delta = ""
            if value != "" and base_value != "":
                delta = float(value) - float(base_value)
            delta_rows.append({"method": row["method"], "baseline": baseline, "delta_laplacian_split_corr": delta, "status": row["status"]})
        write_csv(output_dir / f"split_pairwise_vs_{baseline}.csv", delta_rows)

    md = [
        "# RTM Split Consistency Audit",
        "",
        "Interpretation limits:",
        "",
        "1. split consistency only measures image-domain stability.",
        "2. split consistency does not prove velocity-model accuracy.",
        "3. current values are short-record RTM diagnostics and cannot be extrapolated to deep subsalt interpretation.",
        "",
        markdown_table(rows, fields),
    ]
    (output_dir / "split_metrics.md").write_text("\n".join(md), encoding="utf-8")
    manifest = {
        "status": "READY",
        "timestamp_utc": now_utc(),
        "git_commit": git_commit(),
        "command": "run_rtm_split_consistency_audit0.py --smoke" if smoke else "run_rtm_split_consistency_audit0.py",
        "config_hash": text_hash("admit_seg_salt_split_consistency_v1"),
        "input_hash": file_hash(rtm_dir / "gate_rtm_manifest.json"),
        "rtm_dir": str(rtm_dir),
        "shot_split": {"subset_A": AUDIT_RTM_SHOTS[0::2], "subset_B": AUDIT_RTM_SHOTS[1::2]},
        "notes": "Rows with MISSING_SPLIT_RTM_INPUT indicate existing stacked RTM outputs cannot be used as shot-split evidence.",
    }
    write_json(output_dir / "split_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build audit0 RTM split-consistency tables from existing split RTM arrays.")
    parser.add_argument("--config", type=Path, default=Path("configs/admit_seg_salt_main_case.yaml"))
    parser.add_argument("--rtm-dir", type=Path, default=Path("outputs/RTM/audit0_gate_rtm_split_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/admit_fwi_v1/seg_salt_main_case/split_consistency"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    manifest = run(args.output_dir, args.rtm_dir, args.smoke)
    print(f"split consistency audit written to {args.output_dir}; {manifest['notes']}")


if __name__ == "__main__":
    main()
