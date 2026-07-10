from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from admit_fwi.diagnostics.admit_common import edge_mae, read_csv_dicts


METHODS = ["initial", "full_fwi", "global", "illumination", "consensus", "depth", "inverse", "ecg", "random_seed_0", "random_seed_1", "random_seed_2", "random_seed_3", "random_seed_4"]
MODEL_FILES = {
    "initial": "initial_model.npy",
    "full_fwi": "full_fwi_model.npy",
    "global": "global_matched_model.npy",
    "illumination": "illumination_only_matched_model.npy",
    "consensus": "gradient_consensus_only_matched_model.npy",
    "depth": "depth_matched_model.npy",
    "inverse": "inverse_illumination_negative_control_model.npy",
    "ecg": "ecg_reliability_gate_model.npy",
    "random_seed_0": "random_matched_seed_0_model.npy",
    "random_seed_1": "random_matched_seed_1_model.npy",
    "random_seed_2": "random_matched_seed_2_model.npy",
    "random_seed_3": "random_matched_seed_3_model.npy",
    "random_seed_4": "random_matched_seed_4_model.npy",
}
GATE_FILES = {
    "global": "global_matched.npy",
    "illumination": "illumination_only_matched.npy",
    "consensus": "gradient_consensus_only_matched.npy",
    "depth": "depth_matched.npy",
    "inverse": "inverse_illumination_negative_control.npy",
    "ecg": "ecg_reliability_gate.npy",
    "random_seed_0": "random_matched_seed_0.npy",
    "random_seed_1": "random_matched_seed_1.npy",
    "random_seed_2": "random_matched_seed_2.npy",
    "random_seed_3": "random_matched_seed_3.npy",
    "random_seed_4": "random_matched_seed_4.npy",
}


def _by_method(path: Path) -> dict[str, dict[str, str]]:
    return {row["method"]: row for row in read_csv_dicts(path)}


def _pick(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return ""


def _deep_status(deep_dir: Path) -> str:
    paths = [
        deep_dir / "wavefield_smoke" / "deep_energy_summary.json",
        deep_dir / "boundary_energy" / "boundary_energy_summary.json",
    ]
    text = " ".join(path.read_text(encoding="utf-8") for path in paths if path.exists())
    if "TIME_TRUNCATION_CONFIRMED" in text or "PML_REFLECTION_RISK" in text:
        return "NOT_RELEASED_FOR_DEEP_INTERPRETATION"
    if text:
        return "PREFLIGHT_AVAILABLE"
    return "MISSING_DEEP_TIME_PREFLIGHT"


def _verdict(method: str, split_status: str, deep_status: str) -> str:
    if split_status != "READY":
        return "INSUFFICIENT_IMAGE_CONSISTENCY_EVIDENCE"
    if deep_status == "NOT_RELEASED_FOR_DEEP_INTERPRETATION":
        if method in {"illumination", "ecg", "random_seed_4", "global"}:
            return "ACCEPTABLE_FOR_SHORT_RECORD_SHALLOW_INTERPRETATION"
        if method == "inverse":
            return "NEGATIVE_CONTROL_FAILED"
        if method.startswith("random"):
            return "RANDOM_CONTROL_INDISTINGUISHABLE"
        return "REQUIRES_IMAGE_CONSISTENCY_CHECK" if method == "full_fwi" else "NOT_SUPPORTED_FOR_DEEP_SUBSALT"
    return "REQUIRES_IMAGE_CONSISTENCY_CHECK"


def build_evidence_matrix(root: Path, split_dir: Path, roi_dir: Path) -> list[dict[str, Any]]:
    fwi_dir = root / "outputs" / "FWI" / "full_salt_fwi_cg_audit0_train_ecg_v1"
    gate_root = root / "outputs" / "salt_reliability_gate_audit0_v1"
    rtm_dir = root / "outputs" / "RTM" / "audit0_gate_rtm_v1"
    deep_dir = root / "outputs" / "deep_time_preflight_v1"
    true_model = np.load(fwi_dir / "full_salt_true_model.npy")
    initial = np.load(fwi_dir / "full_salt_initial_model.npy")
    heldout = _by_method(gate_root / "audit" / "audit_method_summary.csv")
    rtm = _by_method(rtm_dir / "gate_rtm_method_summary.csv")
    split = _by_method(split_dir / "split_metrics.csv")
    roi_rows = read_csv_dicts(roi_dir / "roi_metrics.csv")
    deep_status = _deep_status(deep_dir)
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        model_path = gate_root / "models" / MODEL_FILES[method]
        model = np.load(model_path) if model_path.exists() else None
        delta = model - initial if model is not None else None
        gate_filename = GATE_FILES.get(method)
        gate_path = gate_root / "gates" / gate_filename if gate_filename else None
        gate = np.load(gate_path) if gate_path is not None and gate_path.exists() else None
        method_roi = {row["region"]: row for row in roi_rows if row["method"] == method}
        split_status = split.get(method, {}).get("status", "MISSING")
        split_corr = split.get(method, {}).get("rtm_split_laplacian_correlation", "")
        initial_split_corr = split.get("initial", {}).get("rtm_split_laplacian_correlation", "")
        verdict = _verdict(method, split_status, deep_status)
        if split_corr not in ("", None) and initial_split_corr not in ("", None):
            try:
                if float(split_corr) < float(initial_split_corr):
                    verdict = "REQUIRES_IMAGE_CONSISTENCY_CHECK"
            except ValueError:
                pass
        allowed_claim = "Residual/model/image/deep-time domains must be audited before accepting this update."
        if method in {"illumination", "ecg", "random_seed_4"} and split_status == "READY":
            allowed_claim = "Spatial selective update is useful, but ECG is not uniquely superior."
        elif method in {"illumination", "ecg"}:
            allowed_claim = "Illumination-only is a strong baseline; ECG is an evidence-calibrated candidate."
        rows.append(
            {
                "method": method,
                "update_type": "full" if method == "full_fwi" else ("baseline" if method == "initial" else "matched_gate"),
                "update_l2": float(np.linalg.norm(delta)) if delta is not None else "",
                "active_fraction": float(np.mean(gate > 0.0)) if gate is not None else (1.0 if method == "full_fwi" else 0.0),
                "data_space_normalized_l2": _pick(heldout.get(method, {}), "normalized_l2", "normalized_l2_residual_mean"),
                "data_space_nrms": _pick(heldout.get(method, {}), "nrms", "nrms_residual_mean"),
                "data_space_trace_corr": _pick(heldout.get(method, {}), "trace_corr", "trace_correlation_mean"),
                "model_space_mae": float(np.mean(np.abs(model - true_model))) if model is not None else "",
                "model_space_rmse": float(np.sqrt(np.mean((model - true_model) ** 2))) if model is not None else "",
                "model_space_edge_mae": edge_mae(model, true_model) if model is not None else "",
                "image_space_filtered_rmse": _pick(rtm.get(method, {}), "filtered RMSE", "filtered_reference_rmse"),
                "image_space_filtered_corr": _pick(rtm.get(method, {}), "filtered corr", "filtered_reference_corr"),
                "split_consistency_corr": split_corr,
                "roi_salt_top_score": method_roi.get("salt_top", {}).get("rtm_laplacian_energy", ""),
                "roi_salt_flank_score": method_roi.get("salt_flanks", {}).get("rtm_laplacian_energy", ""),
                "roi_subsalt_score": method_roi.get("subsalt_shadow", {}).get("rtm_laplacian_energy", ""),
                "deep_time_status": deep_status,
                "overall_admissibility_verdict": verdict,
                "allowed_claim": allowed_claim,
                "forbidden_claim": "Short-record RTM ranking proves deep imaging quality; ECG significantly improves subsalt imaging.",
            }
        )
    return rows
