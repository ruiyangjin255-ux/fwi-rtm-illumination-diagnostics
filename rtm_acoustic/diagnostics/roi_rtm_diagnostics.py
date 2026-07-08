from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rtm_acoustic.diagnostics.admit_common import edge_mae, image_stats, read_csv_dicts


METHOD_TO_GATE = {
    "global": "global_matched.npy",
    "illumination": "illumination_only_matched.npy",
    "consensus": "gradient_consensus_only_matched.npy",
    "depth": "depth_matched.npy",
    "inverse": "inverse_illumination_negative_control.npy",
    "ecg": "ecg_reliability_gate.npy",
    "random_seed_0": "random_matched_seed_0.npy",
    "random_seed_4": "random_matched_seed_4.npy",
}


def percentile_mask(arr: np.ndarray, *, low: float | None = None, high: float | None = None) -> np.ndarray:
    values = np.asarray(arr, dtype=float)
    mask = np.ones(values.shape, dtype=bool)
    if low is not None:
        mask &= values <= np.percentile(values, low)
    if high is not None:
        mask &= values >= np.percentile(values, high)
    return mask


def build_regions(true_model: np.ndarray, illumination: np.ndarray, consensus: np.ndarray | None = None) -> dict[str, dict[str, Any]]:
    nz, nx = true_model.shape
    salt = true_model >= max(3500.0, float(np.percentile(true_model, 85.0)))
    top = np.zeros_like(salt)
    flanks = np.zeros_like(salt)
    for x in range(nx):
        idx = np.flatnonzero(salt[:, x])
        if idx.size:
            z0 = int(idx[0])
            top[max(0, z0 - 2) : min(nz, z0 + 3), x] = True
            flanks[idx, x] = True
    flanks &= ~top
    subsalt = np.zeros_like(salt)
    for x in range(nx):
        idx = np.flatnonzero(salt[:, x])
        if idx.size:
            subsalt[min(nz, int(idx[-1]) + 1) :, x] = True
    deep_roi = np.zeros_like(salt)
    deep_roi[int(0.65 * nz) :, :] = True
    regions = {
        "salt_top": {"mask": top, "region_type": "truth-aware", "claim_scope": "TRUTH_AWARE_BENCHMARK_ONLY"},
        "salt_flanks": {"mask": flanks, "region_type": "truth-aware", "claim_scope": "TRUTH_AWARE_BENCHMARK_ONLY"},
        "subsalt_shadow": {"mask": subsalt, "region_type": "truth-aware", "claim_scope": "TRUTH_AWARE_BENCHMARK_ONLY"},
        "shallow_high_illumination": {"mask": percentile_mask(illumination, high=75.0) & ~deep_roi, "region_type": "truth-free", "claim_scope": "FIELD_TRANSFERABLE_PROXY"},
        "low_illumination": {"mask": percentile_mask(illumination, low=25.0), "region_type": "truth-free", "claim_scope": "FIELD_TRANSFERABLE_PROXY"},
        "deep_roi": {"mask": deep_roi, "region_type": "truth-free", "claim_scope": "FIELD_TRANSFERABLE_PROXY"},
    }
    if consensus is not None:
        regions["low_consensus"] = {"mask": percentile_mask(consensus, low=25.0), "region_type": "truth-free", "claim_scope": "FIELD_TRANSFERABLE_PROXY"}
    return regions


def _heldout_by_method(path: Path) -> dict[str, dict[str, str]]:
    return {row["method"]: row for row in read_csv_dicts(path)}


def _pick(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return ""


def compute_roi_rows(
    *,
    fwi_dir: Path,
    gate_root: Path,
    rtm_dir: Path,
    methods: list[str],
) -> list[dict[str, Any]]:
    true_model = np.load(fwi_dir / "full_salt_true_model.npy")
    initial = np.load(fwi_dir / "full_salt_initial_model.npy")
    illumination = np.load(gate_root / "diagnostics" / "illumination_score.npy")
    consensus_path = gate_root / "diagnostics" / "gradient_consensus.npy"
    consensus = np.load(consensus_path) if consensus_path.exists() else None
    descent_path = gate_root / "diagnostics" / "descent_alignment.npy"
    descent = np.load(descent_path) if descent_path.exists() else None
    ecg_path = gate_root / "diagnostics" / "ecg_reliability_score.npy"
    ecg = np.load(ecg_path) if ecg_path.exists() else None
    heldout = _heldout_by_method(gate_root / "audit" / "audit_method_summary.csv")
    regions = build_regions(true_model, illumination, consensus)

    rows: list[dict[str, Any]] = []
    for method in methods:
        model_path = gate_root / "models" / {
            "initial": "initial_model.npy",
            "full_fwi": "full_fwi_model.npy",
            "global": "global_matched_model.npy",
            "illumination": "illumination_only_matched_model.npy",
            "consensus": "gradient_consensus_only_matched_model.npy",
            "depth": "depth_matched_model.npy",
            "inverse": "inverse_illumination_negative_control_model.npy",
            "ecg": "ecg_reliability_gate_model.npy",
            "random_seed_0": "random_matched_seed_0_model.npy",
            "random_seed_4": "random_matched_seed_4_model.npy",
        }[method]
        model = np.load(model_path) if model_path.exists() else None
        update = (model - initial) if model is not None else None
        gate_filename = METHOD_TO_GATE.get(method)
        gate_path = gate_root / "gates" / gate_filename if gate_filename else None
        gate = np.load(gate_path) if gate_path is not None and gate_path.exists() else None
        rtm_lap_path = rtm_dir / method / "rtm_laplacian_filtered_physical.npy"
        rtm_norm_path = rtm_dir / method / "rtm_source_normalized_physical.npy"
        rtm_lap = np.load(rtm_lap_path) if rtm_lap_path.exists() else None
        rtm_norm = np.load(rtm_norm_path) if rtm_norm_path.exists() else None

        for region_name, region in regions.items():
            mask = region["mask"]
            lap_stats = image_stats(rtm_lap, mask) if rtm_lap is not None else {"energy": "", "abs_mean": "", "abs_p95": ""}
            norm_stats = image_stats(rtm_norm, mask) if rtm_norm is not None else {"energy": "", "abs_mean": "", "abs_p95": ""}
            row = {
                "method": method,
                "region": region_name,
                "region_type": region["region_type"],
                "claim_scope": region["claim_scope"],
                "pixels": int(np.sum(mask)),
                "rtm_laplacian_energy": lap_stats["energy"],
                "rtm_source_normalized_energy": norm_stats["energy"],
                "rtm_abs_mean": lap_stats["abs_mean"],
                "rtm_abs_p95": lap_stats["abs_p95"],
                "rtm_split_correlation": "",
                "active_gate_ratio": float(np.mean(gate[mask] > 0.0)) if gate is not None and mask.any() else (1.0 if method == "full_fwi" else 0.0),
                "update_l2": float(np.linalg.norm(update[mask])) if update is not None and mask.any() else "",
                "update_abs_mean": float(np.mean(np.abs(update[mask]))) if update is not None and mask.any() else "",
                "illumination_mean": float(np.mean(illumination[mask])) if mask.any() else "",
                "consensus_mean": float(np.mean(consensus[mask])) if consensus is not None and mask.any() else "",
                "descent_mean": float(np.mean(descent[mask])) if descent is not None and mask.any() else "",
                "ecg_score_mean": float(np.mean(ecg[mask])) if ecg is not None and mask.any() else "",
                "model_mae": float(np.mean(np.abs(model[mask] - true_model[mask]))) if model is not None and region["region_type"] == "truth-aware" and mask.any() else "",
                "model_rmse": float(np.sqrt(np.mean((model[mask] - true_model[mask]) ** 2))) if model is not None and region["region_type"] == "truth-aware" and mask.any() else "",
                "edge_mae": edge_mae(model, true_model, mask) if model is not None and region["region_type"] == "truth-aware" and mask.any() else "",
                "gradient_mae": edge_mae(model, true_model, mask) if model is not None and region["region_type"] == "truth-aware" and mask.any() else "",
                "heldout_nrms": _pick(heldout.get(method, {}), "nrms", "nrms_residual_mean"),
                "heldout_trace_corr": _pick(heldout.get(method, {}), "trace_corr", "trace_correlation_mean"),
            }
            rows.append(row)
    return rows
