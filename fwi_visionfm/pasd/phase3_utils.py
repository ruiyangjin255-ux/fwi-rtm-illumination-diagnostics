"""Utilities shared by PASD Phase-3 paper-locking commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .diagnostics import (
    edge_mask_from_threshold,
    edge_prf,
    gradient_magnitude_np,
    gradient_metrics,
    masked_mae_identity,
    simple_metrics,
)


AGGREGATION_CANDIDATES: dict[str, str] = {
    "C1_pasd_core_mean": "B4_no_geometry_attention",
    "C2_pasd_core_attention": "B4_pasd_fwi",
}


def candidate_to_variant(candidate: str) -> str:
    try:
        return AGGREGATION_CANDIDATES[candidate]
    except KeyError as exc:
        choices = ", ".join(sorted(AGGREGATION_CANDIDATES))
        raise KeyError(f"Unknown Phase-3 aggregation candidate '{candidate}'. Choose one of: {choices}") from exc


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return path


def source_threshold(train_velocities: np.ndarray, percentile: float = 90.0) -> float:
    gradients = gradient_magnitude_np(np.asarray(train_velocities, dtype=np.float32))
    return float(np.percentile(gradients.reshape(-1), percentile))


def corrected_prediction_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    true_edge_threshold: float,
    pred_edge_threshold: float | None = None,
) -> dict[str, float]:
    """Compute Phase-3 corrected structural metrics in physical velocity space."""

    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    true_mask = edge_mask_from_threshold(target, true_edge_threshold)
    if pred_edge_threshold is None:
        pred_edge_threshold = float(np.percentile(gradient_magnitude_np(prediction).reshape(-1), 90.0))
    pred_mask = edge_mask_from_threshold(prediction, pred_edge_threshold)
    out = {
        **simple_metrics(prediction, target, mask=true_mask),
        **gradient_metrics(prediction, target, mask=true_mask),
        **edge_prf(pred_mask, true_mask, tolerance_pixels=1),
    }
    masked = masked_mae_identity(prediction, target, true_mask)
    out["source_threshold_edge_MAE"] = masked["edge_MAE"]
    out["source_threshold_nonedge_MAE"] = masked["nonedge_MAE"]
    out["source_threshold_edge_coverage"] = masked["edge_coverage"]
    out["weighted_identity_error"] = masked["weighted_identity_error"]
    out["true_edge_threshold"] = float(true_edge_threshold)
    out["pred_edge_threshold"] = float(pred_edge_threshold)
    return {key: float(value) for key, value in out.items()}


def archive_sample_rows(
    archive_path: str | Path,
    train_velocities: np.ndarray,
    dataset_name: str,
    variant: str,
    seed: int,
) -> list[dict[str, Any]]:
    tau = source_threshold(train_velocities)
    rows: list[dict[str, Any]] = []
    with np.load(archive_path) as archive:
        sample_ids = archive["sample_id"]
        predictions = archive["prediction"]
        targets = archive["target"]
        pred_tau = float(np.percentile(gradient_magnitude_np(predictions).reshape(-1), 90.0))
        for idx, sample_id in enumerate(sample_ids.tolist()):
            metrics = corrected_prediction_metrics(predictions[idx], targets[idx], tau, pred_tau)
            rows.append(
                {
                    "dataset": dataset_name,
                    "variant": variant,
                    "seed": int(seed),
                    "sample_id": int(sample_id),
                    **metrics,
                }
            )
    return rows


def mean_metric_rows(rows: list[Mapping[str, Any]], group_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    metric_keys: list[str] = []
    if rows:
        ignored = set(group_keys) | {"sample_id"}
        for key in rows[0].keys():
            if key in ignored:
                continue
            try:
                float(rows[0][key])
            except (TypeError, ValueError):
                continue
            metric_keys.append(key)
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in group_keys), []).append(row)
    out: list[dict[str, Any]] = []
    for group, members in sorted(groups.items(), key=lambda item: item[0]):
        result = {key: value for key, value in zip(group_keys, group)}
        for metric in metric_keys:
            values = [float(member[metric]) for member in members if member.get(metric) not in ("", None)]
            if values:
                result[metric] = float(np.mean(values))
                result[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        out.append(result)
    return out


def select_source_candidate(candidate_means: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the locked source-val-only Phase-3 selection rule."""

    if not candidate_means:
        raise ValueError("No candidates available for source aggregation selection.")
    min_mae = min(float(row["MAE"]) for row in candidate_means)
    eligible = [row for row in candidate_means if float(row["MAE"]) <= min_mae * 1.01]

    def rank(metric: str, lower: bool) -> dict[str, int]:
        ordered = sorted(candidate_means, key=lambda row: float(row[metric]), reverse=not lower)
        return {str(row["candidate"]): idx + 1 for idx, row in enumerate(ordered)}

    ranks = {
        "MAE": rank("MAE", True),
        "SSIM": rank("SSIM", False),
        "edge_F1": rank("edge_F1", False),
        "gradient_l1_edge": rank("gradient_l1_edge", True),
    }
    scored: list[dict[str, Any]] = []
    for row in eligible:
        candidate = str(row["candidate"])
        score = sum(ranks[metric][candidate] for metric in ranks)
        scored.append(
            {
                **dict(row),
                "rank_score": int(score),
                "eligible_mae_within_1pct": True,
            }
        )
    scored.sort(
        key=lambda row: (
            int(row["rank_score"]),
            float(row["MAE"]),
            -float(row["SSIM"]),
            float(row["gradient_l1_edge"]),
            0 if row["candidate"] == "C1_pasd_core_mean" else 1,
        )
    )
    selected = scored[0]
    return {
        "selected_candidate": selected["candidate"],
        "selected_variant": candidate_to_variant(str(selected["candidate"])),
        "selection_rule": "MAE <= min_MAE*1.01 on source_val, then rank(MAE)+rank(SSIM)+rank(edge_F1)+rank(gradient_l1_edge); ties prefer lower MAE, higher SSIM, lower gradient_l1_edge, then C1.",
        "min_source_val_MAE": float(min_mae),
        "candidate_scores": scored,
        "all_candidate_means": [dict(row) for row in candidate_means],
        "target_access": "forbidden; only source train/val arrays are loaded during selection",
    }
