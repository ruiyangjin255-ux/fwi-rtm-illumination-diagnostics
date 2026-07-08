from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _to_string_ids(values: Any, *, count: int) -> list[str]:
    if values is None:
        return [str(index) for index in range(count)]
    array = np.asarray(values)
    if array.ndim == 0:
        return [str(array.item())]
    return [str(item) for item in array.tolist()]


def load_prediction_npz(path: str | Path) -> dict[str, Any]:
    prediction_path = Path(path)
    with np.load(prediction_path, allow_pickle=True) as payload:
        prediction_key = "velocity_pred_physical" if "velocity_pred_physical" in payload.files else "prediction"
        target_key = "velocity_true_physical" if "velocity_true_physical" in payload.files else "target"
        prediction = np.asarray(payload[prediction_key], dtype=np.float32)
        target = np.asarray(payload[target_key], dtype=np.float32)
        sample_ids = _to_string_ids(payload["sample_id"] if "sample_id" in payload.files else None, count=int(prediction.shape[0]))
        metric_space = ""
        if "metric_space" in payload.files:
            value = np.asarray(payload["metric_space"])
            metric_space = str(value.item() if value.ndim == 0 else value.tolist()[0])
        has_explicit_sample_id = "sample_id" in payload.files
    return {
        "path": str(prediction_path),
        "prediction": prediction,
        "target": target,
        "sample_id": sample_ids,
        "metric_space": metric_space,
        "has_explicit_sample_id": has_explicit_sample_id,
    }


def validate_prediction_targets(pred_a: dict[str, Any], pred_b: dict[str, Any], *, atol: float = 1.0e-5) -> dict[str, Any]:
    ids_a = list(pred_a["sample_id"])
    ids_b = list(pred_b["sample_id"])
    same_ids = ids_a == ids_b
    same_shape = tuple(pred_a["target"].shape) == tuple(pred_b["target"].shape)
    if not same_shape:
        return {
            "status": "TARGET_MISMATCH",
            "same_shape": False,
            "same_sample_id": same_ids,
            "max_target_diff": None,
        }
    if not same_ids:
        return {
            "status": "TARGET_MISMATCH",
            "same_shape": True,
            "same_sample_id": False,
            "max_target_diff": None,
        }
    max_diff = float(np.max(np.abs(np.asarray(pred_a["target"]) - np.asarray(pred_b["target"])))) if pred_a["target"].size else 0.0
    return {
        "status": "MATCH" if max_diff <= atol else "TARGET_MISMATCH",
        "same_shape": True,
        "same_sample_id": True,
        "max_target_diff": max_diff,
    }


def align_predictions_by_sample_id(pred_a: dict[str, Any], pred_b: dict[str, Any], *, atol: float = 1.0e-5) -> dict[str, Any]:
    ids_a = {sample_id: index for index, sample_id in enumerate(pred_a["sample_id"])}
    ids_b = {sample_id: index for index, sample_id in enumerate(pred_b["sample_id"])}
    common_ids = sorted(set(ids_a) & set(ids_b))
    if not common_ids:
        return {"status": "NO_COMMON_SAMPLE_ID", "sample_id": []}
    idx_a = [ids_a[sample_id] for sample_id in common_ids]
    idx_b = [ids_b[sample_id] for sample_id in common_ids]
    target_a = np.asarray(pred_a["target"])[idx_a]
    target_b = np.asarray(pred_b["target"])[idx_b]
    max_diff = float(np.max(np.abs(target_a - target_b))) if target_a.size else 0.0
    if max_diff > atol:
        return {
            "status": "TARGET_MISMATCH",
            "sample_id": common_ids,
            "max_target_diff": max_diff,
        }
    return {
        "status": "ALIGNED",
        "sample_id": common_ids,
        "prediction_a": np.asarray(pred_a["prediction"])[idx_a],
        "prediction_b": np.asarray(pred_b["prediction"])[idx_b],
        "target": target_a,
        "metric_space_a": pred_a.get("metric_space", ""),
        "metric_space_b": pred_b.get("metric_space", ""),
        "max_target_diff": max_diff,
    }
