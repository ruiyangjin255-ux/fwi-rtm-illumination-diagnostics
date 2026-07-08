from __future__ import annotations

from typing import Any


LOWER_IS_BETTER = ("MAE", "RMSE", "gradient_error", "edge_MAE")
HIGHER_IS_BETTER = ("SSIM",)


def _float(value: Any) -> float | None:
    try:
        if value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize(values: list[float], value: float, *, higher_is_better: bool) -> float:
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return 1.0
    scaled = (value - lo) / (hi - lo)
    return scaled if higher_is_better else 1.0 - scaled


def add_visual_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored = [dict(row) for row in rows]
    metric_values: dict[str, list[float]] = {}
    for metric in (*LOWER_IS_BETTER, *HIGHER_IS_BETTER):
        metric_values[metric] = [float(value) for value in (_float(row.get(metric)) for row in scored) if value is not None]
    for row in scored:
        components: list[float] = []
        for metric in LOWER_IS_BETTER:
            value = _float(row.get(metric))
            if value is not None and metric_values[metric]:
                components.append(_normalize(metric_values[metric], value, higher_is_better=False))
        for metric in HIGHER_IS_BETTER:
            value = _float(row.get(metric))
            if value is not None and metric_values[metric]:
                components.append(_normalize(metric_values[metric], value, higher_is_better=True))
        row["visual_score"] = float(sum(components) / len(components)) if components else 0.0
    return scored
