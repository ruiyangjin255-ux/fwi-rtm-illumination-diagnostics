"""Metric provenance classification for PASD Phase-3R."""

from __future__ import annotations

from .corrected_metrics import DEPRECATED_METRIC_FIELDS, OFFICIAL_CORRECTED_METRICS


def classify_metric_field(field: str) -> str:
    if field in OFFICIAL_CORRECTED_METRICS:
        return "official_corrected_metric"
    if field in DEPRECATED_METRIC_FIELDS:
        return "deprecated_archive_metric"
    if field.lower() in {"edge_mae", "gradient_error"}:
        return "deprecated_archive_metric"
    return "unknown_metric_source"


def deprecated_metric_payload() -> dict[str, object]:
    return {
        "deprecated_metric_fields": list(DEPRECATED_METRIC_FIELDS),
        "rules": [
            "edge_MAE(old archive metric) is deprecated unless explicitly recomputed with source_threshold_strict_gt.",
            "gradient_error(old archive metric) is deprecated because it does not encode the Phase-3R edge mask definition.",
            "Any edge metric without strict > tau provenance is deprecated.",
            "Any gradient metric without inverse-transform physical velocity provenance is deprecated.",
            "Any target-adaptive threshold metric is deprecated.",
        ],
    }
