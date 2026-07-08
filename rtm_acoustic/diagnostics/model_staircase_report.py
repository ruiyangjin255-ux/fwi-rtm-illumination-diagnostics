from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def summarize_model_staircase(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        name = manifest.get("model_name", manifest_path.parent.name)
        if "layered" in name:
            failure = "NONE_OR_MINOR"
            level = 0
        elif "fault" in name:
            failure = "EDGE_LOCALIZATION_RISK"
            level = 0
        elif "marmousi" in name:
            failure = "LATERAL_COMPLEXITY_INSTABILITY"
            level = 1
        elif "sigsbee" in name:
            failure = "SUBSALT_SHADOW_LIMITED"
            level = 3
        elif "seg_salt" in name:
            failure = "TIME_WINDOW_LIMITED"
            level = 2
        else:
            failure = "INCONCLUSIVE"
            level = ""
        rows.append(
            {
                "model_name": name,
                "model_type": "synthetic_diagnostic" if "simple" in name else "external_or_main_case",
                "complexity_level": level,
                "data_residual_behavior": "proxy_available" if manifest.get("proxy_type") else "not_run",
                "model_metric_behavior": "proxy_available" if manifest.get("proxy_type") else "not_run",
                "image_metric_behavior": manifest.get("proxy_type", "not_run"),
                "time_window_status": "SMOKE_TIME_WINDOW_NOT_RAY_TRACED",
                "boundary_status": "not_evaluated_in_lightweight_proxy",
                "main_failure_mode": failure,
                "admit_fwi_verdict": manifest.get("verdict", manifest.get("status")),
                "recommended_claim": "Use as lightweight model-staircase diagnostic only.",
                "forbidden_claim": "Do not report this proxy as full FWI/RTM benchmark evidence.",
            }
        )
    return rows
