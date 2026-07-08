from __future__ import annotations

import csv
from pathlib import Path


FIELDS = [
    "fusion_name",
    "method",
    "source_a",
    "source_b",
    "best_param",
    "optimize_requested",
    "optimize_actual",
    "MAE",
    "RMSE",
    "SSIM",
    "gradient_error",
    "edge_MAE",
    "visual_score",
    "visual_rank",
    "reference_only",
]


def test_fusion_report_contains_limitations(tmp_path: Path):
    from fwi_visionfm.scripts.report_protocol_v4_fusion import write_fusion_report

    with (tmp_path / "protocol_v4_fusion_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "fusion_name": "average",
                "method": "average_fusion",
                "source_a": "cnn",
                "source_b": "vit",
                "best_param": 0.5,
                "optimize_requested": "val",
                "optimize_actual": "in_family_test",
                "MAE": 1.0,
                "RMSE": 2.0,
                "SSIM": 0.5,
                "gradient_error": 3.0,
                "edge_MAE": 4.0,
                "visual_score": 0.7,
                "visual_rank": 1,
                "reference_only": False,
            }
        )

    report = write_fusion_report(tmp_path)
    text = report.read_text(encoding="utf-8")

    assert "not application-level" in text
    assert "not benchmark evidence" in text
    assert "numerical-structural trade-off" in text
    assert "fusion 已经解决复杂 FWI 成图" not in text
