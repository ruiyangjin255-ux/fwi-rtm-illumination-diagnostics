from __future__ import annotations

import csv
from pathlib import Path


def test_protocol_v14_robustness_report_is_evaluation_only(tmp_path: Path):
    from scripts.evaluate_protocol_v14_acquisition_robustness import write_protocol_v14_robustness_outputs

    root = tmp_path / "robustness"
    rows = [
        {"method_id": "M3", "bridge_id": "B0", "perturbation": "clean", "metric_name": "mae", "metric_value": 1.0, "degradation": 0.0},
        {"method_id": "M3", "bridge_id": "B3", "perturbation": "few_shot_3", "metric_name": "mae", "metric_value": 1.2, "degradation": 0.2},
    ]
    write_protocol_v14_robustness_outputs(root=root, rows=rows)
    report = (root / "robustness_report.md").read_text(encoding="utf-8")
    assert "evaluation-only" in report
    csv_rows = list(csv.DictReader((root / "protocol_v14_robustness_metrics.csv").open("r", encoding="utf-8")))
    assert len(csv_rows) == 2
