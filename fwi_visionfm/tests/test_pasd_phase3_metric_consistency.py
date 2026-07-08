import numpy as np

from fwi_visionfm.pasd.phase3_utils import corrected_prediction_metrics
from fwi_visionfm.pasd.phase3_postprocess import build_canonical_summary


def test_phase3_corrected_metrics_are_zero_for_identical_fields():
    target = np.zeros((8, 8), dtype=np.float32)
    target[:, 4:] = 1.0
    metrics = corrected_prediction_metrics(target, target, true_edge_threshold=0.1, pred_edge_threshold=0.1)
    assert metrics["MAE"] == 0.0
    assert metrics["RMSE"] == 0.0
    assert metrics["source_threshold_edge_MAE"] == 0.0
    assert metrics["gradient_l1_edge"] == 0.0
    assert metrics["edge_F1"] == 1.0


def test_phase3_summary_coalesces_corrected_uppercase_metrics(tmp_path):
    formal = tmp_path
    rows = [
        {
            "variant": "B1_raw_unet",
            "dataset": "cross_flatfault_a",
            "seed": "0",
            "MAE": "10",
            "RMSE": "20",
            "SSIM": "0.5",
            "edge_MAE": "11",
            "gradient_l1_edge": "3",
            "edge_F1": "0.2",
        },
        {
            "variant": "B1_raw_unet",
            "dataset": "cross_flatfault_a",
            "seed": "1",
            "MAE": "14",
            "RMSE": "24",
            "SSIM": "0.7",
            "edge_MAE": "15",
            "gradient_l1_edge": "5",
            "edge_F1": "0.4",
        },
    ]
    (formal / "tables").mkdir()
    with (formal / "protocol_runs.csv").open("w", encoding="utf-8", newline="") as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = build_canonical_summary(formal)
    assert summary[0]["MAE"] == 12
    assert summary[0]["RMSE"] == 22
    assert summary[0]["SSIM"] == 0.6
