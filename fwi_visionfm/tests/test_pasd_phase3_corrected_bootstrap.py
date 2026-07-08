import argparse
import csv
from pathlib import Path

from fwi_visionfm.pasd.bootstrap_corrected_metrics import bootstrap_corrected


def test_phase3r_corrected_bootstrap_uses_aligned_samples(tmp_path: Path):
    root = tmp_path / "corrected"
    target_dir = root / "curvevel_a"
    target_dir.mkdir(parents=True)
    rows = []
    for variant, offset in (("B1_raw_unet", 1.0), ("PASD_Core_locked", 0.5)):
        for sample_id in (1, 2, 3):
            rows.append({"variant": variant, "seed": 0, "sample_id": sample_id, "MAE": sample_id + offset, "RMSE": sample_id + offset, "SSIM": 1.0 - offset / 10, "source_threshold_edge_MAE": sample_id + offset, "gradient_l1_edge": sample_id + offset, "edge_F1": 0.5 + offset / 10})
    with (target_dir / "corrected_per_sample_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args = argparse.Namespace(
        corrected_metrics_root=root,
        variants=["B1_raw_unet", "PASD_Core_locked"],
        targets=["cross_curvevel_a"],
        seeds=[0],
        metrics=["MAE"],
        bootstrap_resamples=100,
        output=tmp_path / "bootstrap",
    )
    bootstrap_corrected(args)
    assert (tmp_path / "bootstrap" / "bootstrap_summary_curvevel_a.csv").exists()
