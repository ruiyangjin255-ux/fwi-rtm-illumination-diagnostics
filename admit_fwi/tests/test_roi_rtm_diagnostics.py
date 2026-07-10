from __future__ import annotations

from pathlib import Path

import numpy as np

from admit_fwi.diagnostics.roi_rtm_diagnostics import compute_roi_rows


def test_roi_diagnostics_marks_truth_aware_regions(tmp_path: Path):
    fwi = tmp_path / "fwi"
    gate = tmp_path / "gate"
    rtm = tmp_path / "rtm"
    (gate / "models").mkdir(parents=True)
    (gate / "diagnostics").mkdir()
    (gate / "gates").mkdir()
    (gate / "audit").mkdir()
    fwi.mkdir()
    true = np.full((20, 30), 2000.0, dtype=np.float32)
    true[6:14, 10:20] = 4500.0
    initial = true * 0.95
    np.save(fwi / "full_salt_true_model.npy", true)
    np.save(fwi / "full_salt_initial_model.npy", initial)
    np.save(gate / "diagnostics" / "illumination_score.npy", np.linspace(0, 1, true.size, dtype=np.float32).reshape(true.shape))
    np.save(gate / "diagnostics" / "gradient_consensus.npy", np.ones_like(true))
    np.save(gate / "diagnostics" / "descent_alignment.npy", np.ones_like(true))
    np.save(gate / "diagnostics" / "ecg_reliability_score.npy", np.ones_like(true))
    np.save(gate / "models" / "initial_model.npy", initial)
    np.save(gate / "models" / "full_fwi_model.npy", initial + 0.1 * (true - initial))
    (gate / "audit" / "audit_method_summary.csv").write_text("method,nrms,trace_corr\ninitial,1,0\nfull_fwi,0.5,0.5\n", encoding="utf-8")
    for method in ["initial", "full_fwi"]:
        (rtm / method).mkdir(parents=True)
        np.save(rtm / method / "rtm_laplacian_filtered_physical.npy", np.ones_like(true))
        np.save(rtm / method / "rtm_source_normalized_physical.npy", np.ones_like(true) * 2)
    rows = compute_roi_rows(fwi_dir=fwi, gate_root=gate, rtm_dir=rtm, methods=["initial", "full_fwi"])
    truth_rows = [row for row in rows if row["region"] == "salt_top"]
    assert truth_rows
    assert truth_rows[0]["claim_scope"] == "TRUTH_AWARE_BENCHMARK_ONLY"
    proxy_rows = [row for row in rows if row["region"] == "low_illumination"]
    assert proxy_rows[0]["region_type"] == "truth-free"
