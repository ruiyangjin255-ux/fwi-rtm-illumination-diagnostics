from __future__ import annotations

from pathlib import Path

import numpy as np

from rtm_acoustic.diagnostics.admit_evidence_matrix import MODEL_FILES, build_evidence_matrix


def test_evidence_matrix_contains_required_methods_and_deep_limit(tmp_path: Path):
    root = tmp_path
    fwi = root / "outputs" / "FWI" / "full_salt_fwi_cg_audit0_train_ecg_v1"
    gate = root / "outputs" / "salt_reliability_gate_audit0_v1"
    rtm = root / "outputs" / "RTM" / "audit0_gate_rtm_v1"
    deep = root / "outputs" / "deep_time_preflight_v1" / "wavefield_smoke"
    split = root / "split"
    roi = root / "roi"
    for path in [fwi, gate / "models", gate / "gates", gate / "audit", rtm, deep, split, roi]:
        path.mkdir(parents=True)
    true = np.ones((5, 6), dtype=np.float32) * 2000
    initial = true * 0.9
    np.save(fwi / "full_salt_true_model.npy", true)
    np.save(fwi / "full_salt_initial_model.npy", initial)
    for method, filename in MODEL_FILES.items():
        np.save(gate / "models" / filename, initial + 1)
    (gate / "audit" / "audit_method_summary.csv").write_text("method,normalized_l2,nrms,trace_corr\ninitial,1,1,0\nfull_fwi,0.5,0.5,0.9\n", encoding="utf-8")
    (rtm / "gate_rtm_method_summary.csv").write_text("method,filtered RMSE,filtered corr\ninitial,1,0\nfull_fwi,0.5,0.9\n", encoding="utf-8")
    (split / "split_metrics.csv").write_text("method,status,rtm_split_laplacian_correlation\ninitial,READY,0.1\nfull_fwi,READY,0.9\n", encoding="utf-8")
    (roi / "roi_metrics.csv").write_text("method,region,rtm_laplacian_energy\ninitial,salt_top,1\ninitial,salt_flanks,1\ninitial,subsalt_shadow,1\nfull_fwi,salt_top,2\nfull_fwi,salt_flanks,2\nfull_fwi,subsalt_shadow,2\n", encoding="utf-8")
    (deep / "deep_energy_summary.json").write_text('{"status":"TIME_TRUNCATION_CONFIRMED"}', encoding="utf-8")
    rows = build_evidence_matrix(root, split, roi)
    methods = {row["method"] for row in rows}
    assert {"initial", "full_fwi", "global", "illumination", "ecg", "inverse", "random_seed_4"} <= methods
    assert all(row["deep_time_status"] == "NOT_RELEASED_FOR_DEEP_INTERPRETATION" for row in rows)
    assert all(row["overall_admissibility_verdict"] != "ACCEPTABLE_FOR_DEEP_SUBSALT" for row in rows)
    assert "ECG significantly improves subsalt imaging" in rows[0]["forbidden_claim"]
