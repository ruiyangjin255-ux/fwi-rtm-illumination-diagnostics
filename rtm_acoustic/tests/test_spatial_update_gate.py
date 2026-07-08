from __future__ import annotations

from pathlib import Path

import pytest

from rtm_acoustic.build_spatial_update_gate import build, load_gate_inputs, scan_spatial_update_gates


RAW_INPUT = Path("rtm_acoustic/outputs/FWI/full_salt_fwi_cg_allshots_v2/full_salt_true_model.npy")


pytestmark = pytest.mark.skipif(not RAW_INPUT.exists(), reason="raw FWI/RTM arrays are not included in the lightweight release")


def test_spatial_update_gate_selects_edge_safe_update() -> None:
    inputs = load_gate_inputs()
    rows, selected, selected_alpha, selected_model = scan_spatial_update_gates(inputs)
    global_row = next(row for row in rows if row["candidate"] == "global_alpha0.1_thr0")

    assert selected["accepted"] is True
    assert selected["candidate"] != global_row["candidate"]
    assert selected["mae_improvement_pct"] > global_row["mae_improvement_pct"]
    assert selected["rmse_improvement_pct"] > global_row["rmse_improvement_pct"]
    assert global_row["edge_mae_improvement_pct"] < 0.0
    assert selected["edge_mae_improvement_pct"] >= 0.0
    assert 0.0 < selected_alpha.mean() < 0.15
    assert selected_model.shape == inputs.initial_velocity.shape


def test_spatial_update_gate_writes_outputs() -> None:
    outputs = build()

    for key in ["csv", "json", "markdown", "alpha", "model", "figure_png", "figure_pdf", "figure_svg", "figure_tiff"]:
        path = Path(outputs[key])
        assert path.exists(), path
        assert path.stat().st_size > 0, path
