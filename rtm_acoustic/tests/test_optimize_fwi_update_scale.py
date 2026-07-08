from __future__ import annotations

from pathlib import Path

import numpy as np

from rtm_acoustic.optimize_fwi_update_scale import optimize_run_dir, optimize_update_scale


def test_optimize_update_scale_prefers_conservative_edge_safe_update() -> None:
    true_model = np.full((8, 10), 2000.0, dtype=np.float32)
    true_model[:, 5:] = 3200.0
    initial_model = np.full_like(true_model, 2000.0)
    inverted_model = initial_model.copy()
    inverted_model[:, 5:] = 4400.0
    inverted_model[:, 4] = 1200.0

    result = optimize_update_scale(
        true_model=true_model,
        initial_model=initial_model,
        inverted_model=inverted_model,
        alphas=[0.0, 0.25, 0.5, 1.0],
        edge_tolerance=0.0,
        gradient_tolerance=1.0,
    )

    assert result["selected_alpha"] in {0.25, 0.5}
    assert result["selected_alpha"] < 1.0
    assert result["selected_model"].shape == true_model.shape
    assert any(row["accepted"] for row in result["candidates"])


def test_optimize_run_dir_writes_selected_model_and_tables(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    true_model = np.full((6, 8), 2000.0, dtype=np.float32)
    true_model[:, 4:] = 2600.0
    initial_model = np.full_like(true_model, 2000.0)
    inverted_model = initial_model.copy()
    inverted_model[:, 4:] = 2400.0
    np.save(run_dir / "full_salt_true_model.npy", true_model)
    np.save(run_dir / "full_salt_initial_model.npy", initial_model)
    np.save(run_dir / "full_salt_inverted_model.npy", inverted_model)

    written = optimize_run_dir(
        run_dir,
        output_dir=tmp_path / "optimized",
        alphas=[0.0, 0.25, 0.5, 1.0],
        gradient_tolerance=1.0,
    )

    assert written["json"].exists()
    assert written["csv"].exists()
    assert written["markdown"].exists()
    selected = np.load(written["model"])
    assert selected.shape == true_model.shape
    assert np.max(selected - initial_model) > 0.0
