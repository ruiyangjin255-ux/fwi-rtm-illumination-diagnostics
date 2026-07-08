from __future__ import annotations

import numpy as np

from rtm_acoustic.diagnostics.deep_time_window import compute_required_record_time


def test_deep_time_planner_keeps_dt_and_excludes_pml_depth() -> None:
    model = np.full((100, 50), 2000.0, dtype=np.float32)
    plan = compute_required_record_time(
        model,
        dx=10.0,
        dz=10.0,
        source_positions=[5, 25, 45],
        receiver_positions=list(range(50)),
        dt=0.001,
        wavelet_peak_time=1.0 / 15.0,
        f0=15.0,
        pml_thickness=20,
        current_nt=900,
        nt_floor=5000,
        stride=100,
    )
    assert plan.nt_recommended >= 5000
    assert plan.current_time == 0.9
    assert plan.target_depth_m < 100 * 10.0
    assert plan.target_depth_m <= 0.95 * (100 - 20 - 1) * 10.0 + 1.0e-6
