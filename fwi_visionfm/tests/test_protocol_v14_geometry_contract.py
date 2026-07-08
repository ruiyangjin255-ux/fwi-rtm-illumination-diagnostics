from __future__ import annotations

import numpy as np


def test_protocol_v14_prediction_contract_includes_geometry_fields(tmp_path):
    from scripts.run_protocol_v14_geometry_aware_trace_bridge import write_protocol_v14_prediction_npz

    path = tmp_path / "predictions.npz"
    prediction = np.ones((2, 1, 70, 70), dtype=np.float32)
    target = np.zeros((2, 1, 70, 70), dtype=np.float32)
    write_protocol_v14_prediction_npz(
        path,
        prediction=prediction,
        target=target,
        sample_ids=["a", "b"],
        metadata={
            "model_id": "M3",
            "bridge_name": "geometry_aware_trace_bridge_geometry",
            "geometry_mode": "canonical",
            "geometry_provenance": "CANONICAL_RECONSTRUCTED",
            "trace_context_radius": 2,
            "use_shot_global_context": False,
            "use_multiscale_context": False,
            "source_family": "flatvel_a",
            "target_family": "curvevel_a",
            "seed": 0,
            "metric_space": "physical_velocity",
            "is_real_feature": True,
        },
    )
    payload = np.load(path, allow_pickle=True)
    assert payload["geometry_provenance"].item() == "CANONICAL_RECONSTRUCTED"
    assert payload["trace_context_radius"].item() == 2
