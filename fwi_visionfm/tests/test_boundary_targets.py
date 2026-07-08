from __future__ import annotations

import pytest


@pytest.mark.parametrize("method", ["gradient_magnitude", "sobel", "laplacian", "thresholded_gradient"])
def test_boundary_targets_build_and_normalize(method: str):
    from fwi_visionfm.data.boundary_targets import build_velocity_boundary_target
    from fwi_visionfm.torch_backend import require_torch_backend

    torch = require_torch_backend()
    velocity = torch.linspace(1500.0, 4500.0, 16).view(1, 1, 4, 4)
    boundary = build_velocity_boundary_target(velocity, method=method, threshold=0.2)
    assert tuple(boundary.shape) == (1, 1, 4, 4)
    assert float(boundary.min()) >= 0.0
    assert float(boundary.max()) <= 1.0

