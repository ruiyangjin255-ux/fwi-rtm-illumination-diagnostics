from __future__ import annotations


def test_geometry_embedding_shape_and_cpu_forward():
    from fwi_visionfm.models.geometry_embedding import GeometryEmbedding
    from fwi_visionfm.torch_backend import require_torch_backend

    torch = require_torch_backend()
    module = GeometryEmbedding(embed_dim=8, mode="sinusoidal")
    source_index = torch.arange(3).view(1, 3).repeat(2, 1)
    receiver_index = torch.arange(5).view(1, 1, 5).repeat(2, 3, 1)
    time_index = torch.arange(7).view(1, 7).repeat(2, 1)
    out = module(source_index=source_index, receiver_index=receiver_index, time_index=time_index)
    assert tuple(out.shape) == (2, 24, 7, 5)


def test_geometry_embedding_missing_geometry_uses_fallback():
    from fwi_visionfm.models.geometry_embedding import GeometryEmbedding
    from fwi_visionfm.torch_backend import require_torch_backend

    torch = require_torch_backend()
    module = GeometryEmbedding(embed_dim=4, mode="sinusoidal")
    out = module(time_index=torch.arange(6).view(1, 6), target_hw=(6, 4), batch_size=2, shots=3)
    assert tuple(out.shape) == (2, 16, 6, 4)


def test_geometry_embedding_learnable_mode_runs():
    from fwi_visionfm.models.geometry_embedding import GeometryEmbedding
    from fwi_visionfm.torch_backend import require_torch_backend

    torch = require_torch_backend()
    module = GeometryEmbedding(embed_dim=6, mode="learnable")
    out = module(
        source_index=torch.arange(2).view(1, 2),
        receiver_index=torch.arange(4).view(1, 1, 4).repeat(1, 2, 1),
        time_index=torch.arange(5).view(1, 5),
        target_hw=(5, 4),
    )
    assert tuple(out.shape) == (1, 18, 5, 4)
