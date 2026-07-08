from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize(
    "mask_type",
    ["random_patch", "time_block", "receiver_block", "trace_dropout", "hybrid_seismic_mask"],
)
def test_mae_masking_strategies_return_valid_shapes(mask_type: str):
    torch = pytest.importorskip("torch")

    from fwi_visionfm.models.seismic_backbones.local_mae import LocalSeismicMAE

    model = LocalSeismicMAE(mask_ratio=0.75, mask_type=mask_type)
    batch = torch.randn(2, 3, 64, 64)
    out = model(batch)

    assert tuple(out["reconstruction"].shape) == (2, 3, 64, 64)
    assert tuple(out["masked_input"].shape) == (2, 3, 64, 64)
    assert out["mask"].shape[0] == 2


def test_incompatible_frequency_band_mask_is_skipped_for_raw_envelope(tmp_path: Path):
    from fwi_visionfm.training.mae_pretrain import run_local_mae_pretrain

    sample_root = Path("D:/ryjin/fwi_visionfm/data/flatvel_a_subset2k")
    paths = sorted(sample_root.glob("sample_*.npz"))[:4]
    result = run_local_mae_pretrain(
        train_paths=paths[:3],
        val_paths=paths[3:],
        output_dir=tmp_path / "pretrain",
        bridge="raw_envelope_spectrum3",
        seed=0,
        epochs=1,
        batch_size=2,
        device="cpu",
        mask_type="frequency_band",
    )
    assert result["status"] == "SKIPPED_INCOMPATIBLE_MASK"

