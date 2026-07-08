import argparse
from pathlib import Path

import pytest

from fwi_visionfm.pasd.run_source_aggregation_selection import run_selection


def test_phase3_source_selection_requires_target_access_guard():
    args = argparse.Namespace(
        protocol=Path("protocols/pasd_phase1b_locked_flatvel_a_to_curvevel_a.json"),
        locked_config=Path("configs/pasd_phase1b_locked_config.json"),
        output=Path("outputs/pasd_phase3_paper/source_aggregation_selection_test"),
        candidates=["C1_pasd_core_mean"],
        seeds=[0],
        selection_split="source_val",
        epochs=1,
        batch_size=1,
        base_channels=4,
        latent_channels=8,
        torch_threads=1,
        forbid_target_access=False,
    )
    with pytest.raises(ValueError, match="forbid-target-access"):
        run_selection(args)
