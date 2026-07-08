import argparse
from pathlib import Path

import pytest

from fwi_visionfm.pasd.recompute_phase3_metrics import recompute_metrics


def test_phase3r_requires_source_val_prediction_threshold():
    args = argparse.Namespace(
        phase3_root=Path("outputs/pasd_phase3_paper"),
        locked_config=Path("configs/pasd_phase3_pasd_core_locked.json"),
        dual_target_protocol=Path("protocols/pasd_phase3_dual_target_locked.json"),
        output=Path("outputs/pasd_phase3r_metric_repair_test/corrected_metrics"),
        use_fresh_prediction_archives_only=True,
        edge_mask="source_threshold_strict_gt",
        prediction_edge_threshold="target_adaptive",
        dx="auto",
        dz="auto",
    )
    with pytest.raises(ValueError, match="source_val_locked"):
        recompute_metrics(args)
