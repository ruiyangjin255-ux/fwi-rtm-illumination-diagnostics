from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from rtm_acoustic.scripts.run_rtm_split_consistency_audit0 import run


def test_split_audit_reports_missing_and_ready_cases(tmp_path: Path):
    rtm_dir = tmp_path / "rtm"
    initial = rtm_dir / "initial"
    initial.mkdir(parents=True)
    (initial / "subset_A").mkdir()
    (initial / "subset_B").mkdir()
    np.save(initial / "subset_A" / "rtm_laplacian_filtered_physical.npy", np.ones((4, 5), dtype=np.float32))
    np.save(initial / "subset_B" / "rtm_laplacian_filtered_physical.npy", np.ones((4, 5), dtype=np.float32) * 2)
    manifest = run(tmp_path / "out", rtm_dir, smoke=True)
    assert manifest["status"] == "READY"
    rows = list(csv.DictReader((tmp_path / "out" / "split_metrics.csv").open(encoding="utf-8")))
    by_method = {row["method"]: row for row in rows}
    assert by_method["initial"]["status"] == "READY"
    assert by_method["full_fwi"]["status"] == "MISSING_RTM_CASE"
    assert set(by_method["initial"]["subset_A_shots"].split(";")).isdisjoint(by_method["initial"]["subset_B_shots"].split(";"))
