from __future__ import annotations

import csv
from pathlib import Path

from rtm_acoustic.build_jge_method_synthesis import build


def test_build_method_synthesis_writes_literature_guided_matrix(tmp_path: Path) -> None:
    paths = build(tmp_path)

    assert paths["csv"].exists()
    assert paths["markdown"].exists()

    rows = list(csv.DictReader(paths["csv"].open(encoding="utf-8")))
    assert len(rows) >= 5
    assert any("Optimization" in row["literature_direction"] for row in rows)
    assert "should not claim a new high-performance FWI algorithm" in paths["markdown"].read_text(encoding="utf-8")
