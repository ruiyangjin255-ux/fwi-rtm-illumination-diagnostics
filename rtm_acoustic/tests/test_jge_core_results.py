from __future__ import annotations

import csv

from rtm_acoustic.build_jge_core_results import build


def _read_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_build_jge_core_results_outputs_ranked_tables(tmp_path) -> None:
    written = build(tmp_path)

    full_fwi = _read_rows(written["full_fwi_csv"])
    fwi_quality = _read_rows(written["fwi_quality_csv"])
    update_scale = _read_rows(written["update_scale_csv"])
    local_fwi = _read_rows(written["local_fwi_csv"])
    rtm = _read_rows(written["rtm_csv"])
    innovations = _read_rows(written["innovation_csv"])

    assert full_fwi[0]["case"] == "CG_allshots_v2"
    assert float(full_fwi[0]["misfit_reduction_pct"]) > float(full_fwi[1]["misfit_reduction_pct"])
    assert fwi_quality[0]["case"] == "CG_allshots_v2"
    assert "edge_mae_improvement_pct" in fwi_quality[0]
    assert any(row["selected"] == "True" and row["alpha"] == "0.1" for row in update_scale)
    assert local_fwi[0]["case"] == "adaptive_line_search_baseline"
    assert local_fwi[1]["case"] == "adaptive_line_search_illumination_preconditioned"
    assert rtm[0]["case"] == "source_receiver_vs_source_normalized"
    assert float(rtm[0]["low_illumination_fraction"]) < 0.02
    assert innovations[0]["paper_position"] == "main contribution"
    assert written["plan_md"].read_text(encoding="utf-8").startswith("# JGE-oriented Upgrade Plan")
