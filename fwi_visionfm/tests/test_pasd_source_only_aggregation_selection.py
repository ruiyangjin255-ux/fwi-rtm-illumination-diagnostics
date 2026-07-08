from fwi_visionfm.pasd.phase3_utils import candidate_to_variant, select_source_candidate


def test_phase3_candidate_mapping_is_locked():
    assert candidate_to_variant("C1_pasd_core_mean") == "B4_no_geometry_attention"
    assert candidate_to_variant("C2_pasd_core_attention") == "B4_pasd_fwi"


def test_phase3_source_selection_prefers_eligible_rank_score():
    rows = [
        {"candidate": "C1_pasd_core_mean", "MAE": 1.0, "SSIM": 0.9, "edge_F1": 0.5, "gradient_l1_edge": 2.0},
        {"candidate": "C2_pasd_core_attention", "MAE": 1.005, "SSIM": 0.91, "edge_F1": 0.6, "gradient_l1_edge": 1.8},
    ]
    decision = select_source_candidate(rows)
    assert decision["selected_candidate"] == "C2_pasd_core_attention"
    assert decision["selected_variant"] == "B4_pasd_fwi"
