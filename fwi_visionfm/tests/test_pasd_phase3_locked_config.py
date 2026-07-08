from fwi_visionfm.pasd.phase3_utils import AGGREGATION_CANDIDATES


def test_phase3_locked_config_candidate_space_is_source_only_c1_c2():
    assert set(AGGREGATION_CANDIDATES) == {"C1_pasd_core_mean", "C2_pasd_core_attention"}
    assert AGGREGATION_CANDIDATES["C1_pasd_core_mean"] != AGGREGATION_CANDIDATES["C2_pasd_core_attention"]
