from fwi_visionfm.pasd.run_source_val_sweep import _candidate_lambda


def test_candidate_lambda_uses_current_without_target_context():
    assert _candidate_lambda("current", 0.1) == 0.1
    assert _candidate_lambda("lambda_edge_0.03", 0.1) == 0.03
