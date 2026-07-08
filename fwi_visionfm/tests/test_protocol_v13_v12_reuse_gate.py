from scripts.verify_v12_reuse_for_v13 import compare_reuse_contract


def test_reuse_requires_exact_scientific_contract() -> None:
    base = {"manifest_hash": "abc", "sample_ids_hash": "ids", "source_family": "flatvel_a", "target_family": "curvevel_a", "seed": 0, "shot_count": 5, "bridge": "raw_envelope_spectrum3", "image_size": 224, "decoder": "common_bounded_velocity_decoder", "decoder_config_hash": "decoder", "optimizer_registered": True, "loss": "default_l1", "epochs": 2, "metric_space": "physical_velocity", "target_isolated": True}
    passed, reasons = compare_reuse_contract(base, dict(base))
    assert passed and reasons == []
    changed = dict(base); changed["manifest_hash"] = "different"
    passed, reasons = compare_reuse_contract(base, changed)
    assert not passed and "manifest_hash" in reasons

