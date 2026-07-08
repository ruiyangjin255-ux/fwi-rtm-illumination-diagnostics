from scripts.analyze_protocol_v13_generalization_gaps import compute_generalization_gaps


def test_generalization_gap_uses_inverse_direction_for_ssim() -> None:
    in_family = {"mae": 10, "rmse": 20, "ssim": .9, "gradient_error": 3, "edge_mae": 4}
    cross = {"mae": 15, "rmse": 28, "ssim": .7, "gradient_error": 5, "edge_mae": 7}
    gaps = compute_generalization_gaps(in_family, cross)
    assert gaps == {"mae_generalization_gap": 5, "rmse_generalization_gap": 8, "ssim_generalization_gap": .2, "gradient_generalization_gap": 2, "edge_generalization_gap": 3}

