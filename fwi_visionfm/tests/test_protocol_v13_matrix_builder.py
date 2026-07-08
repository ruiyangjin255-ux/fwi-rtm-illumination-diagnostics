from scripts.build_protocol_v13_matrix import build_v13_matrix_rows


def test_matrix_contains_six_methods_three_transfers_three_seeds() -> None:
    rows = build_v13_matrix_rows()
    assert len(rows) == 54
    assert {row["method_key"] for row in rows} == {"cnn_baseline", "random_vit", "dinov2_frozen", "dinov2_lora", "spectrogram_dinov2_lora", "ncs2d_frozen"}
    assert sum(row["method_key"] == "ncs2d_frozen" for row in rows) == 9
    assert all(row["train_size"] == 200 and row["epochs"] == 2 for row in rows)

