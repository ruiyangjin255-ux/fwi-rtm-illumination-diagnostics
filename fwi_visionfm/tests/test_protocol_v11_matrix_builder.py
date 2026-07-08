from scripts.build_protocol_v11_matrix import build_run_matrix_rows


def test_complete_matrix_has_six_methods_three_transfers_three_seeds() -> None:
    rows = build_run_matrix_rows(
        available_families={"flatvel_a": True, "curvevel_a": True, "flatfault_a": True},
        seeds=[0, 1, 2],
    )
    assert len(rows) == 54
    assert len({row["method_id"] for row in rows}) == 6
    assert len({(row["source_family"], row["target_family"]) for row in rows}) == 3
    assert {row["seed"] for row in rows} == {0, 1, 2}
    for row in rows:
        assert row["train_size"] == 100
        assert row["val_size"] == 50
        assert row["in_family_test_size"] == 50
        assert row["cross_family_test_size"] == 50
        assert row["epochs"] == 2
        assert row["shot_count"] == 5
        assert row["decoder"] == "common_bounded_velocity_decoder"
        assert row["loss"] == "default_l1"


def test_missing_fault_marks_fault_transfers_without_fabricating_results() -> None:
    rows = build_run_matrix_rows(
        available_families={"flatvel_a": True, "curvevel_a": True, "flatfault_a": False},
        seeds=[0],
    )
    assert len(rows) == 18
    fault_rows = [row for row in rows if row["target_family"] == "flatfault_a"]
    assert fault_rows
    assert all(row["status"] == "SKIPPED_DATA_UNAVAILABLE" for row in fault_rows)

