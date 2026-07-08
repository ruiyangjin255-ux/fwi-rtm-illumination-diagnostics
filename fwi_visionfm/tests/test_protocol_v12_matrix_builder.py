from scripts.build_protocol_v12_matrix import V12_METHOD_SPECS, build_v12_matrix_rows
import subprocess
import sys


def test_v12_matrix_has_five_methods_three_transfers_three_seeds() -> None:
    rows = build_v12_matrix_rows(
        available_families={"flatvel_a": True, "curvevel_a": True, "flatfault_a": True},
        seeds=[0, 1, 2],
    )
    assert len(rows) == 45
    assert {row["method_key"] for row in rows} == {spec["method_key"] for spec in V12_METHOD_SPECS}
    assert {row["train_size"] for row in rows} == {200}
    assert {row["decoder"] for row in rows} == {"common_bounded_velocity_decoder"}
    assert all(not row["boundary_auxiliary"] and not row["geometry_embedding"] and not row["fusion"] for row in rows)
    assert len({tuple(sorted(row)) for row in rows}) == 1


def test_v12_matrix_script_supports_direct_execution() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/build_protocol_v12_matrix.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
