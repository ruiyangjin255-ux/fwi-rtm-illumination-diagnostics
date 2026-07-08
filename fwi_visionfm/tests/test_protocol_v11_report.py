import csv
from pathlib import Path

from scripts.report_protocol_v11_visionfm_crossfamily import write_protocol_v11_report


def test_report_and_claims_keep_pre_registered_claim_boundary(tmp_path: Path) -> None:
    aggregate = tmp_path / "protocol_v11_aggregate_metrics.csv"
    with aggregate.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method_id", "method_name", "available_transfer_count", "evidence_level"])
        writer.writeheader()
        writer.writerow({"method_id": "M3", "method_name": "DINOv2 frozen", "available_transfer_count": 3, "evidence_level": "当前未形成一致证据"})
    report, claims = write_protocol_v11_report(root=tmp_path)
    report_text = report.read_text(encoding="utf-8")
    claims_text = claims.read_text(encoding="utf-8")
    assert "视觉模型是否适用于端到端 FWI" in report_text
    assert "不构成标准基准级结论" in report_text
    assert "已经证明提升 FWI 泛化" not in report_text
    assert "## Can Claim" in claims_text
    assert "## Cannot Claim" in claims_text


def test_report_includes_per_transfer_metrics_and_baseline_deltas(tmp_path: Path) -> None:
    fields = [
        "transfer_id", "method_key", "method_name", "evidence_level",
        "cross_family_mae_mean", "cross_family_rmse_mean", "cross_family_ssim_mean",
        "cross_family_gradient_error_mean", "cross_family_edge_mae_mean",
        "mae_generalization_gap_mean", "rmse_generalization_gap_mean",
        "edge_mae_generalization_gap_mean",
    ]
    rows = [
        ["flatvel_a_to_curvevel_a", "cnn_baseline", "CNN baseline", "基线参考", 100, 120, 0.8, 30, 50, 10, 12, 5],
        ["flatvel_a_to_curvevel_a", "random_vit", "random ViT", "基线参考", 110, 130, 0.7, 35, 55, 20, 22, 8],
        ["flatvel_a_to_curvevel_a", "dinov2_frozen", "DINOv2 frozen", "存在部分或混合证据", 90, 115, 0.81, 29, 49, 5, 7, 3],
    ]
    with (tmp_path / "protocol_v11_aggregate_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(fields); writer.writerows(rows)

    report, _ = write_protocol_v11_report(root=tmp_path)
    text = report.read_text(encoding="utf-8")
    assert "flatvel_a_to_curvevel_a" in text
    assert "ΔMAE vs CNN" in text
    assert "gradient_error" in text
    assert "DINOv2 frozen 的直接迁移" in text
