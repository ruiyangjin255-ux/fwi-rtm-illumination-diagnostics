import csv
import json
from pathlib import Path

from scripts.report_protocol_v12_spectrogram_dinov2_confirmation import write_protocol_v12_report


def test_v12_report_and_claims_keep_confirmation_boundary(tmp_path: Path) -> None:
    with (tmp_path / "protocol_v12_aggregate_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["transfer_id", "method_key", "method_name", "cross_family_mae_mean", "cross_family_rmse_mean", "cross_family_ssim_mean", "cross_family_gradient_error_mean", "cross_family_edge_mae_mean", "mae_generalization_gap_mean", "rmse_generalization_gap_mean", "ssim_generalization_gap_mean", "gradient_generalization_gap_mean", "edge_generalization_gap_mean"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        writer.writerow({"transfer_id": "flatvel_a_to_curvevel_a", "method_key": "cnn_baseline", "method_name": "CNN baseline", **{field: 1 for field in fields[3:]}})
    (tmp_path / "protocol_v12_summary.json").write_text(json.dumps({"comparison_evidence": {"M5_vs_M4": {"evidence_level": "存在部分或混合证据", "qualifying_transfer_count": 0, "details": [{"transfer_id": "flatvel_a_to_curvevel_a", "numerical_seed_count": 2, "structural_seed_count": 1, "bootstrap_ci_below_zero_seed_count": 2, "qualifying": False}]}}}), encoding="utf-8")
    report, claims, integrity = write_protocol_v12_report(tmp_path)
    report_text = report.read_text(encoding="utf-8")
    claims_text = claims.read_text(encoding="utf-8")
    assert "频谱 DINOv2 跨构造确认评测" in report_text
    assert "不构成标准基准级结论" in report_text
    assert "DINOv2 已证明提升 FWI 泛化能力" not in report_text
    assert "数值改善 seed" in report_text
    assert "bootstrap 支持 seed" in report_text
    assert "## Can Claim" in claims_text and "## Cannot Claim" in claims_text
    assert integrity.is_file()
