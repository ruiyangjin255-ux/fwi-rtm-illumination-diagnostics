from scripts.report_protocol_v13_natural_vs_seismic_pretraining import write_protocol_v13_report


def test_report_keeps_pretraining_claim_boundary(tmp_path) -> None:
    report, claims, integrity = write_protocol_v13_report(tmp_path)
    text = report.read_text(encoding="utf-8"); claim_text = claims.read_text(encoding="utf-8")
    assert "自然图像与地震域预训练跨构造确认" in text
    assert "不构成标准基准级结论" in text
    assert "NCS2D 已证明提升 FWI 泛化能力" not in text
    assert "## Can Claim" in claim_text and "## Cannot Claim" in claim_text
    assert integrity.is_file()

