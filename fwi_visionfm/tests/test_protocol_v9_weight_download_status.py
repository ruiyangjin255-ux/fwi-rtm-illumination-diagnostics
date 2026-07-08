from __future__ import annotations

import json
from pathlib import Path


def test_protocol_v9_fake_download_status_generates_report(tmp_path: Path):
    from fwi_visionfm.scripts.report_protocol_v9_weight_setup_and_real_probe import write_protocol_v9_weight_setup_and_real_probe_report

    download_root = tmp_path / "download"
    probe_root = tmp_path / "probe"
    download_root.mkdir(parents=True, exist_ok=True)
    probe_root.mkdir(parents=True, exist_ok=True)
    (download_root / "download_status.json").write_text(
        json.dumps(
            {
                "entries": [
                    {"name": "ncs_repo", "status": "ALREADY_EXISTS"},
                    {"name": "ncs_2d", "status": "DOWNLOADED"},
                    {"name": "ncs_2p5d", "status": "DOWNLOAD_FAILED"},
                    {"name": "vit_mae_base", "status": "ALREADY_EXISTS"},
                    {"name": "vit_mae_large", "status": "SKIPPED"},
                ]
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (download_root / "availability_report.json").write_text(
        json.dumps(
            {
                "models": [
                    {"name": "ncs_2d", "status": "AVAILABLE"},
                    {"name": "ncs_2p5d", "status": "WEIGHTS_PRESENT_ADAPTER_PENDING"},
                ]
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    payload = write_protocol_v9_weight_setup_and_real_probe_report(download_root=download_root, probe_root=probe_root, output_dir=probe_root)
    text = payload["report_path"].read_text(encoding="utf-8")
    assert "not benchmark-level proof" in text
    assert "NCS improves FWI" not in text

