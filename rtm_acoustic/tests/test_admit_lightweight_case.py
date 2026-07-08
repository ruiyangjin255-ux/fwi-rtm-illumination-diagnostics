from __future__ import annotations

from pathlib import Path

from rtm_acoustic.models.synthetic_model_bank import save_synthetic_model
from rtm_acoustic.scripts.run_admit_lightweight_case import run


def test_lightweight_case_writes_manifest_and_metrics(tmp_path: Path):
    save_synthetic_model("simple_layered", tmp_path / "models", nx=48, nz=24)
    manifest = run(tmp_path / "models" / "simple_layered", tmp_path / "out", smoke=True)
    assert manifest["status"] == "READY"
    assert manifest["proxy_type"] == "SIMPLIFIED_DIAGNOSTIC_PROXY_NOT_FWI"
    assert (tmp_path / "out" / "simple_layered" / "tables" / "lightweight_metrics.csv").exists()
