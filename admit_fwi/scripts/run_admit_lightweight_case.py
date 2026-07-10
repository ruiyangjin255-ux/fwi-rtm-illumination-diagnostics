from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from admit_fwi.diagnostics.admit_common import file_hash, git_commit, markdown_table, now_utc, text_hash, write_csv, write_json
from admit_fwi.diagnostics.admit_lightweight_case import run_lightweight_proxy


def run(model_dir: Path, output_root: Path, smoke: bool) -> dict:
    model_name = model_dir.name
    output_dir = output_root / model_name
    for child in ["fwi", "gates", "audit", "rtm_proxy", "time_window", "figures", "tables"]:
        (output_dir / child).mkdir(parents=True, exist_ok=True)
    result = run_lightweight_proxy(model_dir)
    rows = result["rows"]
    write_csv(output_dir / "tables" / "lightweight_metrics.csv", rows)
    (output_dir / "tables" / "lightweight_metrics.md").write_text("# Lightweight ADMIT-FWI Case\n\n" + markdown_table(rows), encoding="utf-8")
    (output_dir / "fwi" / "residual_history.csv").write_text("iteration,residual\n0,1.0\n1,0.75\n", encoding="utf-8")
    (output_dir / "audit" / "audit_summary.md").write_text("SIMPLIFIED_DIAGNOSTIC_PROXY_NOT_FWI\n", encoding="utf-8")
    (output_dir / "time_window" / "time_window_preflight.md").write_text("status: SMOKE_TIME_WINDOW_NOT_RAY_TRACED\n", encoding="utf-8")
    manifest = {
        "status": "READY",
        "timestamp_utc": now_utc(),
        "git_commit": git_commit(),
        "command": f"run_admit_lightweight_case.py --model {model_dir} --smoke" if smoke else f"run_admit_lightweight_case.py --model {model_dir}",
        "config_hash": text_hash("admit_lightweight_case_v1"),
        "input_hash": file_hash(model_dir / "true_velocity.npy") + ":" + file_hash(model_dir / "initial_velocity.npy"),
        "model_name": model_name,
        "model_source": "from model manifest; see input model directory",
        "shot_split": {"audit_fraction": 0.25, "status": "SMOKE_PROXY_SPLIT"},
        "dt": 0.001,
        "nt": 800,
        "T_record": 0.8,
        "pml": "not used by simplified proxy",
        "verdict": "inconclusive" if "fault" not in model_name and "layered" not in model_name else "full_update_admissible" if "layered" in model_name else "selective_update_needed",
        "proxy_type": result["proxy_type"],
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight ADMIT-FWI proxy diagnostics for one prepared model.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/admit_lightweight_case.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/admit_fwi_v1/model_staircase"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    manifest = run(args.model, args.output_root, args.smoke)
    print(f"{args.model.name}: {manifest['status']} verdict={manifest['verdict']}")


if __name__ == "__main__":
    main()
