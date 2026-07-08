from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from rtm_acoustic.diagnostics.rtm_split_runner import run_split_subsets


def main() -> None:
    parser = argparse.ArgumentParser(description="Run true RTM subset A/B outputs for audit0 gate split consistency.")
    parser.add_argument("--config", type=Path, default=Path("configs/admit_seg_salt_main_case.yaml"))
    parser.add_argument("--methods", nargs="+", default=["initial", "illumination"])
    parser.add_argument("--fwi-dir", type=Path, default=Path("outputs/FWI/full_salt_fwi_cg_audit0_train_ecg_v1"))
    parser.add_argument("--model-dir", type=Path, default=Path("outputs/salt_reliability_gate_audit0_v1/models"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/RTM/audit0_gate_rtm_split_v1"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = run_split_subsets(
        root=root,
        fwi_dir=args.fwi_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        methods=args.methods,
        smoke=args.smoke,
        workers=args.workers,
        command=" ".join(sys.argv),
    )
    print(json.dumps({"status": manifest["status"], "methods": manifest["methods"], "subsets": manifest["subsets"]}, indent=2))


if __name__ == "__main__":
    main()
