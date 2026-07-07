from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rtm_acoustic.diagnostics.shot_partition import interleaved_audit_split
from rtm_acoustic.scripts._common import ensure_output_tree, read_simple_yaml, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare held-out shot split audit manifest.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--audit-fold", type=int, default=None)
    parser.add_argument("--all-audit-folds", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    config = read_simple_yaml(args.config)
    output_dir = root / config.get("output_dir", "rtm_acoustic/outputs/salt_reliability_gate_v1")
    ensure_output_tree(output_dir)
    shots = list(range(16 if args.smoke else 224))
    folds = range(4) if args.all_audit_folds else [args.audit_fold if args.audit_fold is not None else int(config.get("audit_fold", 0))]
    payload = {
        "status": "SPLIT_PREPARED",
        "folds": [
            {
                "audit_fold": split.audit_fold,
                "inversion_shots": split.inversion_shots,
                "audit_shots": split.audit_shots,
            }
            for split in (interleaved_audit_split(shots, int(fold), 4) for fold in folds)
        ],
        "note": "This smoke prepares split metadata only; no full FWI/RTM is launched.",
    }
    write_json(output_dir / "audit" / "heldout_audit_manifest.json", payload)
    print(output_dir)


if __name__ == "__main__":
    main()
