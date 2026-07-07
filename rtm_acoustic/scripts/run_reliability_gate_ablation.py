from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rtm_acoustic.diagnostics.gate_ablation import GateAblationConfig, build_matched_gate_suite
from rtm_acoustic.diagnostics.matched_budget import BudgetMatchError, update_l2
from rtm_acoustic.diagnostics.update_reliability import array_hash
from rtm_acoustic.scripts._common import ensure_output_tree, read_simple_yaml, write_json


def _save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("wb") as handle:
        np.save(handle, np.asarray(array, dtype=np.float32))
    tmp.replace(path)


def _first_float(config: dict, key: str, default: float) -> float:
    values = config.get(key)
    if isinstance(values, list) and values:
        return float(values[0])
    if values is None:
        return float(default)
    return float(values)


def _max_float(config: dict, key: str, default: float) -> float:
    values = config.get(key)
    if isinstance(values, list) and values:
        return float(max(float(value) for value in values))
    if values is None:
        return float(default)
    return float(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ECG reliability-gate ablation when diagnostics are available.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    config = read_simple_yaml(args.config)
    output_dir = root / config.get("output_dir", "rtm_acoustic/outputs/salt_reliability_gate_v1")
    ensure_output_tree(output_dir)
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit("BLOCKED_MISSING_REPLAY_MANIFEST")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "READY":
        raise SystemExit(manifest.get("status", "BLOCKED"))
    delta_model = np.load(output_dir / "diagnostics" / "delta_model.npy").astype(np.float32, copy=False)
    illumination = np.load(output_dir / "diagnostics" / "illumination_score.npy").astype(np.float32, copy=False)
    consensus = np.load(output_dir / "diagnostics" / "gradient_consensus.npy").astype(np.float32, copy=False)
    reliability = np.load(output_dir / "diagnostics" / "ecg_reliability_score.npy").astype(np.float32, copy=False)
    ablation_config = GateAblationConfig(
        coverage=_first_float(config, "coverage_candidates", 0.3635),
        alpha_max=_max_float(config, "alpha_max_candidates", 0.3),
        sigma_x=_first_float(config, "sigma_x_candidates", 4.0),
        sigma_z=_first_float(config, "sigma_z_candidates", 4.0),
    )
    target_update_l2 = float(manifest.get("target_update_l2", 0.05 * update_l2(np.ones_like(delta_model), delta_model)))
    try:
        gates = build_matched_gate_suite(
            delta_model=delta_model,
            illumination=illumination,
            consensus=consensus,
            reliability=reliability,
            target_update_l2=target_update_l2,
            config=ablation_config,
        )
    except BudgetMatchError as exc:
        write_json(
            output_dir / "gates" / "gate_ablation_manifest.json",
            {
                "status": "BLOCKED_BUDGET_MATCH_FAILED",
                "message": str(exc),
                "target_update_l2": target_update_l2,
            },
        )
        raise SystemExit("BLOCKED_BUDGET_MATCH_FAILED") from exc

    records = []
    for name, gate in gates.items():
        path = output_dir / "gates" / f"{name}.npy"
        _save_array(path, gate)
        records.append(
            {
                "name": name,
                "path": str(path.relative_to(output_dir)),
                "update_l2": update_l2(gate, delta_model),
                "alpha_min": float(np.min(gate)),
                "alpha_max": float(np.max(gate)),
                "array_hash": array_hash(gate),
            }
        )
    write_json(
        output_dir / "gates" / "gate_ablation_manifest.json",
        {
            "status": "READY",
            "target_update_l2": target_update_l2,
            "gate_count": len(records),
            "gates": records,
        },
    )
    print(output_dir)


if __name__ == "__main__":
    main()
