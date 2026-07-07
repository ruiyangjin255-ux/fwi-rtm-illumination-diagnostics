from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rtm_acoustic.diagnostics.matched_budget import BudgetMatchError, update_l2
from rtm_acoustic.diagnostics.update_reliability import (
    ReliabilityConfig,
    array_hash,
    build_reliability_components,
    build_soft_gate,
    config_hash,
)
from rtm_acoustic.scripts._common import ensure_output_tree, read_simple_yaml, write_json


def _save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("wb") as handle:
        np.save(handle, np.asarray(array, dtype=np.float32))
    tmp.replace(path)


def required_paths(config: dict, root: Path) -> list[Path]:
    diagnostics_dir = root / config.get("output_dir", "rtm_acoustic/outputs/salt_reliability_gate_v1") / "diagnostics"
    return [diagnostics_dir / name for name in config.get("required_diagnostics", [])]


def _first_float(config: dict, key: str, default: float) -> float:
    values = config.get(key)
    if isinstance(values, list) and values:
        return float(values[0])
    if values is None:
        return float(default)
    return float(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate required FWI diagnostics for reliability-gate replay.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    config = read_simple_yaml(args.config)
    output_dir = root / config.get("output_dir", "rtm_acoustic/outputs/salt_reliability_gate_v1")
    diagnostics_dir = output_dir / "diagnostics"
    ensure_output_tree(output_dir)
    missing = [str(path) for path in required_paths(config, root) if not path.exists()]
    status = "READY" if not missing else "BLOCKED_MISSING_FWI_DIAGNOSTICS"
    manifest = {
        "script": "replay_fwi_diagnostics.py",
        "smoke": bool(args.smoke),
        "status": status,
        "missing": missing,
        "message": "Existing FWI outputs lack per-shot-group gradients and source-adjoint energy proxies." if missing else "Required diagnostics are present.",
    }
    write_json(output_dir / "manifest.json", manifest)
    if missing:
        raise SystemExit(status)

    group_count = int(config.get("num_shot_groups", 4))
    group_gradients = np.stack(
        [np.load(diagnostics_dir / f"gradient_group_{index:02d}.npy").astype(np.float32, copy=False) for index in range(group_count)],
        axis=0,
    )
    delta_model = np.load(diagnostics_dir / "delta_model.npy").astype(np.float32, copy=False)
    hessian_proxy = np.load(diagnostics_dir / "source_adjoint_energy_proxy.npy").astype(np.float32, copy=False)
    reliability_config = ReliabilityConfig(
        eps=float(config.get("eps", 1.0e-8)),
        coverage=_first_float(config, "coverage_candidates", 0.3635),
        sigma_x=_first_float(config, "sigma_x_candidates", 4.0),
        sigma_z=_first_float(config, "sigma_z_candidates", 4.0),
        alpha_max=_first_float(config, "alpha_max_candidates", 0.3),
    )
    components = build_reliability_components(
        delta_model=delta_model,
        group_gradients=group_gradients,
        hessian_proxy=hessian_proxy,
        config=reliability_config,
    )
    target_fraction = float(config.get("target_update_fraction", 0.05))
    target_update_l2 = target_fraction * update_l2(np.ones_like(delta_model), delta_model)
    try:
        gate = build_soft_gate(
            components["reliability"],
            coverage=reliability_config.coverage,
            sigma_x=reliability_config.sigma_x,
            sigma_z=reliability_config.sigma_z,
            alpha_max=reliability_config.alpha_max,
            target_update_l2=target_update_l2,
            delta_model=delta_model,
        )
    except BudgetMatchError as exc:
        blocked = {
            **manifest,
            "status": "BLOCKED_BUDGET_MATCH_FAILED",
            "message": str(exc),
            "target_update_l2": float(target_update_l2),
        }
        write_json(output_dir / "manifest.json", blocked)
        raise SystemExit("BLOCKED_BUDGET_MATCH_FAILED") from exc

    component_paths = {
        "illumination": output_dir / "diagnostics" / "illumination_score.npy",
        "consensus": output_dir / "diagnostics" / "gradient_consensus.npy",
        "descent": output_dir / "diagnostics" / "descent_alignment.npy",
        "reliability": output_dir / "diagnostics" / "ecg_reliability_score.npy",
        "aggregate_gradient": output_dir / "diagnostics" / "aggregate_gradient_from_groups.npy",
    }
    for name, path in component_paths.items():
        _save_array(path, components[name])
    gate_path = output_dir / "gates" / "ecg_reliability_gate.npy"
    _save_array(gate_path, gate)

    ready_manifest = {
        **manifest,
        "status": "READY",
        "message": "Required diagnostics are present; ECG reliability components and preferred gate were generated.",
        "config_hash": config_hash(config),
        "target_update_fraction": float(target_fraction),
        "target_update_l2": float(target_update_l2),
        "matched_gate_update_l2": update_l2(gate, delta_model),
        "array_hashes": {
            "delta_model": array_hash(delta_model),
            "source_adjoint_energy_proxy": array_hash(hessian_proxy),
            "group_gradients": array_hash(group_gradients),
            "ecg_reliability_gate": array_hash(gate),
        },
        "outputs": {
            "components": {name: str(path.relative_to(output_dir)) for name, path in component_paths.items()},
            "gate": str(gate_path.relative_to(output_dir)),
        },
    }
    write_json(output_dir / "manifest.json", ready_manifest)
    print(output_dir)


if __name__ == "__main__":
    main()
