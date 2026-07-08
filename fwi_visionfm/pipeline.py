from __future__ import annotations

import argparse
from pathlib import Path

from fwi_visionfm.build_stage_report import build_stage_report
from fwi_visionfm.config import load_yaml_config
from fwi_visionfm.evaluate_existing_checkpoints import evaluate_existing_checkpoints
from fwi_visionfm.generate_prediction_examples import generate_prediction_examples_batch
from fwi_visionfm.index_results import build_results_index, write_results_index_outputs
from fwi_visionfm.plot_protocol_comparison import plot_protocol_comparison


def run_pipeline_stage(*, config_path: str | Path, stage: str, allow_training: bool = False) -> dict:
    if stage == "all" and allow_training:
        raise ValueError("当前 pipeline 不实现训练阶段；--allow-training 仅作为显式保护位，不能触发训练。")
    config = load_yaml_config(config_path)
    outputs_root = Path(config["outputs_root"])
    protocol_dir = Path(config["protocol_dir"])
    results_index_json = outputs_root / "results_index.json"
    results_index_md = outputs_root / "results_index.md"
    eval_root = outputs_root / "protocol_v1_eval"
    figures_dir = outputs_root / "protocol_v1_figures"
    protocol_summary = outputs_root / "protocol_v1_summary.md"
    stage_report = outputs_root / "stage_report_cpu_protocol_v1.md"
    executed = []

    if stage in {"index", "all"}:
        payload = build_results_index(outputs_root)
        write_results_index_outputs(payload, results_index_json, results_index_md)
        executed.append("index")
    if stage in {"evaluate", "all"}:
        evaluate_existing_checkpoints(split_dir=protocol_dir, outputs_root=outputs_root, output_dir=eval_root, device=str(config.get("device", "cpu")))
        executed.append("evaluate")
    if stage in {"predict", "all"}:
        generate_prediction_examples_batch(
            protocol_dir=protocol_dir,
            outputs_root=outputs_root,
            output_subdir="prediction_examples",
            num_samples=int(config.get("num_samples", 8)),
            device=str(config.get("device", "cpu")),
        )
        executed.append("predict")
    if stage in {"plot", "all"}:
        plot_protocol_comparison(summary_path=protocol_summary, eval_csv=eval_root / "all_test_metrics.csv", output_dir=figures_dir)
        executed.append("plot")
    if stage in {"report", "all"}:
        build_stage_report(
            protocol_summary=protocol_summary,
            results_index=results_index_json,
            eval_csv=eval_root / "all_test_metrics.csv",
            figures_dir=figures_dir,
            output=stage_report,
        )
        executed.append("report")
    return {"stage": stage, "executed": executed}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU-only post-training pipeline.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("index", "evaluate", "predict", "plot", "report", "all"))
    parser.add_argument("--allow-training", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_pipeline_stage(config_path=args.config, stage=args.stage, allow_training=args.allow_training)
    print(payload)


if __name__ == "__main__":
    main()
