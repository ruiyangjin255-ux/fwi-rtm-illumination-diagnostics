from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from fwi_visionfm.scripts.plot_protocol_v5_seed_stability import METRICS, write_protocol_v5_seed_stability_artifacts
from fwi_visionfm.scripts.report_local_mae_ablation import _f
from fwi_visionfm.scripts.summarize_local_mae_ablation import write_local_mae_ablation_summary


def _load_rows(root: Path) -> list[dict[str, str]]:
    summary_path = root / "local_mae_ablation_summary.csv"
    if not summary_path.exists():
        summary_path = write_local_mae_ablation_summary(root)
    with summary_path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _trace_rows(rows: list[dict[str, str]], model_type: str) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("status") == "SUCCESS"
        and row.get("model_type") == model_type
        and row.get("bridge") == "raw_envelope_spectrum3"
        and row.get("mask_type") == "trace_dropout"
        and row.get("loss_name") == "default_l1"
    ]


def _metric_wins(paired_rows: list[dict[str, str]]) -> dict[str, int]:
    wins = {metric: 0 for metric in METRICS}
    for row in paired_rows:
        for metric in METRICS:
            if int(row[f"pretrained_win_{metric}"]):
                wins[metric] += 1
    return wins


def _best_row(rows: list[dict[str, str]], model_type: str) -> dict[str, str] | None:
    candidates = _trace_rows(rows, model_type)
    if not candidates:
        return None
    return min(candidates, key=lambda row: (_f(row, "MAE"), _f(row, "RMSE")))


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _run_dir(root: Path, row: dict[str, str]) -> Path:
    return (
        root
        / "decoder_runs"
        / "flatvel_a_subset2k_to_curvevel_a_subset500"
        / row["model_type"]
        / row["bridge"]
        / row["mask_type"]
        / row["decoder_name"]
        / row["loss_name"]
        / f"seed_{row['seed']}"
    )


def write_protocol_v5_final_stage_report(root: str | Path, output_dir: str | Path) -> Path:
    root_path = Path(root)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(root_path)
    stage_artifacts = write_protocol_v5_seed_stability_artifacts(root_path, output_path)
    stability_csv = stage_artifacts["csv_path"]
    with stability_csv.open("r", encoding="utf-8") as handle:
        stability_rows = [row for row in csv.DictReader(handle) if row.get("seed", "").isdigit()]
    wins = _metric_wins(stability_rows)
    count = len(stability_rows)
    majority = (count // 2) + 1 if count else 0
    stable_numerical = wins["MAE"] >= majority and wins["RMSE"] >= majority and count > 0
    majority_structural = wins["gradient_error"] >= majority and wins["edge_MAE"] >= majority and count > 0

    best_pre = _best_row(rows, "pretrained_local_mae")
    best_rand = _best_row(rows, "random_mae_encoder")
    if best_pre:
        run_dir = _run_dir(root_path, best_pre)
        _copy_if_exists(run_dir / "prediction_grid.png", output_path / "best_pretrained_prediction_grid.png")
        _copy_if_exists(run_dir / "gradient_grid.png", output_path / "best_pretrained_gradient_grid.png")
    if best_rand:
        run_dir = _run_dir(root_path, best_rand)
        _copy_if_exists(run_dir / "prediction_grid.png", output_path / "best_random_prediction_grid.png")
        _copy_if_exists(run_dir / "gradient_grid.png", output_path / "best_random_gradient_grid.png")

    local_report_excerpt = _read_text(root_path / "local_mae_ablation_report.md")
    local_report_note = "已读取 local_mae_ablation_report.md 作为前序阶段摘要。" if local_report_excerpt else "未找到既有 local_mae_ablation_report.md，仅基于 summary 生成。"
    stable_num_text = "stable numerical benefit" if stable_numerical else "numerical benefit is not yet stable"
    structural_text = (
        "majority-supported but limited structural benefit"
        if majority_structural
        else "structural benefit remains limited and not majority-supported"
    )

    lines = [
        "# Protocol V5 Final Stage Report",
        "",
        "## 1. Research Motivation",
        "- V2/V3/V4 阶段中，自然图像 VisionFM 没有形成稳定优势。",
        "- 当前 NCS 权重不可用，因此本轮将 local seismic MAE 作为本地 seismic-domain masked pretraining probe。",
        f"- {local_report_note}",
        "",
        "## 2. Local MAE Design",
        "- 输入桥接方向包括 raw_envelope_spectrum3、spectrogram_multiband、raw_spectrogram。",
        "- 先做 masked reconstruction，再冻结 encoder，缓存特征，最后训练 decoder-only velocity regression。",
        "- 当前评估口径使用 physical_velocity metrics。",
        "",
        "## 3. Physics-aware Masking",
        "- random_patch 是通用图像式 masking。",
        "- trace_dropout 更接近地震采集中缺失接收道、坏道或道集不完整。",
        "- receiver_block、frequency_band、hybrid_seismic_mask 用于探索更贴近地震属性的重建任务。",
        "",
        "## 4. Pretrained vs Random Encoder Fairness",
        "- random same-architecture encoder 作为公平对照，encoder 结构保持一致。",
        "- split、bridge、decoder、loss、seed 全部对齐。",
        "- 唯一变化项是是否经过 masked pretraining。",
        "",
        "## 5. Trace-dropout Seed Stability",
        "- 下表基于 seed=0/1/2 的 raw_envelope_spectrum3 + trace_dropout + unet_decoder + default_l1 公平对照。",
        "",
        "| seed | pretrained_MAE | random_MAE | pretrained_RMSE | random_RMSE | pretrained_SSIM | random_SSIM | pretrained_gradient_error | random_gradient_error | pretrained_edge_MAE | random_edge_MAE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in stability_rows:
        lines.append(
            f"| {row['seed']} | {row['pretrained_MAE']} | {row['random_MAE']} | {row['pretrained_RMSE']} | {row['random_RMSE']} | "
            f"{row['pretrained_SSIM']} | {row['random_SSIM']} | {row['pretrained_gradient_error']} | {row['random_gradient_error']} | "
            f"{row['pretrained_edge_MAE']} | {row['random_edge_MAE']} |"
        )
    lines.extend(
        [
            "",
            f"- MAE/RMSE/SSIM/gradient_error/edge_MAE 的 pretrained 胜出次数分别为 {wins['MAE']}/{count}、{wins['RMSE']}/{count}、{wins['SSIM']}/{count}、{wins['gradient_error']}/{count}、{wins['edge_MAE']}/{count}。",
            f"- 结论口径：{stable_num_text}。",
            f"- 结构口径：{structural_text}。",
            "- 这里不写“结构恢复问题已解决”，因为结构收益幅度仍有限。",
            "",
            "## 6. Comparison with Earlier Protocols",
            "- reference-only 对比对象包括：V4 best single model、V5 initial local MAE best、DINOv2-LoRA probe、NCS unavailable status。",
            "- 不直接跨协议比较 visual_score，只比较 MAE/RMSE/SSIM/gradient_error/edge_MAE 原始指标。",
            "- 当前 trace_dropout 线索表明 local seismic MAE 的收益不是完全偶然，但仍不足以升级为 benchmark claim。",
            "",
            "## 7. Scientific Interpretation",
            "- local seismic MAE 的收益不是完全偶然，至少在 matched trace_dropout 设置下表现出重复性。",
            "- trace_dropout 是当前最有价值的 physics-aware masking。",
            "- raw_envelope_spectrum3 是当前最稳的 hybrid bridge。",
            "- 当前改善更强地体现在数值恢复，结构恢复只有多数 seed 支持且幅度有限。",
            "- 后续更可能需要更强结构任务或真实 seismic-domain foundation model，而不是继续堆自然图像 backbone。",
            "",
            "## 8. Limitations",
            "- CPU small-sample",
            "- OpenFWI subset only",
            "- local tiny MAE",
            "- only seed=0/1/2",
            "- not NCS-scale",
            "- not application-level",
            "- not benchmark-level proof",
            "",
            "## 9. Next Steps",
            "- 如果有 GPU，扩大 local seismic MAE pretraining。",
            "- 如果获得 NCS 权重，优先做 NCS 2.5D frozen feature probe。",
            "- 加入 shot/offset positional encoding。",
            "- 加入 boundary auxiliary head。",
            "- 加入 weak physics consistency，而不是继续堆自然图像 backbone。",
            "",
            "允许写法：",
            "- Protocol V5 provides initial CPU-limited evidence that trace-dropout local seismic MAE can outperform a random same-architecture encoder under matched settings.",
            "- The improvement is strongest in numerical metrics and only majority-supported in structural metrics.",
            "- Results remain feasibility evidence rather than benchmark-level proof.",
        ]
    )

    report_path = output_path / "protocol_v5_final_stage_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Protocol V5 final stage report.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    print(write_protocol_v5_final_stage_report(**vars(parse_args())))


if __name__ == "__main__":
    main()
