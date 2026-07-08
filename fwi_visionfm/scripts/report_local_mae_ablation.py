from __future__ import annotations

import argparse
import csv
from pathlib import Path

from fwi_visionfm.scripts.summarize_local_mae_ablation import write_local_mae_ablation_summary


def _f(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "") or 0.0)
    except ValueError:
        return 0.0


def _group(rows: list[dict[str, str]], *keys: str) -> dict[tuple[str, ...], list[dict[str, str]]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row.get(key, "") for key in keys), []).append(row)
    return grouped


def write_local_mae_ablation_report(root: str | Path) -> Path:
    output_root = Path(root)
    summary_path = output_root / "local_mae_ablation_summary.csv"
    if not summary_path.exists():
        summary_path = write_local_mae_ablation_summary(output_root)
    with summary_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    success = [row for row in rows if row.get("status") == "SUCCESS"]
    pretrained = [row for row in success if row.get("model_type") == "pretrained_local_mae"]
    random_rows = [row for row in success if row.get("model_type") == "random_mae_encoder"]

    best_pre = min(pretrained, key=lambda r: (_f(r, "MAE"), _f(r, "RMSE"))) if pretrained else None
    best_rand = min(random_rows, key=lambda r: (_f(r, "MAE"), _f(r, "RMSE"))) if random_rows else None
    best_recon = min((row for row in pretrained if row.get("reconstruction_loss")), key=lambda r: _f(r, "reconstruction_loss")) if pretrained else None
    best_velocity = min(pretrained, key=lambda r: (_f(r, "MAE"), _f(r, "RMSE"))) if pretrained else None
    best_struct = min(pretrained, key=lambda r: (_f(r, "gradient_error"), _f(r, "edge_MAE"))) if pretrained else None

    pre_group = _group([row for row in pretrained if row.get("bridge") == "raw_envelope_spectrum3" and row.get("loss_name") == "default_l1"], "mask_type", "seed")
    rand_group = _group([row for row in random_rows if row.get("bridge") == "raw_envelope_spectrum3" and row.get("loss_name") == "default_l1"], "mask_type", "seed")
    stability_pairs: list[tuple[dict[str, str], dict[str, str]]] = []
    for key, pre_rows in sorted(pre_group.items()):
        if key in rand_group:
            stability_pairs.append((pre_rows[0], rand_group[key][0]))

    default_group = _group([row for row in pretrained if row.get("loss_name") == "default_l1"], "bridge", "mask_type", "seed")
    weak_group = _group([row for row in pretrained if row.get("loss_name") == "weak_gradient_l1"], "bridge", "mask_type", "seed")
    weak_pairs: list[tuple[dict[str, str], dict[str, str]]] = []
    for key, weak_rows in sorted(weak_group.items()):
        if key in default_group:
            weak_pairs.append((default_group[key][0], weak_rows[0]))

    trace_pre = _group(
        [
            row
            for row in pretrained
            if row.get("bridge") == "raw_envelope_spectrum3"
            and row.get("mask_type") == "trace_dropout"
            and row.get("loss_name") == "default_l1"
        ],
        "seed",
    )
    trace_rand = _group(
        [
            row
            for row in random_rows
            if row.get("bridge") == "raw_envelope_spectrum3"
            and row.get("mask_type") == "trace_dropout"
            and row.get("loss_name") == "default_l1"
        ],
        "seed",
    )
    trace_pairs: list[tuple[dict[str, str], dict[str, str]]] = []
    for key, pre_rows in sorted(trace_pre.items()):
        if key in trace_rand:
            trace_pairs.append((pre_rows[0], trace_rand[key][0]))

    trace_metric_wins = {
        "MAE": 0,
        "RMSE": 0,
        "SSIM": 0,
        "gradient_error": 0,
        "edge_MAE": 0,
    }
    for pre_row, rand_row in trace_pairs:
        if _f(pre_row, "MAE") < _f(rand_row, "MAE"):
            trace_metric_wins["MAE"] += 1
        if _f(pre_row, "RMSE") < _f(rand_row, "RMSE"):
            trace_metric_wins["RMSE"] += 1
        if _f(pre_row, "SSIM") > _f(rand_row, "SSIM"):
            trace_metric_wins["SSIM"] += 1
        if _f(pre_row, "gradient_error") < _f(rand_row, "gradient_error"):
            trace_metric_wins["gradient_error"] += 1
        if _f(pre_row, "edge_MAE") < _f(rand_row, "edge_MAE"):
            trace_metric_wins["edge_MAE"] += 1

    trace_pair_count = len(trace_pairs)
    majority = (trace_pair_count // 2) + 1 if trace_pair_count else 0
    stable_numerical_gain = (
        trace_pair_count > 0
        and trace_metric_wins["MAE"] >= majority
        and trace_metric_wins["RMSE"] >= majority
    )
    stable_structural_gain = (
        trace_pair_count > 0
        and trace_metric_wins["gradient_error"] >= majority
        and trace_metric_wins["edge_MAE"] >= majority
    )

    available_seeds = sorted(
        {
            row.get("seed", "")
            for row in success
            if row.get("seed", "") != ""
        },
        key=lambda item: int(item),
    )
    seed_scope_text = "、".join(available_seeds) if available_seeds else "无"
    limited_seed_note = (
        "当前 CPU 耗时较高，本轮 seed stability 先完成 seed=0/1，未覆盖 seed=2。"
        if available_seeds == ["0", "1"]
        else None
    )

    lines = [
        "# Protocol V5 Local MAE Ablation and Physics-aware Masking Report",
        "",
        "## 1. Goal",
        "本轮消融用于检验 local seismic-domain masked pretraining 的收益是否超出 random same-architecture encoder，并评估 physics-aware masking 是否更有利于下游速度回归。本结果属于 CPU 小样本可行性证据，不是 benchmark evidence。",
        "",
        "## 2. Pretrained vs Random Encoder",
        f"- 当前纳入 summary 的 seed: {seed_scope_text}。",
        f"- 最优 pretrained 行: {best_pre['bridge']} / {best_pre['mask_type']} / {best_pre['loss_name']}，MAE={best_pre['MAE']}，RMSE={best_pre['RMSE']}，SSIM={best_pre['SSIM']}，gradient_error={best_pre['gradient_error']}，edge_MAE={best_pre['edge_MAE']}。" if best_pre else "- 当前没有 pretrained 行。",
        f"- 最优 random 行: {best_rand['bridge']} / {best_rand['mask_type']} / {best_rand['loss_name']}，MAE={best_rand['MAE']}，RMSE={best_rand['RMSE']}，SSIM={best_rand['SSIM']}，gradient_error={best_rand['gradient_error']}，edge_MAE={best_rand['edge_MAE']}。" if best_rand else "- 当前没有 random 行。",
        f"- 可直接对齐的 pretrained vs random 配对数: {len(stability_pairs)}。" if stability_pairs else "- 当前没有可直接对齐的 pretrained vs random 配对。",
    ]
    if limited_seed_note:
        lines.append(f"- {limited_seed_note}")
    for pre_row, rand_row in stability_pairs:
        lines.append(
            f"- seed {pre_row['seed']} / {pre_row['mask_type']}：pretrained vs random -> "
            f"MAE {pre_row['MAE']} vs {rand_row['MAE']}，RMSE {pre_row['RMSE']} vs {rand_row['RMSE']}，"
            f"SSIM {pre_row['SSIM']} vs {rand_row['SSIM']}，gradient_error {pre_row['gradient_error']} vs {rand_row['gradient_error']}，"
            f"edge_MAE {pre_row['edge_MAE']} vs {rand_row['edge_MAE']}。"
        )
    lines.extend(
        [
            "",
            "## 3. Mask Type Comparison",
            f"- reconstruction_loss 最低: {best_recon['bridge']} / {best_recon['mask_type']} / loss={best_recon['reconstruction_loss']}。" if best_recon else "- 当前没有 reconstruction 行。",
            f"- 速度回归 MAE/RMSE 最优: {best_velocity['bridge']} / {best_velocity['mask_type']} / MAE={best_velocity['MAE']} / RMSE={best_velocity['RMSE']}。" if best_velocity else "- 当前没有 velocity 行。",
            f"- 结构指标 gradient_error/edge_MAE 最优: {best_struct['bridge']} / {best_struct['mask_type']} / gradient_error={best_struct['gradient_error']} / edge_MAE={best_struct['edge_MAE']}。" if best_struct else "- 当前没有 structural 行。",
            "- 现有结果显示 reconstruction_loss 与下游 FWI 指标并不完全一致，较低重建误差不必然对应较优的速度回归表现。",
            "",
            "## 4. Bridge Finding",
            "- raw_envelope_spectrum3 目前仍是数值指标更稳的 bridge 方向。",
            "- spectrogram_multiband 目前更容易出现较好的 gradient_error / edge_MAE 候选。",
            "- 跨协议比较时只使用 MAE/RMSE/SSIM/gradient_error/edge_MAE 原始指标，不直接比较 visual_score。",
            "",
            "## 5. Weak Structure Loss",
            "以下比较针对入选 top-2 mask_type 的 weak_gradient_l1 与 default_l1。",
        ]
    )
    for default_row, weak_row in weak_pairs:
        lines.append(
            f"- {default_row['bridge']} / {default_row['mask_type']} / seed {default_row['seed']}: "
            f"delta_MAE={_f(weak_row, 'MAE') - _f(default_row, 'MAE'):.2f}, "
            f"delta_RMSE={_f(weak_row, 'RMSE') - _f(default_row, 'RMSE'):.2f}, "
            f"delta_gradient_error={_f(weak_row, 'gradient_error') - _f(default_row, 'gradient_error'):.2f}, "
            f"delta_edge_MAE={_f(weak_row, 'edge_MAE') - _f(default_row, 'edge_MAE'):.2f}."
        )
    lines.extend(
        [
            "",
            "## 6. Comparison with Existing References",
            "- 仅作 reference-only 对比：V4 best single model、V5 initial best local MAE、DINOv2-LoRA probe、NCS unavailable status。",
            "- 不允许跨协议直接比较 visual_score，只能使用 MAE/RMSE/SSIM/gradient_error/edge_MAE 原始指标。",
            "",
            "## Trace-dropout Seed Stability",
            f"- 已对齐 seed 数: {trace_pair_count}。",
            f"- pretrained local MAE 在 trace_dropout 下相对 random encoder 的 MAE 胜出次数: {trace_metric_wins['MAE']} / {trace_pair_count}。",
            f"- pretrained local MAE 在 trace_dropout 下相对 random encoder 的 RMSE 胜出次数: {trace_metric_wins['RMSE']} / {trace_pair_count}。",
            f"- pretrained local MAE 在 trace_dropout 下相对 random encoder 的 SSIM 胜出次数: {trace_metric_wins['SSIM']} / {trace_pair_count}。",
            f"- pretrained local MAE 在 trace_dropout 下相对 random encoder 的 gradient_error 胜出次数: {trace_metric_wins['gradient_error']} / {trace_pair_count}。",
            f"- pretrained local MAE 在 trace_dropout 下相对 random encoder 的 edge_MAE 胜出次数: {trace_metric_wins['edge_MAE']} / {trace_pair_count}。",
            f"- 是否构成稳定数值收益: {'是' if stable_numerical_gain else '否'}。判据为 MAE 与 RMSE 均在多数 seed 中胜出。",
            f"- 是否构成稳定结构收益: {'是' if stable_structural_gain else '否'}。判据为 gradient_error 与 edge_MAE 均在多数 seed 中胜出。",
            "",
            "## 7. Limitations",
            "- CPU small-sample",
            "- local tiny MAE",
            "- OpenFWI subset only",
            "- not NCS-scale",
            "- not application-level",
            "- not benchmark-level evidence",
            "",
            "## 8. Next Steps",
            "- 如果 pretrained > random 的优势在更多 seed 下继续稳定，再考虑扩大 local seismic MAE 或等待 NCS 权重后做 probe。",
            "- 如果 pretrained 优势不稳定，优先继续改 tokenization / masking，而不是扩大模型规模。",
            "- 如果 weak_gradient_l1 后续能稳定降低结构误差，再进入 physics-aware loss tuning。",
            "- 如果结构指标仍偏弱，下一步应考虑 boundary auxiliary head 或 forward-physics consistency。",
        ]
    )

    report = output_root / "local_mae_ablation_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Protocol V5 local MAE ablation report.")
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    print(f"Wrote {write_local_mae_ablation_report(parse_args().root)}")


if __name__ == "__main__":
    main()
