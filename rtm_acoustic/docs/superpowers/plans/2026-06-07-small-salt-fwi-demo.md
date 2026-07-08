# 小范围盐丘模型 FWI 演示 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 `D:\ryjin\rtm_acoustic` 工程中新增一个可运行的小范围 SEG/Salt FWI 演示，并输出中文解释文档和可视化结果。

**Architecture:** 新增独立脚本 `run_small_salt_fwi.py`，复用 `rtm_acoustic.acoustic_rtm` 中的模型读取、平滑、正演、波场和吸收边界工具。脚本内部保持函数化，便于单元测试；输出写入独立目录 `outputs/small_salt_fwi_demo`，不覆盖已有 RTM 成果。

**Tech Stack:** Python、NumPy、Matplotlib、pytest、现有 `rtm_acoustic.acoustic_rtm` 模块。

---

## File Structure

- Create: `D:\ryjin\rtm_acoustic\run_small_salt_fwi.py`
  - 负责模型裁剪、观测合成、FWI 迭代、图件输出和 `summary.json`。
- Create: `D:\ryjin\rtm_acoustic\tests\test_small_salt_fwi.py`
  - 覆盖裁剪窗口、初始模型、误差计算、更新限制和一次演示运行的轻量 smoke 测试。
- Create: `D:\ryjin\rtm_acoustic\docs\small_salt_fwi_and_illumination.md`
  - 中文解释 RTM 照明归一化、论文照明优化与本次 FWI 演示的关系。

## Task 1: FWI 工具函数与测试

**Files:**
- Create: `D:\ryjin\rtm_acoustic\tests\test_small_salt_fwi.py`
- Create: `D:\ryjin\rtm_acoustic\run_small_salt_fwi.py`

- [ ] **Step 1: 编写失败测试**

```python
import numpy as np

from run_small_salt_fwi import (
    FWIConfig,
    build_initial_model,
    choose_default_crop,
    clip_velocity_update,
    compute_record_misfit,
)


def test_choose_default_crop_returns_requested_window():
    model = np.zeros((230, 676), dtype=np.float32)
    z0, x0, nz, nx = choose_default_crop(model, crop_nz=70, crop_nx=120)
    assert 0 <= z0 <= model.shape[0] - nz
    assert 0 <= x0 <= model.shape[1] - nx
    assert (nz, nx) == (70, 120)


def test_build_initial_model_smooths_true_model():
    true_model = np.full((40, 50), 2000.0, dtype=np.float32)
    true_model[20:, 20:35] = 4200.0
    initial = build_initial_model(true_model, radius_z=3, radius_x=4, passes=2)
    assert initial.shape == true_model.shape
    assert np.isfinite(initial).all()
    assert initial.max() < true_model.max()
    assert initial.min() >= true_model.min()


def test_compute_record_misfit_uses_half_l2_norm():
    residual = np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float32)
    predicted = residual.copy()
    observed = np.zeros_like(predicted)
    misfit = compute_record_misfit(predicted, observed)
    assert np.isclose(misfit, 0.5 * np.mean(residual * residual))


def test_clip_velocity_update_limits_step_amplitude():
    update = np.array([[-100.0, 0.0, 100.0]], dtype=np.float32)
    clipped = clip_velocity_update(update, max_update=20.0)
    assert clipped.min() >= -20.0
    assert clipped.max() <= 20.0
    assert clipped.dtype == np.float32
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```powershell
python -m pytest D:\ryjin\rtm_acoustic\tests\test_small_salt_fwi.py -v
```

Expected: FAIL，原因是 `run_small_salt_fwi.py` 或对应函数尚不存在。

- [ ] **Step 3: 实现最小工具函数**

在 `D:\ryjin\rtm_acoustic\run_small_salt_fwi.py` 中创建脚本骨架，包含：

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rtm_acoustic.acoustic_rtm import smooth_velocity_model


Array = np.ndarray
ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT.parent / "fd2d_pml" / "vel" / "seg676x230.bin"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "small_salt_fwi_demo"


@dataclass(frozen=True)
class FWIConfig:
    crop_nx: int = 120
    crop_nz: int = 70
    dx: float = 10.0
    dz: float = 10.0
    dt: float = 0.001
    nt: int = 450
    f0: float = 10.0
    source_z: int = 4
    receiver_z: int = 4
    absorb_cells: int = 12
    fd_order: int = 4
    iterations: int = 4
    max_update: float = 35.0
    velocity_min: float = 1450.0
    velocity_max: float = 4600.0


def choose_default_crop(model: Array, crop_nz: int, crop_nx: int) -> tuple[int, int, int, int]:
    nz, nx = model.shape
    if crop_nz <= 0 or crop_nx <= 0:
        raise ValueError("裁剪尺寸必须为正数")
    if crop_nz > nz or crop_nx > nx:
        raise ValueError("裁剪窗口不能超过模型尺寸")
    z0 = min(max(70, 0), nz - crop_nz)
    x0 = min(max(310, 0), nx - crop_nx)
    return int(z0), int(x0), int(crop_nz), int(crop_nx)


def build_initial_model(true_model: Array, radius_z: int = 5, radius_x: int = 8, passes: int = 3) -> Array:
    return smooth_velocity_model(true_model, radius_z=radius_z, radius_x=radius_x, passes=passes)


def compute_record_misfit(predicted: Array, observed: Array) -> float:
    residual = np.asarray(predicted, dtype=np.float32) - np.asarray(observed, dtype=np.float32)
    return float(0.5 * np.mean(residual * residual))


def clip_velocity_update(update: Array, max_update: float) -> Array:
    return np.clip(np.asarray(update, dtype=np.float32), -float(max_update), float(max_update)).astype(np.float32)
```

- [ ] **Step 4: 运行测试确认通过**

Run:

```powershell
python -m pytest D:\ryjin\rtm_acoustic\tests\test_small_salt_fwi.py -v
```

Expected: PASS。

## Task 2: 单炮残差反传与模型更新

**Files:**
- Modify: `D:\ryjin\rtm_acoustic\run_small_salt_fwi.py`
- Modify: `D:\ryjin\rtm_acoustic\tests\test_small_salt_fwi.py`

- [ ] **Step 1: 添加更新方向测试**

追加到测试文件：

```python
from rtm_acoustic.acoustic_rtm import RTMConfig
from run_small_salt_fwi import compute_update_direction


def test_compute_update_direction_returns_finite_model_shaped_array(tmp_path):
    nz, nx = 26, 32
    velocity = np.full((nz, nx), 2000.0, dtype=np.float32)
    cfg = RTMConfig(
        nx=nx,
        nz=nz,
        dx=10.0,
        dz=10.0,
        dt=0.001,
        nt=35,
        f0=12.0,
        source_x=nx // 2,
        source_z=4,
        receiver_z=4,
        absorb_cells=6,
        fd_order=4,
    )
    residual = np.zeros((cfg.nt, cfg.nx), dtype=np.float32)
    residual[10:20, 8:24] = 1.0
    update = compute_update_direction(
        velocity=velocity,
        config=cfg,
        residual=residual,
        source_wavefield_path=tmp_path / "source.dat",
    )
    assert update.shape == velocity.shape
    assert np.isfinite(update).all()
```

- [ ] **Step 2: 运行新增测试确认失败**

Run:

```powershell
python -m pytest D:\ryjin\rtm_acoustic\tests\test_small_salt_fwi.py::test_compute_update_direction_returns_finite_model_shaped_array -v
```

Expected: FAIL，原因是 `compute_update_direction` 尚不存在。

- [ ] **Step 3: 实现更新方向函数**

在脚本中新增 imports：

```python
from rtm_acoustic.acoustic_rtm import (
    RTMConfig,
    forward_model,
    laplacian,
    make_absorbing_mask,
    open_wavefield,
    validate_config,
)
```

新增函数：

```python
def compute_update_direction(
    velocity: Array,
    config: RTMConfig,
    residual: Array,
    source_wavefield_path: str | Path,
) -> Array:
    """用残差反传和正传波场相关构造速度更新方向。"""
    validate_config(velocity, config)
    if residual.shape != (config.nt, config.nx):
        raise ValueError("残差记录形状必须与配置一致")

    source_wavefield = open_wavefield(source_wavefield_path, config, mode="r")
    mask = make_absorbing_mask(config)
    velocity2_dt2 = (velocity.astype(np.float32) ** 2) * np.float32(config.dt * config.dt)
    prev = np.zeros((config.nz, config.nx), dtype=np.float32)
    curr = np.zeros_like(prev)
    gradient = np.zeros_like(prev)

    for it in range(config.nt - 1, -1, -1):
        curr[config.receiver_z, :] += residual[it, :]
        curr *= mask
        source = np.asarray(source_wavefield[it, :, :], dtype=np.float32)
        gradient += laplacian(source, config.dx, config.dz, config.fd_order) * curr
        nxt = 2.0 * curr - prev + velocity2_dt2 * laplacian(curr, config.dx, config.dz, config.fd_order)
        nxt *= mask
        prev, curr = curr, nxt.astype(np.float32, copy=False)

    gradient = smooth_velocity_model(gradient, radius_z=2, radius_x=2, passes=1)
    scale = float(np.percentile(np.abs(gradient), 99.0))
    if scale <= 0.0 or not np.isfinite(scale):
        return np.zeros_like(gradient, dtype=np.float32)
    return (-gradient / scale).astype(np.float32)
```

- [ ] **Step 4: 运行测试确认通过**

Run:

```powershell
python -m pytest D:\ryjin\rtm_acoustic\tests\test_small_salt_fwi.py -v
```

Expected: PASS。

## Task 3: FWI 主循环与输出

**Files:**
- Modify: `D:\ryjin\rtm_acoustic\run_small_salt_fwi.py`
- Modify: `D:\ryjin\rtm_acoustic\tests\test_small_salt_fwi.py`

- [ ] **Step 1: 添加 smoke 测试**

追加到测试文件：

```python
from run_small_salt_fwi import run_fwi_demo


def test_run_fwi_demo_writes_summary_and_reduces_misfit(tmp_path):
    true_model = np.full((34, 42), 2000.0, dtype=np.float32)
    true_model[18:, 14:30] = 2600.0
    cfg = FWIConfig(crop_nx=42, crop_nz=34, nt=45, iterations=2, absorb_cells=6, max_update=20.0)
    summary = run_fwi_demo(
        true_model=true_model,
        config=cfg,
        output_dir=tmp_path,
        shot_positions=[14, 28],
        write_figures=False,
    )
    assert (tmp_path / "summary.json").exists()
    assert len(summary["misfit_history"]) == cfg.iterations + 1
    assert summary["final_misfit"] <= summary["initial_misfit"]
```

- [ ] **Step 2: 运行 smoke 测试确认失败**

Run:

```powershell
python -m pytest D:\ryjin\rtm_acoustic\tests\test_small_salt_fwi.py::test_run_fwi_demo_writes_summary_and_reduces_misfit -v
```

Expected: FAIL，原因是 `run_fwi_demo` 尚不存在。

- [ ] **Step 3: 实现主循环**

在脚本中新增 `json`、`matplotlib` 相关 import，并实现：

```python
def make_shot_config(base: FWIConfig, source_x: int, nx: int, nz: int) -> RTMConfig:
    return RTMConfig(
        nx=nx,
        nz=nz,
        dx=base.dx,
        dz=base.dz,
        dt=base.dt,
        nt=base.nt,
        f0=base.f0,
        source_x=int(source_x),
        source_z=base.source_z,
        receiver_z=base.receiver_z,
        absorb_cells=base.absorb_cells,
        fd_order=base.fd_order,
    )


def default_shot_positions(nx: int) -> list[int]:
    margin = max(8, nx // 8)
    return [margin, nx // 2, nx - margin - 1]


def run_fwi_demo(
    true_model: Array,
    config: FWIConfig,
    output_dir: str | Path,
    shot_positions: list[int] | None = None,
    write_figures: bool = True,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    true_model = np.asarray(true_model, dtype=np.float32)
    nz, nx = true_model.shape
    shots = shot_positions or default_shot_positions(nx)
    current = build_initial_model(true_model)
    initial = current.copy()
    observations = []
    for sx in shots:
        cfg = make_shot_config(config, sx, nx, nz)
        observations.append(forward_model(true_model, cfg))

    misfit_history: list[float] = []
    first_predicted = None
    first_observed = observations[0]

    for iteration in range(config.iterations + 1):
        total_misfit = 0.0
        total_update = np.zeros_like(current, dtype=np.float32)
        for shot_index, sx in enumerate(shots):
            cfg = make_shot_config(config, sx, nx, nz)
            wavefield_path = output_dir / f"iter{iteration:02d}_shot{shot_index:02d}_source.dat"
            predicted = forward_model(current, cfg, wavefield_path=wavefield_path)
            observed = observations[shot_index]
            total_misfit += compute_record_misfit(predicted, observed)
            if iteration == config.iterations and shot_index == 0:
                first_predicted = predicted.copy()
            if iteration < config.iterations:
                residual = predicted - observed
                total_update += compute_update_direction(current, cfg, residual, wavefield_path)
        misfit_history.append(float(total_misfit / len(shots)))
        if iteration < config.iterations:
            update = clip_velocity_update(total_update / max(len(shots), 1), config.max_update)
            trial = np.clip(current + update, config.velocity_min, config.velocity_max).astype(np.float32)
            trial_misfit = _evaluate_total_misfit(trial, observations, config, shots)
            if trial_misfit <= misfit_history[-1]:
                current = trial
            else:
                current = np.clip(current + 0.5 * update, config.velocity_min, config.velocity_max).astype(np.float32)

    summary = {
        "crop_shape": [int(nz), int(nx)],
        "shot_positions": [int(s) for s in shots],
        "iterations": int(config.iterations),
        "initial_misfit": float(misfit_history[0]),
        "final_misfit": float(misfit_history[-1]),
        "misfit_reduction_fraction": float((misfit_history[0] - misfit_history[-1]) / max(misfit_history[0], 1.0e-20)),
        "misfit_history": [float(v) for v in misfit_history],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    np.save(output_dir / "true_model.npy", true_model)
    np.save(output_dir / "initial_model.npy", initial)
    np.save(output_dir / "inverted_model.npy", current)
    if write_figures:
        _write_figures(output_dir, true_model, initial, current, misfit_history, first_observed, first_predicted)
    return summary
```

同时实现 `_evaluate_total_misfit` 和 `_write_figures`，其中 `_write_figures` 保存 spec 约定的 PNG 文件，所有图题中文。

- [ ] **Step 4: 运行 smoke 测试确认通过**

Run:

```powershell
python -m pytest D:\ryjin\rtm_acoustic\tests\test_small_salt_fwi.py -v
```

Expected: PASS。

## Task 4: 命令行入口与中文说明

**Files:**
- Modify: `D:\ryjin\rtm_acoustic\run_small_salt_fwi.py`
- Create: `D:\ryjin\rtm_acoustic\docs\small_salt_fwi_and_illumination.md`

- [ ] **Step 1: 添加 CLI 入口**

在脚本末尾添加：

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="小范围 SEG/Salt 声波 FWI 演示")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--crop-nx", type=int, default=FWIConfig.crop_nx)
    parser.add_argument("--crop-nz", type=int, default=FWIConfig.crop_nz)
    parser.add_argument("--nt", type=int, default=FWIConfig.nt)
    parser.add_argument("--iterations", type=int, default=FWIConfig.iterations)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FWIConfig(crop_nx=args.crop_nx, crop_nz=args.crop_nz, nt=args.nt, iterations=args.iterations)
    full_model = read_binary_model(args.model, nx=676, nz=230)
    z0, x0, crop_nz, crop_nx = choose_default_crop(full_model, config.crop_nz, config.crop_nx)
    true_model = full_model[z0 : z0 + crop_nz, x0 : x0 + crop_nx]
    summary = run_fwi_demo(true_model=true_model, config=config, output_dir=args.output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 编写中文说明文档**

创建 `D:\ryjin\rtm_acoustic\docs\small_salt_fwi_and_illumination.md`，内容包括：

```markdown
# 小范围盐丘 FWI 模拟与照明归一化差异说明

## 结论

本次小范围 FWI 模拟用于展示速度模型反演闭环：通过真实盐丘裁剪模型合成观测记录，再从平滑初始模型出发迭代降低数据残差。它和已有 RTM 照明归一化的目标不同。

## RTM 中的照明归一化

现有 RTM 代码中的震源照明归一化为 `I / (L_s + epsilon)`，源检双照明归一化为 `I / (sqrt(L_s L_r) + epsilon)`。二者作用在偏移成像结果上，主要用于削弱照明强弱造成的成像幅值差异。

## 论文中的照明优化

论文中的照明优化通常不只是图像除以照明场，而是把照明用于采集评价、双向传播覆盖分析、梯度预条件、Hessian 近似或分步多参数反演。它更接近反演过程中的条件改善，而不是单纯的成像显示增强。

## 与已有盐丘 RTM 结果的关系

已有 full RTM 诊断中低照明区占比约为 `0.0174`，说明当前 full aperture 盐丘 RTM 结果不是主要受照明缺失控制。源检双照明归一化与震源照明归一化保持较高相似性，Laplacian 和显示优化对结构可读性的影响更明显。

## 本次 FWI 演示的意义

FWI 直接比较预测记录和观测记录，通过数据残差更新速度模型。它回答的是“当前速度模型是否能解释观测数据”，而 RTM 照明归一化回答的是“已有偏移图像的幅值是否被照明不均衡影响”。因此，本次 FWI 输出应与 RTM 图像互补解释。
```

## Task 5: 端到端验证

**Files:**
- Read: `D:\ryjin\rtm_acoustic\outputs\small_salt_fwi_demo\summary.json`

- [ ] **Step 1: 运行单元测试**

Run:

```powershell
python -m pytest D:\ryjin\rtm_acoustic\tests\test_small_salt_fwi.py -v
```

Expected: PASS。

- [ ] **Step 2: 运行演示脚本**

Run:

```powershell
python D:\ryjin\rtm_acoustic\run_small_salt_fwi.py --iterations 3 --nt 360
```

Expected: 输出 JSON，且 `final_misfit <= initial_misfit`。

- [ ] **Step 3: 检查输出文件**

Run:

```powershell
Get-ChildItem -LiteralPath D:\ryjin\rtm_acoustic\outputs\small_salt_fwi_demo
```

Expected: 至少包含 `summary.json`、`true_model.png`、`initial_model.png`、`inverted_model.png`、`model_update.png`、`misfit_curve.png`、`shot_compare.png`。

- [ ] **Step 4: 检查中文文档占位**

Run:

```powershell
rg -n "TODO|TBD|placeholder|待定|占位" D:\ryjin\rtm_acoustic\docs\small_salt_fwi_and_illumination.md
```

Expected: 无匹配。

- [ ] **Step 5: 提交说明**

当前 `D:\ryjin\rtm_acoustic` 不是 git 仓库，跳过提交。最终答复中列出新增文件、验证命令和误差下降结果。
