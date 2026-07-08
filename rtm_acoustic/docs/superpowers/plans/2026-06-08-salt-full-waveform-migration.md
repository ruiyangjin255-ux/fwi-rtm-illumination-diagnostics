# Salt Full Waveform Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个小范围盐丘全波形偏移对比实验，比较完整炮记录 RTM 与直达波扣除后的反射波 RTM。

**Architecture:** 新增独立脚本复用 `rtm_acoustic.acoustic_rtm` 的多炮 RTM 接口，不改动已有 RTM/FWI 主流程。脚本负责模型裁剪、迁移速度构造、两类 RTM 运行、指标统计、图件保存和中文 Markdown 总结。

**Tech Stack:** Python、NumPy、Matplotlib、pytest、现有 `rtm_acoustic.acoustic_rtm` 声波 RTM 工具。

---

### Task 1: 新增全波形偏移脚本骨架和配置

**Files:**
- Create: `D:\ryjin\rtm_acoustic\run_salt_full_waveform_migration.py`
- Test: `D:\ryjin\rtm_acoustic\tests\test_salt_full_waveform_migration.py`

- [ ] **Step 1: 写入失败测试**

```python
from pathlib import Path

import numpy as np

from run_salt_full_waveform_migration import FullWaveformMigrationConfig, default_shot_positions, choose_default_crop


def test_default_crop_and_shots_are_inside_model():
    model = np.ones((80, 120), dtype=np.float32)
    cfg = FullWaveformMigrationConfig(crop_nz=40, crop_nx=70)

    z0, x0, crop_nz, crop_nx = choose_default_crop(model, cfg.crop_nz, cfg.crop_nx)
    shots = default_shot_positions(crop_nx)

    assert 0 <= z0 <= model.shape[0] - crop_nz
    assert 0 <= x0 <= model.shape[1] - crop_nx
    assert shots == [8, 35, 61]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest D:\ryjin\rtm_acoustic\tests\test_salt_full_waveform_migration.py::test_default_crop_and_shots_are_inside_model -q`

Expected: FAIL，提示 `ModuleNotFoundError` 或找不到 `FullWaveformMigrationConfig`。

- [ ] **Step 3: 实现脚本骨架**

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rtm_acoustic.acoustic_rtm import read_binary_model, smooth_velocity_model


Array = np.ndarray
ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT.parent / "fd2d_pml" / "vel" / "seg676x230.bin"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "small_salt_full_waveform_migration"


@dataclass(frozen=True)
class FullWaveformMigrationConfig:
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
    min_illumination_fraction: float = 0.02


def choose_default_crop(model: Array, crop_nz: int, crop_nx: int) -> tuple[int, int, int, int]:
    """选择默认盐丘局部窗口，返回 z 起点、x 起点、深度点数和横向点数。"""
    nz, nx = np.asarray(model).shape
    if crop_nz <= 0 or crop_nx <= 0:
        raise ValueError("裁剪尺寸必须为正数")
    if crop_nz > nz or crop_nx > nx:
        raise ValueError("裁剪窗口不能超过模型尺寸")
    z0 = min(max(70, 0), nz - crop_nz)
    x0 = min(max(310, 0), nx - crop_nx)
    return int(z0), int(x0), int(crop_nz), int(crop_nx)


def default_shot_positions(nx: int) -> list[int]:
    """给小范围全波形偏移生成左、中、右三个炮点。"""
    margin = max(8, nx // 8)
    return [margin, nx // 2, nx - margin - 1]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest D:\ryjin\rtm_acoustic\tests\test_salt_full_waveform_migration.py::test_default_crop_and_shots_are_inside_model -q`

Expected: PASS。

### Task 2: 实现全波形与反射波 RTM 对比运行

**Files:**
- Modify: `D:\ryjin\rtm_acoustic\run_salt_full_waveform_migration.py`
- Modify: `D:\ryjin\rtm_acoustic\tests\test_salt_full_waveform_migration.py`

- [ ] **Step 1: 增加流程测试**

```python
def test_small_demo_writes_expected_outputs(tmp_path):
    true_model = np.full((30, 48), 2000.0, dtype=np.float32)
    true_model[14:, 20:30] = 2600.0
    cfg = FullWaveformMigrationConfig(crop_nz=30, crop_nx=48, nt=80, f0=12.0, absorb_cells=8, fd_order=4)

    summary = run_full_waveform_migration_demo(
        true_model=true_model,
        output_dir=tmp_path,
        config=cfg,
        shot_positions=[10, 24, 37],
    )

    assert summary["shot_count"] == 3
    assert summary["full_waveform"]["image_abs_p99"] > 0.0
    assert summary["reflection_only"]["image_abs_p99"] >= 0.0
    assert (tmp_path / "full_waveform_image.npy").exists()
    assert (tmp_path / "reflection_only_image.npy").exists()
    assert (tmp_path / "summary.json").exists()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest D:\ryjin\rtm_acoustic\tests\test_salt_full_waveform_migration.py::test_small_demo_writes_expected_outputs -q`

Expected: FAIL，提示 `run_full_waveform_migration_demo` 未定义。

- [ ] **Step 3: 实现核心运行函数**

实现 `make_rtm_config`、`_run_case`、`_image_metrics` 和 `run_full_waveform_migration_demo`。`_run_case` 调用 `multishot_reverse_time_migrate`，全波形分支传入 `subtract_direct_wave=False`，反射波分支传入 `subtract_direct_wave=True`。

- [ ] **Step 4: 运行流程测试确认通过**

Run: `python -m pytest D:\ryjin\rtm_acoustic\tests\test_salt_full_waveform_migration.py::test_small_demo_writes_expected_outputs -q`

Expected: PASS。

### Task 3: 保存图件和中文报告

**Files:**
- Modify: `D:\ryjin\rtm_acoustic\run_salt_full_waveform_migration.py`
- Modify: `D:\ryjin\rtm_acoustic\tests\test_salt_full_waveform_migration.py`

- [ ] **Step 1: 增加报告测试**

```python
def test_small_demo_writes_chinese_report(tmp_path):
    true_model = np.full((24, 36), 2100.0, dtype=np.float32)
    true_model[12:, 15:23] = 2800.0
    cfg = FullWaveformMigrationConfig(crop_nz=24, crop_nx=36, nt=70, absorb_cells=8, fd_order=4)

    run_full_waveform_migration_demo(
        true_model=true_model,
        output_dir=tmp_path,
        config=cfg,
        shot_positions=[8, 18, 27],
    )

    report = (tmp_path / "full_waveform_migration_summary.md").read_text(encoding="utf-8")
    assert "全波形偏移" in report
    assert "反射波偏移" in report
    assert (tmp_path / "migration_compare.png").exists()
    assert (tmp_path / "stacked_record_compare.png").exists()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest D:\ryjin\rtm_acoustic\tests\test_salt_full_waveform_migration.py::test_small_demo_writes_chinese_report -q`

Expected: FAIL，提示报告或图件不存在。

- [ ] **Step 3: 实现图件和报告输出**

实现 `configure_chinese_matplotlib`、`_normalize_display`、`_write_figures`、`_write_report`。图件使用中文标题，报告说明全波形偏移不是 LSRTM，主要用于观察完整炮记录成像响应。

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest D:\ryjin\rtm_acoustic\tests\test_salt_full_waveform_migration.py -q`

Expected: PASS。

### Task 4: 增加命令行入口并运行真实盐丘小范围实验

**Files:**
- Modify: `D:\ryjin\rtm_acoustic\run_salt_full_waveform_migration.py`

- [ ] **Step 1: 实现 CLI**

增加 `parse_args` 和 `main`，支持参数 `--model-path`、`--output-dir`、`--crop-nx`、`--crop-nz`、`--nt`、`--shots`。

- [ ] **Step 2: 运行单元测试**

Run: `python -m pytest D:\ryjin\rtm_acoustic\tests\test_salt_full_waveform_migration.py D:\ryjin\rtm_acoustic\tests\test_small_salt_fwi.py D:\ryjin\rtm_acoustic\tests\test_acoustic_rtm.py D:\ryjin\rtm_acoustic\tests\test_optimize_existing_salt_result.py -q`

Expected: PASS。

- [ ] **Step 3: 运行盐丘小范围实验**

Run: `python D:\ryjin\rtm_acoustic\run_salt_full_waveform_migration.py --output-dir D:\ryjin\rtm_acoustic\outputs\small_salt_full_waveform_migration`

Expected: 输出 `summary.json`、`.npy` 数组、两张 PNG 图和中文 Markdown 总结。

- [ ] **Step 4: 检查结果文件**

Run: `Get-ChildItem D:\ryjin\rtm_acoustic\outputs\small_salt_full_waveform_migration`

Expected: 能看到设计中列出的核心输出文件。

### Task 5: 最终验证和结论

**Files:**
- Read: `D:\ryjin\rtm_acoustic\outputs\small_salt_full_waveform_migration\summary.json`
- Read: `D:\ryjin\rtm_acoustic\outputs\small_salt_full_waveform_migration\full_waveform_migration_summary.md`

- [ ] **Step 1: 汇总指标**

读取 `summary.json`，比较全波形偏移和反射波偏移的 `image_abs_p99`、`normalized_abs_p99`、`illumination_low_fraction`、`image_difference_l2`。

- [ ] **Step 2: 给用户中文结论**

说明本次实现文件、运行命令、测试结果、输出目录和主要指标。明确提醒：本阶段是完整记录 RTM 对比实验，不是 LSRTM。
