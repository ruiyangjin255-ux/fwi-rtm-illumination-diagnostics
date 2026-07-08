# 小范围盐丘 FWI 照明预条件对比 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在已有小范围 SEG/Salt FWI 演示中增加 baseline 与照明预条件 FWI 的对比输出。

**Architecture:** 在 `run_small_salt_fwi.py` 中新增震源照明计算、照明预条件更新和对比运行入口。保持 `run_fwi_demo()` 默认 baseline 行为不变，新增显式 `update_mode` 和 `run_fwi_compare()`，测试覆盖新增函数与 smoke 对比输出。

**Tech Stack:** Python、NumPy、Matplotlib、pytest、现有 `rtm_acoustic.acoustic_rtm` 正演与波场工具。

---

## File Structure

- Modify: `D:\ryjin\rtm_acoustic\run_small_salt_fwi.py`
  - 新增照明场计算、更新方向预条件、对比运行和对比图输出。
- Modify: `D:\ryjin\rtm_acoustic\tests\test_small_salt_fwi.py`
  - 新增照明预条件单元测试和对比 smoke 测试。
- Modify: `D:\ryjin\rtm_acoustic\docs\small_salt_fwi_and_illumination.md`
  - 补充 RTM 图像归一化与 FWI 梯度预条件的差异。

## Task 1: 照明场和预条件函数

- [ ] **Step 1: 写失败测试**

在测试文件中新增：

```python
from rtm_acoustic.run_small_salt_fwi import (
    apply_illumination_preconditioner,
    compute_source_illumination,
)


def test_compute_source_illumination_is_nonnegative(tmp_path):
    nz, nx = 20, 24
    velocity = np.full((nz, nx), 2000.0, dtype=np.float32)
    cfg = RTMConfig(nx=nx, nz=nz, dx=10.0, dz=10.0, dt=0.001, nt=30, f0=12.0, source_x=12, source_z=4, receiver_z=4, absorb_cells=5, fd_order=4)
    source_path = tmp_path / "source.dat"
    from rtm_acoustic.acoustic_rtm import forward_model
    forward_model(velocity, cfg, wavefield_path=source_path)
    illum = compute_source_illumination(source_path, cfg)
    assert illum.shape == velocity.shape
    assert np.isfinite(illum).all()
    assert illum.min() >= 0.0
    assert illum.max() > 0.0


def test_apply_illumination_preconditioner_preserves_shape_and_finiteness():
    update = np.ones((4, 5), dtype=np.float32)
    illumination = np.zeros((4, 5), dtype=np.float32)
    illumination[:, 2:] = 10.0
    conditioned = apply_illumination_preconditioner(update, illumination, epsilon=0.1)
    assert conditioned.shape == update.shape
    assert np.isfinite(conditioned).all()
    assert conditioned.dtype == np.float32
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```powershell
python -m pytest D:\ryjin\rtm_acoustic\tests\test_small_salt_fwi.py::test_compute_source_illumination_is_nonnegative D:\ryjin\rtm_acoustic\tests\test_small_salt_fwi.py::test_apply_illumination_preconditioner_preserves_shape_and_finiteness -v
```

Expected: FAIL，原因是新增函数不存在。

- [ ] **Step 3: 实现新增函数**

实现 `compute_source_illumination()` 和 `apply_illumination_preconditioner()`，照明场用 source wavefield 的时间平方和，预条件先除以归一化照明加 `epsilon`，再按 99 百分位归一化。

- [ ] **Step 4: 运行测试确认通过**

Run:

```powershell
python -m pytest D:\ryjin\rtm_acoustic\tests\test_small_salt_fwi.py -v
```

Expected: PASS。

## Task 2: FWI 对比运行

- [ ] **Step 1: 写失败测试**

新增 smoke 测试：

```python
from rtm_acoustic.run_small_salt_fwi import run_fwi_compare


def test_run_fwi_compare_writes_compare_summary(tmp_path):
    true_model = np.full((30, 36), 2000.0, dtype=np.float32)
    true_model[16:, 12:26] = 2600.0
    cfg = FWIConfig(crop_nx=36, crop_nz=30, nt=40, iterations=1, absorb_cells=6, max_update=15.0)
    summary = run_fwi_compare(true_model=true_model, config=cfg, output_dir=tmp_path, shot_positions=[12, 24], write_figures=False)
    assert (tmp_path / "summary_compare.json").exists()
    assert "baseline" in summary
    assert "illumination_preconditioned" in summary
    assert (tmp_path / "baseline_inverted_model.npy").exists()
    assert (tmp_path / "preconditioned_inverted_model.npy").exists()
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```powershell
python -m pytest D:\ryjin\rtm_acoustic\tests\test_small_salt_fwi.py::test_run_fwi_compare_writes_compare_summary -v
```

Expected: FAIL，原因是 `run_fwi_compare` 不存在。

- [ ] **Step 3: 实现对比运行**

实现 `run_fwi_demo(update_mode=...)`，默认仍为 `baseline`。当 `update_mode="illumination_preconditioned"` 时，对每炮更新方向应用照明预条件。实现 `run_fwi_compare()` 依次运行两种方法，写出 `baseline_summary.json`、`preconditioned_summary.json`、`summary_compare.json` 和两种反演模型。

- [ ] **Step 4: 运行测试确认通过**

Run:

```powershell
python -m pytest D:\ryjin\rtm_acoustic\tests\test_small_salt_fwi.py -v
```

Expected: PASS。

## Task 3: CLI、图件、文档和端到端验证

- [ ] **Step 1: 增加 CLI 参数**

给 `parse_args()` 增加 `--compare-illumination`。默认不启用，启用时输出到 `outputs/small_salt_fwi_illumination_compare` 或用户指定目录。

- [ ] **Step 2: 写对比图**

新增 `fwi_method_compare.png`，包含 baseline 和照明预条件误差曲线、最终模型更新对比。

- [ ] **Step 3: 更新中文文档**

补充“FWI 梯度照明预条件”小节，说明其作用对象是更新方向 `g`，不是 RTM 图像 `I`。

- [ ] **Step 4: 运行完整验证**

Run:

```powershell
python -m pytest D:\ryjin\rtm_acoustic\tests\test_small_salt_fwi.py D:\ryjin\rtm_acoustic\tests\test_acoustic_rtm.py D:\ryjin\rtm_acoustic\tests\test_optimize_existing_salt_result.py -q
python -m rtm_acoustic.run_small_salt_fwi --compare-illumination --iterations 3 --nt 360
```

Expected: 所有测试通过，输出 `summary_compare.json` 和对比图件。
