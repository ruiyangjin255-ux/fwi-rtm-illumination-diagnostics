# 照明预条件 FWI 二维参数扫描 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 增加 `epsilon × max_update` 二维扫描，判断照明预条件 FWI 是否可通过步长约束调节超过 baseline。

**Architecture:** 在 `run_small_salt_fwi.py` 复用 `run_fwi_demo(update_mode="illumination_preconditioned")`，新增 `parse_float_values()`、`run_illumination_2d_scan()` 和热力图输出。CLI 新增 `--scan-illumination-2d`、`--max-updates`。

**Tech Stack:** Python、NumPy、Matplotlib、pytest、csv/json。

---

## Task 1: 参数解析

- [ ] 测试 `parse_float_values("20,35,80", name="max_update") == [20.0, 35.0, 80.0]`。
- [ ] 测试非正数抛出 `ValueError`。
- [ ] 实现通用正浮点列表解析，保留 `parse_epsilon_values()` 包装。
- [ ] 运行小范围 FWI 测试。

## Task 2: 二维扫描

- [ ] 写 smoke 测试：小模型、2 个 epsilon、2 个 max_update，调用 `run_illumination_2d_scan()`。
- [ ] 检查 `summary_2d_scan.json`、`scan_2d_results.csv` 和 `best_preconditioned`。
- [ ] 实现二维扫描：先跑 baseline，再逐组复制 `FWIConfig(max_update=...)` 跑预条件。
- [ ] 输出 CSV 和 JSON。
- [ ] 运行小范围 FWI 测试。

## Task 3: CLI、热力图和文档

- [ ] 新增 CLI：`--scan-illumination-2d`、`--max-updates`。
- [ ] 新增 `epsilon_update_heatmap.png` 与 `best_preconditioned_misfit_curve.png`。
- [ ] 更新中文说明文档，补充二维扫描结论判读。
- [ ] 运行完整验证：

```powershell
python -m pytest D:\ryjin\rtm_acoustic\tests\test_small_salt_fwi.py D:\ryjin\rtm_acoustic\tests\test_acoustic_rtm.py D:\ryjin\rtm_acoustic\tests\test_optimize_existing_salt_result.py -q
python -m rtm_acoustic.run_small_salt_fwi --scan-illumination-2d --iterations 3 --nt 360
```

Expected: 测试通过，输出二维扫描 JSON、CSV 和热力图。
