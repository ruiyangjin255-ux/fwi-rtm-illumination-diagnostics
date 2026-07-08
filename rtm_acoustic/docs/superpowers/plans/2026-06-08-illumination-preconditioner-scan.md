# 照明预条件 FWI 参数扫描 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 增加照明预条件 `epsilon` 参数扫描，输出 baseline 与多组预条件 FWI 的误差对比。

**Architecture:** 在 `run_small_salt_fwi.py` 中新增 `parse_epsilon_values()`、`run_illumination_scan()` 和扫描图/CSV 输出。复用现有 `run_fwi_demo(update_mode="illumination_preconditioned")`，只给它增加显式 `preconditioner_epsilon` 参数，默认行为保持不变。

**Tech Stack:** Python、NumPy、Matplotlib、pytest、标准库 csv/json。

---

## Task 1: 参数解析

- [ ] 写测试：`parse_epsilon_values("0.01,0.05,0.2") == [0.01, 0.05, 0.2]`，非法或非正数抛出 `ValueError`。
- [ ] 运行测试确认失败。
- [ ] 实现 `parse_epsilon_values()`。
- [ ] 运行 `test_small_salt_fwi.py` 确认通过。

## Task 2: 扫描运行

- [ ] 写 smoke 测试：小模型、两组 epsilon，调用 `run_illumination_scan()`，检查 `summary_scan.json`、`scan_results.csv`、baseline 和预条件结果存在。
- [ ] 运行测试确认失败。
- [ ] 实现 `run_illumination_scan()`：先跑 baseline，再按 epsilon 跑照明预条件，记录最优结果。
- [ ] 给 `run_fwi_demo()` 增加 `preconditioner_epsilon` 参数，并传给 `apply_illumination_preconditioner()`。
- [ ] 运行测试确认通过。

## Task 3: CLI、图件和文档

- [ ] 给 CLI 增加 `--scan-illumination` 和 `--illumination-epsilons`。
- [ ] 新增 `illumination_scan_compare.png`，展示 baseline 和各 epsilon 的误差曲线。
- [ ] 更新中文说明文档，补充扫描结论判读。
- [ ] 运行完整验证：

```powershell
python -m pytest D:\ryjin\rtm_acoustic\tests\test_small_salt_fwi.py D:\ryjin\rtm_acoustic\tests\test_acoustic_rtm.py D:\ryjin\rtm_acoustic\tests\test_optimize_existing_salt_result.py -q
python -m rtm_acoustic.run_small_salt_fwi --scan-illumination --iterations 3 --nt 360
```

Expected: 测试通过，输出扫描摘要、CSV 和图件。
