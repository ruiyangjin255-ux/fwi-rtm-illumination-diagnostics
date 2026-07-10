# 小范围盐丘 FWI 自适应步长线搜索 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 增加 baseline 与照明预条件 FWI 的自适应步长线搜索对比。

**Architecture:** 在 `run_small_salt_fwi.py` 中新增 `run_fwi_line_search_demo()` 与 `run_line_search_compare()`。复用现有更新方向和 `_evaluate_total_misfit()`，每次迭代遍历 `step_scale` 候选，选择误差最低的候选模型。

**Tech Stack:** Python、NumPy、Matplotlib、pytest、csv/json。

---

## Task 1: 参数解析

- [ ] 测试 `parse_float_values("0.25,0.5,1.0", name="step_scale")`。
- [ ] 复用已实现 `parse_float_values()`，无需新增生产代码。
- [ ] 运行小范围 FWI 测试。

## Task 2: 线搜索单方法运行

- [ ] 写 smoke 测试：小模型、2 个 step_scale，调用 `run_fwi_line_search_demo()`，检查 `line_search_summary.json`、`line_search_results.csv` 和 `selected_step_scales`。
- [ ] 实现 `run_fwi_line_search_demo()`：每轮构造平均更新方向，测试候选步长，选择误差最低且不高于当前误差的模型。
- [ ] 运行小范围 FWI 测试。

## Task 3: baseline 与照明预条件线搜索对比

- [ ] 写 smoke 测试：调用 `run_line_search_compare()`，检查 `line_search_summary.json` 和两种方法摘要。
- [ ] 实现 `run_line_search_compare()` 和 `line_search_misfit_compare.png`。
- [ ] 增加 CLI：`--line-search`、`--step-scales`。
- [ ] 更新中文说明文档。
- [ ] 运行完整验证：

```powershell
python -m pytest D:\ryjin\admit_fwi\tests\test_small_salt_fwi.py D:\ryjin\admit_fwi\tests\test_acoustic_rtm.py D:\ryjin\admit_fwi\tests\test_optimize_existing_salt_result.py -q
python -m admit_fwi.run_small_salt_fwi --line-search --iterations 3 --nt 360
```

Expected: 测试通过，输出线搜索摘要、CSV 和误差曲线图。
