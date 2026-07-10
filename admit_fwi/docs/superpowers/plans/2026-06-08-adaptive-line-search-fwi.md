# 小范围盐丘 FWI 自适应扩展线搜索 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 增加 baseline 与照明预条件 FWI 的自适应扩展线搜索对比。

**Architecture:** 在 `run_small_salt_fwi.py` 中新增 `run_fwi_adaptive_line_search_demo()` 和 `run_adaptive_line_search_compare()`。复用现有更新方向、照明预条件和误差评估逻辑；当最佳候选位于当前最大步长且仍降低误差时，继续测试扩展步长。

**Tech Stack:** Python、NumPy、Matplotlib、pytest、csv/json。

---

## Task 1: 自适应线搜索单方法运行

- [ ] 写 smoke 测试：小模型，调用 `run_fwi_adaptive_line_search_demo()`，检查 `adaptive_line_search_summary.json`、`adaptive_line_search_results.csv`、`selected_step_scales` 和 `tested_step_scales_by_iteration`。
- [ ] 实现单方法自适应扩展线搜索。
- [ ] 运行小范围 FWI 测试。

## Task 2: baseline 与照明预条件对比

- [ ] 写 smoke 测试：调用 `run_adaptive_line_search_compare()`，检查总摘要和两种方法子目录。
- [ ] 实现 `run_adaptive_line_search_compare()` 和 `adaptive_line_search_compare.png`。
- [ ] 增加 CLI：`--adaptive-line-search`、`--initial-step-scales`、`--expanded-step-scales`。
- [ ] 运行小范围 FWI 测试。

## Task 3: 文档和端到端验证

- [ ] 更新中文说明文档，补充自适应扩展线搜索的目的和判读。
- [ ] 运行完整验证：

```powershell
python -m pytest D:\ryjin\admit_fwi\tests\test_small_salt_fwi.py D:\ryjin\admit_fwi\tests\test_acoustic_rtm.py D:\ryjin\admit_fwi\tests\test_optimize_existing_salt_result.py -q
python -m admit_fwi.run_small_salt_fwi --adaptive-line-search --iterations 3 --nt 360
```

Expected: 测试通过，输出自适应线搜索摘要、CSV 和对比曲线。
