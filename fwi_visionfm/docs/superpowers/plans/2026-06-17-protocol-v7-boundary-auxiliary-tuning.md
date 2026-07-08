# Protocol V7 Boundary Auxiliary Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不扩大 backbone 和数据范围的前提下，完成 Protocol V7 boundary auxiliary 的小范围 lambda / boundary_method 调参，复用已有 baseline 与 lambda=0.10 结果，生成 tuning summary 和 report。

**Architecture:** 新建一个 tuning runner，底层直接复用 `run_protocol_v7_boundary_auxiliary_smoke.py` 与 `run_protocol_v7_boundary_auxiliary_seed_stability.py` 的训练、评估、npz 写出和绘图函数。报告脚本从 tuning summary 中提取 lambda 对比和 method 对比，同时记录复用来源与限制口径。

**Tech Stack:** Python, NumPy, CSV/JSON, pytest, existing fwi_visionfm V7 training/evaluation pipeline

---

### Task 1: 先写 tuning 测试

**Files:**
- Create: `D:/ryjin/fwi_visionfm/tests/test_protocol_v7_boundary_tuning.py`
- Create: `D:/ryjin/fwi_visionfm/tests/test_protocol_v7_boundary_tuning_report.py`

- [ ] **Step 1: 写失败测试，覆盖 reused_from、thresholded_gradient 和 summary 结构**
- [ ] **Step 2: 写失败测试，覆盖 report 的 reused results、not benchmark-level proof 和禁用结论**
- [ ] **Step 3: 运行 pytest，确认因实现缺失而失败**

### Task 2: 实现 tuning runner 与 summary

**Files:**
- Create: `D:/ryjin/fwi_visionfm/scripts/run_protocol_v7_boundary_auxiliary_tuning.py`
- Modify: `D:/ryjin/fwi_visionfm/data/boundary_targets.py`

- [ ] **Step 1: 若缺失则补 `thresholded_gradient` boundary target 方法**
- [ ] **Step 2: 实现 tuning run 配置枚举，区分新增运行与复用运行**
- [ ] **Step 3: 复用 baseline seed=0/1/2 与 lambda=0.10 + gradient_magnitude seed=0/1/2**
- [ ] **Step 4: 运行 lambda=0.03/0.05 的 seed=0/1/2，以及 sobel/thresholded_gradient 的 seed=0**
- [ ] **Step 5: 生成 `protocol_v7_boundary_auxiliary_tuning_summary.csv`，包含 `reused_from` 与 `threshold`**

### Task 3: 实现 tuning report

**Files:**
- Create: `D:/ryjin/fwi_visionfm/scripts/report_protocol_v7_boundary_auxiliary_tuning.py`

- [ ] **Step 1: 汇总 Reused Results**
- [ ] **Step 2: 生成 Lambda Tuning 与 Boundary Method Tuning 分析**
- [ ] **Step 3: 输出 Recommended Setting、Limitations、Next Step**
- [ ] **Step 4: 保持 not benchmark-level proof，避免禁用短语**

### Task 4: 验证并跑真实 tuning

**Files:**
- Verify outputs under: `D:/ryjin/fwi_visionfm/outputs/protocol_v7_boundary_auxiliary_tuning`

- [ ] **Step 1: 运行新增 pytest**
- [ ] **Step 2: 执行 CPU tuning runner**
- [ ] **Step 3: 生成 report 并核对复用行与新增行**
- [ ] **Step 4: 汇总哪个 lambda 更稳、哪个 method 值得进入后续 seed=1/2**
