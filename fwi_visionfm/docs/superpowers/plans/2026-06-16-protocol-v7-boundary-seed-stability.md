# Protocol V7 Boundary Seed Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 补齐 Protocol V7 boundary auxiliary 的 seed=1/2 公平对照，复用既有 seed=0 结果，生成稳定性 summary/report，并优化 V7 成图网格使之更接近速度模型常见画法。

**Architecture:** 复用 `run_protocol_v7_boundary_auxiliary_smoke.py` 的真实训练与评估函数，新增一个 seed-stability runner 负责运行 baseline/boundary 两组配置、续跑跳过和复用 seed=0。报告脚本单独读取新 summary 并统计 seed 胜出次数，同时把预测网格改成统一色标、带 colorbar 风格信息的速度模型画法。

**Tech Stack:** Python, NumPy, CSV/JSON, Pillow, pytest, existing fwi_visionfm training/evaluation pipeline

---

### Task 1: 先补 seed-stability 测试

**Files:**
- Create: `D:/ryjin/fwi_visionfm/tests/test_protocol_v7_boundary_seed_stability.py`
- Create: `D:/ryjin/fwi_visionfm/tests/test_protocol_v7_boundary_seed_stability_report.py`

- [ ] **Step 1: 写失败测试，覆盖 summary 合并与 win count**
- [ ] **Step 2: 运行新增测试，确认因缺少实现而失败**

### Task 2: 实现 seed-stability runner 与 summary

**Files:**
- Create: `D:/ryjin/fwi_visionfm/scripts/run_protocol_v7_boundary_auxiliary_seed_stability.py`
- Modify: `D:/ryjin/fwi_visionfm/scripts/run_protocol_v7_boundary_auxiliary_smoke.py`

- [ ] **Step 1: 抽取可复用的 V7 单 run 执行和绘图能力**
- [ ] **Step 2: 新增 seed-stability runner，仅跑 seed=1/2 baseline 与 selected boundary**
- [ ] **Step 3: 支持复用 `outputs/protocol_v7_boundary_auxiliary_smoke` 的 seed=0 summary 行**
- [ ] **Step 4: 缺失 run 标记 `SKIPPED` 或 `FAILED`，不中断汇总**

### Task 3: 实现报告与成图优化

**Files:**
- Create: `D:/ryjin/fwi_visionfm/scripts/report_protocol_v7_boundary_auxiliary_seed_stability.py`
- Modify: `D:/ryjin/fwi_visionfm/scripts/run_protocol_v4_integrated_visual_search.py`

- [ ] **Step 1: 报告输出 Seed Stability Table、Win Counts、Interpretation、Limitations、Next Step**
- [ ] **Step 2: 明确保留 `not benchmark-level proof` 且避免越界结论**
- [ ] **Step 3: 优化预测/梯度 grid 的速度模型画法，统一 pred/target/error 的布局与显示强度**

### Task 4: 验证并跑真实 seed=1/2

**Files:**
- Verify outputs under: `D:/ryjin/fwi_visionfm/outputs/protocol_v7_boundary_auxiliary_seed_stability`

- [ ] **Step 1: 运行新增 pytest**
- [ ] **Step 2: 执行 seed=1/2 CPU 真实运行**
- [ ] **Step 3: 生成并检查 summary/report**
- [ ] **Step 4: 汇总胜出次数、多数支持性与数值 trade-off**
