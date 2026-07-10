# Direct Mute Illumination Scan Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在小范围盐丘全波形偏移中增加直达波静音参数扫描，并比较原始、照明归一化和 Laplacian 成像指标。

**Architecture:** 扩展 `run_salt_full_waveform_migration.py`，新增扫描函数和 CLI 分支。核心 RTM 算子不改动，通过 `record_provider` 对完整记录施加 `mute_direct_arrivals` 后再进入已有 `multishot_reverse_time_migrate`。

**Tech Stack:** Python、NumPy、Matplotlib、CSV、pytest、现有 `admit_fwi.acoustic_rtm`。

---

### Task 1: 增加参数解析和扫描测试

**Files:**
- Modify: `D:\ryjin\admit_fwi\run_salt_full_waveform_migration.py`
- Modify: `D:\ryjin\admit_fwi\tests\test_salt_full_waveform_migration.py`

- [ ] 增加测试 `test_parse_float_values_rejects_nonpositive_padding`，验证扫描参数解析。
- [ ] 实现 `parse_float_values(raw, name, allow_zero=False)`，支持 `padding_time` 允许 0，`taper_time` 必须为正。

### Task 2: 增加静音扫描核心函数

**Files:**
- Modify: `D:\ryjin\admit_fwi\run_salt_full_waveform_migration.py`
- Modify: `D:\ryjin\admit_fwi\tests\test_salt_full_waveform_migration.py`

- [ ] 增加测试 `test_mute_scan_writes_metrics_and_report`，使用小模型、短 `nt`、两组参数快速运行。
- [ ] 实现 `run_direct_mute_scan_demo`，对每个参数组合调用 `multishot_reverse_time_migrate`，用 `record_provider` 调用 `mute_direct_arrivals`。
- [ ] 保存每组结果的 `.npy` 数组、`mute_scan_metrics.csv`、`mute_scan_summary.json`。

### Task 3: 增加图件、中文报告和 CLI

**Files:**
- Modify: `D:\ryjin\admit_fwi\run_salt_full_waveform_migration.py`

- [ ] 实现 `mute_scan_compare.png`，展示未静音、最佳静音、反射波参考三类图像。
- [ ] 实现 `mute_scan_best_summary.md`，用中文解释直达波静音、照明归一化、Laplacian 的差异。
- [ ] CLI 增加 `--mute-scan`、`--padding-times`、`--taper-times`。

### Task 4: 验证和真实盐丘实验

**Files:**
- Read: `D:\ryjin\admit_fwi\outputs\small_salt_full_waveform_mute_scan\mute_scan_summary.json`

- [ ] 运行完整相关测试。
- [ ] 运行默认盐丘小范围静音扫描。
- [ ] 汇总最佳参数、浅部能量比例、成像差异和报告路径。
