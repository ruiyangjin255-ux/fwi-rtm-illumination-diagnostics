# Protocol V12 Spectrogram DINOv2 Confirmation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在锁定 200/50/50 CPU 协议下完成五方法、三 transfer、三 seed 的频谱 DINOv2 确认性评测。

**Architecture:** 复用 V11 模型与训练内核，通过 V12 专用 manifest、矩阵、runner、bootstrap、summary 和 report 薄层增加确认性协议约束。所有训练由 manifest hash、config hash 与 optimizer registration 三重门禁保护。

**Tech Stack:** Python、PyTorch、NumPy、PyYAML、Matplotlib、pytest。

---

### Task 1: 锁定 manifest 与矩阵

**Files:**
- Create: `tests/test_protocol_v12_manifest_integrity.py`
- Create: `tests/test_protocol_v12_matrix_builder.py`
- Create: `scripts/build_protocol_v12_manifests.py`
- Create: `scripts/build_protocol_v12_matrix.py`
- Create: `configs/protocol_v12_spectrogram_dinov2_confirmation.yaml`

- [ ] 写 manifest hash、split 不重叠、45 条矩阵的失败测试。
- [ ] 运行测试并确认因 V12 API 缺失而失败。
- [ ] 实现固定 CSV、哈希报告、五方法矩阵和预注册输出。
- [ ] 运行测试并确认通过。

### Task 2: Runner 公平性与隔离

**Files:**
- Create: `tests/test_protocol_v12_optimizer_registration.py`
- Create: `tests/test_protocol_v12_target_isolation.py`
- Create: `scripts/run_protocol_v12_spectrogram_dinov2_confirmation.py`

- [ ] 写 decoder 参数必须属于 optimizer、target test 不得进入训练、预测契约的失败测试。
- [ ] 运行测试并确认预期失败。
- [ ] 实现 V11 内核复用、lazy decoder 初始化、参数统计、config/manifest hash 和 resume 门禁。
- [ ] 运行测试并确认通过。

### Task 3: 多指标配对 bootstrap

**Files:**
- Create: `tests/test_protocol_v12_bootstrap.py`
- Create: `scripts/bootstrap_protocol_v12_comparisons.py`

- [ ] 写乱序 sample_id 的四指标配对差值测试。
- [ ] 运行测试并确认预期失败。
- [ ] 实现 2000 次配对 bootstrap、90 个比较与 seed consistency 输出。
- [ ] 运行测试并确认通过。

### Task 4: Summary 与中文报告

**Files:**
- Create: `tests/test_protocol_v12_report.py`
- Create: `scripts/summarize_protocol_v12_spectrogram_dinov2_confirmation.py`
- Create: `scripts/report_protocol_v12_spectrogram_dinov2_confirmation.py`

- [ ] 写报告标题、限制措辞、claims 和逐 transfer 结果的失败测试。
- [ ] 运行测试并确认预期失败。
- [ ] 实现五组预注册证据判断、泛化差距、完整性报告和 8 张中文图。
- [ ] 运行全部 V12 测试并确认通过。

### Task 5: 两阶段真实训练与验收

**Files:**
- Generate: `outputs/protocol_v12_spectrogram_dinov2_confirmation/**`

- [ ] 生成并锁定 manifest、availability、矩阵和预注册文件。
- [ ] 执行 Stage A 15 run，核验协议但不调参。
- [ ] 执行 Stage B，复用 seed 0 并完成 45 run。
- [ ] 重算 2000 次 bootstrap、summary、report 和图表。
- [ ] 运行最终测试、文件契约、hash、target 隔离和结论措辞扫描。

