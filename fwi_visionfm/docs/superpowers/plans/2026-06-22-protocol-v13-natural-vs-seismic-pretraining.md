# Protocol V13 Natural vs Seismic Pretraining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 V12 锁定协议下复用 45 个自然视觉路线 run，并新增 9 个真实 NCS2D frozen run，形成统一 54-run 预训练来源比较。

**Architecture:** V12 reuse gate 负责逐 run 科学配置与 manifest 验证；V13 matrix builder 只复制通过门禁的 run 并建立 M6 矩阵。NCS runner 使用 family/split 共享真实特征缓存，后续统计模块统一读取 54 个 run。

**Tech Stack:** Python、PyTorch、NumPy、Transformers、Matplotlib、pytest。

---

### Task 1: V12 复用门禁与 54-run 矩阵

**Files:**
- Create: `tests/test_protocol_v13_v12_reuse_gate.py`
- Create: `tests/test_protocol_v13_matrix_builder.py`
- Create: `tests/test_protocol_v13_manifest_integrity.py`
- Create: `scripts/verify_v12_reuse_for_v13.py`
- Create: `scripts/build_protocol_v13_matrix.py`
- Create: `configs/protocol_v13_natural_vs_seismic_pretraining.yaml`

- [ ] 写 hash/config/manifest 不一致必须拒绝、矩阵必须为 54 条的失败测试。
- [ ] 运行并确认因 V13 API 缺失而失败。
- [ ] 实现 reuse-critical hash、逐 run 门禁、硬链接复制、V13 manifest 和预注册文件。
- [ ] 运行测试并确认通过。

### Task 2: 真实 NCS2D 合同与训练

**Files:**
- Create: `tests/test_protocol_v13_ncs2d_real_feature_contract.py`
- Create: `scripts/run_protocol_v13_ncs2d_frozen.py`

- [ ] 写真实 cache 接受、fallback cache 拒绝和 optimizer 注册测试。
- [ ] 运行并确认预期失败。
- [ ] 实现共享特征 cache、9 个 decoder run、预测 NPZ 与 metadata。
- [ ] 运行测试并确认通过。

### Task 3: 泛化差距与 paired bootstrap

**Files:**
- Create: `tests/test_protocol_v13_generalization_gap.py`
- Create: `tests/test_protocol_v13_bootstrap.py`
- Create: `scripts/analyze_protocol_v13_generalization_gaps.py`
- Create: `scripts/bootstrap_protocol_v13_pretraining_source.py`

- [ ] 写 SSIM 反向 gap 和 sample_id 配对测试。
- [ ] 运行并确认预期失败。
- [ ] 实现五指标 gap、A-E 比较、多指标 2000 次 bootstrap 与 seed consistency。
- [ ] 运行测试并确认通过。

### Task 4: Summary、中文报告与验收

**Files:**
- Create: `tests/test_protocol_v13_report.py`
- Create: `scripts/summarize_protocol_v13_natural_vs_seismic_pretraining.py`
- Create: `scripts/report_protocol_v13_natural_vs_seismic_pretraining.py`

- [ ] 写报告标题、结论边界、claims 和完整性测试。
- [ ] 运行并确认预期失败。
- [ ] 实现 54-run 汇总、A-E 判级、7 张中文图和协议完整性报告。
- [ ] 运行 9 个 M6 run、统计脚本和全部测试。
- [ ] 核验 54 个输出合同、真实 NCS 标志、hash、target isolation 和限制措辞。

