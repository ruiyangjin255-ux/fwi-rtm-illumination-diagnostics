# ECG Reliability Gate Execution Report

## 1. 执行依据

本轮根据 `D:/新建文件夹/FWI-RTM Update Framework.docx` 的研究思路执行，目标不是继续叠加经验性阻尼，而是把现有 `smooth_alpha0.3_thr0.5` 照明门控基线升级为可复现、可解释、可审计的多证据可信更新门控框架。

核心要求如下：

- 保留现有照明门控作为 baseline，不把它包装成最终创新点。
- 新增 ECG reliability field：illumination proxy、cross-shot gradient consensus、descent alignment 三类证据共同约束 FWI update。
- 所有 gate 对照必须 matched update budget，证明改进不是单纯来自减小更新幅度。
- 增加 held-out shot audit 与 RTM split consistency，避免只用真值挑图。
- 缺少必要诊断量时必须 fail fast，禁止用 dummy diagnostics 或 fallback wavefield diagnostics 生成论文图。
- 当前 full 224-shot / 4-fold 正式实验不得自动运行；先完成 tests 与 smoke。

## 2. 已完成的程序框架

新增诊断模块：

- `rtm_acoustic/diagnostics/update_reliability.py`
  - 计算 illumination score、cross-shot gradient consensus、descent alignment 与 ECG reliability score。
  - 支持 coverage support、Gaussian smoothing、alpha bound、matched update budget。
  - 输出 config hash 与 array hash，用于后续结果复现。
- `rtm_acoustic/diagnostics/shot_partition.py`
  - 实现 interleaved shot groups 与 audit fold split。
  - 提供 audit isolation 检查，防止 hold-out shots 泄漏。
- `rtm_acoustic/diagnostics/matched_budget.py`
  - 将不同 gate 的 update L2 norm 对齐到同一 target budget。
  - budget 不可达时抛出 `BUDGET_MATCH_FAILED`。
- `rtm_acoustic/diagnostics/gate_ablation.py`
  - 构建 matched gate suite，包括 global matched、illumination-only、gradient-consensus-only、ECG reliability gate、inverse illumination negative control、depth-matched、random depth-matched masks。
- `rtm_acoustic/diagnostics/heldout_audit.py`
  - 提供 truth-free audit 指标：normalized L2 residual、NRMS、trace correlation、envelope error。
- `rtm_acoustic/diagnostics/rtm_split_consistency.py`
  - 提供 RTM split consistency 指标：image correlation、simple SSIM、local structure tensor coherence。

新增配置与脚本：

- `rtm_acoustic/configs/salt_reliability_gate_v1.yaml`
- `rtm_acoustic/scripts/replay_fwi_diagnostics.py`
- `rtm_acoustic/scripts/run_reliability_gate_ablation.py`
- `rtm_acoustic/scripts/run_holdout_gate_audit.py`
- `rtm_acoustic/scripts/report_reliability_gate.py`
- `rtm_acoustic/scripts/_common.py`

已修改正式 FWI runner：

- `rtm_acoustic/run_full_salt_fwi.py`
  - 新增 `--save-iteration-diagnostics`。
  - 新增 `--num-shot-groups` 与 `--shot-group-mode interleaved`。
  - 新增 `--diagnostics-dir`。
  - 每轮 FWI 按 shot group 写出 `gradient_group_XX.npy`。
  - 写出 `source_adjoint_energy_proxy.npy`、`delta_model.npy`、`aggregate_gradient.npy`、`average_update.npy`、`search_direction.npy` 与 `step_length.json`。
  - 默认不开启诊断保存，因此旧命令行为不变。

新增测试：

- `rtm_acoustic/tests/test_update_reliability.py`
- `rtm_acoustic/tests/test_matched_budget.py`
- `rtm_acoustic/tests/test_holdout_isolation.py`
- `rtm_acoustic/tests/test_reliability_outputs.py`

## 3. 当前 smoke 结果

已运行测试：

```powershell
python -m pytest -q rtm_acoustic\tests\test_update_reliability.py rtm_acoustic\tests\test_matched_budget.py rtm_acoustic\tests\test_holdout_isolation.py rtm_acoustic\tests\test_reliability_outputs.py
```

结果：

```text
15 passed in 4.89s
```

已运行 smoke：

```powershell
python rtm_acoustic\scripts\replay_fwi_diagnostics.py --config rtm_acoustic\configs\salt_reliability_gate_v1.yaml --smoke
```

结果为预期阻塞：

```text
BLOCKED_MISSING_FWI_DIAGNOSTICS
```

已写出 manifest：

- `rtm_acoustic/outputs/salt_reliability_gate_v1/manifest.json`

已运行 hold-out split smoke：

```powershell
python rtm_acoustic\scripts\run_holdout_gate_audit.py --config rtm_acoustic\configs\salt_reliability_gate_v1.yaml --all-audit-folds --smoke
```

已写出：

- `rtm_acoustic/outputs/salt_reliability_gate_v1/audit/heldout_audit_manifest.json`

已生成 smoke report：

- `rtm_acoustic/outputs/salt_reliability_gate_v1/report/reliability_gate_smoke_report.md`

已运行极小 FWI diagnostics smoke：

```powershell
python rtm_acoustic\run_full_salt_fwi.py --model "D:\ryjin\波场数值模拟\声波场\fd2d_pml\vel\seg676x230.bin" --output-dir rtm_acoustic\outputs\diagnostics_smoke --iterations 1 --nt 20 --max-shots 2 --shot-spacing 300 --optimizer steepest --no-figures --save-iteration-diagnostics --num-shot-groups 2 --diagnostics-dir rtm_acoustic\outputs\diagnostics_smoke_ecg
```

该 smoke 已验证 FWI runner 能写出：

- `gradient_group_00.npy`
- `gradient_group_01.npy`
- `source_adjoint_energy_proxy.npy`
- `delta_model.npy`
- `step_length.json`
- `diagnostics_manifest.json`

该结果只用于程序 smoke，不作为论文结果。

## 4. 当前阻塞原因

当前已有正式 FWI output 只包含 total gradient checkpoint 与 model checkpoint，不能支撑 ECG reliability gate 的正式计算。新的 FWI runner 已具备诊断保存能力，但尚未用正式长记录、多炮配置重新运行。

缺失的必要诊断量如下：

- `rtm_acoustic/outputs/salt_reliability_gate_v1/diagnostics/gradient_group_00.npy`
- `rtm_acoustic/outputs/salt_reliability_gate_v1/diagnostics/gradient_group_01.npy`
- `rtm_acoustic/outputs/salt_reliability_gate_v1/diagnostics/gradient_group_02.npy`
- `rtm_acoustic/outputs/salt_reliability_gate_v1/diagnostics/gradient_group_03.npy`
- `rtm_acoustic/outputs/salt_reliability_gate_v1/diagnostics/source_adjoint_energy_proxy.npy`
- `rtm_acoustic/outputs/salt_reliability_gate_v1/diagnostics/delta_model.npy`
- `rtm_acoustic/outputs/salt_reliability_gate_v1/diagnostics/step_length.json`

这些诊断量不能从现有 total gradient 反推：

- cross-shot gradient consensus 必须来自按 shot group 分组累积的 gradients。
- source-adjoint energy proxy 必须来自正传/伴随场能量或等价照明代理。
- step length 必须来自当轮 FWI update 记录，用于解释 `delta_model` 与 gate scaling。

因此当前正确状态是阻塞，而不是生成看似完整但不可发表的结果。

## 5. 下一步必须执行的正式实验

正式结果生成前，需要用新增参数重新运行 FWI 以补采诊断量：

1. 用正式 shot 数、nt、optimizer 和初始模型重新运行 `run_full_salt_fwi.py`。
2. 开启 `--save-iteration-diagnostics`，写入 `rtm_acoustic/outputs/salt_reliability_gate_v1/diagnostics`。
3. 运行 `replay_fwi_diagnostics.py` 生成 ECG components 与 preferred gate。
4. 运行 `run_reliability_gate_ablation.py` 生成 matched controls。
5. 再执行 held-out audit、RTM split consistency 和论文 Figure 生成。

注意：当前 `source_adjoint_energy_proxy.npy` 使用 source-wavefield energy proxy，manifest 中已明确标注代理类型。若后续要进一步增强严谨性，可在 `compute_update_direction` 中额外返回 adjoint wavefield energy，从 source-only proxy 升级为 source-adjoint product proxy。

## 6. 正式实验命令草案

以下是正式命令草案。当前没有自动运行 full 224-shot / 4-fold 实验，因为该实验耗时且文档要求先完成 tests 与 smoke。

```powershell
python rtm_acoustic\run_full_salt_fwi.py `
  --save-iteration-diagnostics `
  --num-shot-groups 4 `
  --shot-group-mode interleaved `
  --diagnostics-dir rtm_acoustic\outputs\salt_reliability_gate_v1\diagnostics
```

诊断量齐备后再运行：

```powershell
python rtm_acoustic\scripts\replay_fwi_diagnostics.py --config rtm_acoustic\configs\salt_reliability_gate_v1.yaml
python rtm_acoustic\scripts\run_reliability_gate_ablation.py --config rtm_acoustic\configs\salt_reliability_gate_v1.yaml
python rtm_acoustic\scripts\run_holdout_gate_audit.py --config rtm_acoustic\configs\salt_reliability_gate_v1.yaml --all-audit-folds
python rtm_acoustic\scripts\report_reliability_gate.py --config rtm_acoustic\configs\salt_reliability_gate_v1.yaml
```

## 7. 对论文创新点的当前支撑状态

当前已具备可写入 Methods 的框架基础：

- 从 single-evidence illumination gate 升级为 multi-evidence ECG reliability gate。
- 引入 cross-shot gradient consensus，约束不稳定 shot-dependent update。
- 引入 descent alignment，避免 gate 接受局部能量强但方向不可信的更新。
- 用 matched update budget 设计 ablation，排除“只是更新小所以更稳”的解释。
- 用 hold-out audit 与 RTM split consistency 做 truth-free evidence，弱化只依赖真值图像的论文风险。

当前还不具备可写入 Results 的正式 Figure 结果：

- ECG gate 尚未从真实 per-shot diagnostics 计算。
- 对照组尚未基于同一 update budget 生成图像。
- before/after FWI 多炮 RTM 与 held-out audit 指标尚未接入论文 Figure。

结论：本轮完成的是“可发表方向的程序框架与 fail-fast 审计基础”，不是最终发表级成图。下一步必须优先补强 FWI runner 的诊断量保存，然后再生成 SCI 论文图与结果段。
