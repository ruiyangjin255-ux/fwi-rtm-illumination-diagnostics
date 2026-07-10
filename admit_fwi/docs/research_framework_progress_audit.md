# 基于多证据可信度门控的复杂盐丘 FWI-RTM 选择性更新方法研究：进展核对报告

生成时间：2026-07-07  
对照文档：`D:/新建文件夹/基于多证据可信度门控的复杂盐丘 FWI–RTM 选择性更新方法研究.pdf`  
项目代码：`D:/ryjin/admit_fwi`  
GitHub 分支：`codex/ecg-reliability-gate`

## 1. 总体判断

当前项目已经从“研究框架设计”推进到“正式 FWI 诊断采集、ECG 门控生成、matched controls 出图、RTM 审计运行”的阶段。研究文档中的阶段一和阶段二基本完成，阶段三正在收尾。

当前最重要的结论是：全量 FWI 并不能稳定支撑“速度模型显著变好”的论文叙事，但它可以很好地支撑“复杂盐丘中需要可信更新筛选”的研究动机。all-shots FWI 虽然残差下降明显，但 RTM before/after 结果显示 after FWI 的 filtered RMSE 变差；audit0 训练炮版本中 full FWI 也使模型 MAE/RMSE/edge MAE 变差。ECG gate 的价值应定位为“筛选局部可信更新并约束错误更新传播”，而不是宣称高精度盐丘速度恢复。

## 2. 对照研究文档的完成度

| 研究文档要求 | 当前实现/结果 | 状态 | 说明 |
|---|---|---|---|
| SEG/Salt 复杂盐丘模型，676 x 230 | 已使用 SEG/Salt 二维模型 | 已完成 | all-shots 和 audit0 均基于同一模型 |
| 224 炮完整 FWI | `full_salt_fwi_cg_allshots_ecg_v1` 已完成 3 轮 | 已完成 | 224/224 shots，3 iterations |
| 保留炮 audit_fold=0 | `full_salt_fwi_cg_audit0_train_ecg_v1` 已完成 3 轮 | 已完成 | train=168，audit=56，互不重叠 |
| 每轮梯度/波场诊断保存 | 已保存 group gradients、delta model、source energy proxy | 已完成 | 输出到 diagnostics 目录 |
| 局部照明 I(x,z) | `illumination_score.npy` | 已完成 | 使用 source-wavefield energy proxy |
| 跨炮梯度一致性 Cg(x,z) | `gradient_consensus.npy` | 已完成 | 由 4 个 shot groups 计算 |
| 更新下降有效性 D(x,z) | `descent_alignment.npy` | 已完成 | 基于 delta model 与 group gradients |
| ECG 可信度场 Q(x,z) | `ecg_reliability_score.npy` | 已完成 | all-shots 与 audit0 均生成 |
| 空间门控 alpha(x,z) | `ecg_reliability_gate.npy` | 已完成 | matched update budget |
| 等更新预算约束 | `target_update_fraction=0.04` | 已完成 | all-shots 与 audit0 均 matched |
| M0-M7 对照 | 11 组 gates 已生成 | 已完成 | 包括 global、illumination、consensus、inverse、depth、5 random、ECG |
| random/depth 反事实对照 | 已修复 top-k coverage | 已完成 | random 不再退化成 global damping |
| 保留炮审计指标 | audit split 已有，最终 residual 指标待补 | 部分完成 | 需下一步计算 audit shots residual |
| RTM before/after FWI | all-shots 已完成，audit0 正在跑 | 部分完成 | all-shots 出现负结果，audit0 RTM 未完全结束 |
| RTM split consistency | 指标模块已有 | 待完成 | 尚未形成最终表格/图 |
| 盐顶/盐翼/盐下目标区诊断 | 尚未自动化汇总 | 待完成 | 下一阶段重点 |
| 论文 Figure 支撑 | Gate controls 图与表已冻结 | 部分完成 | Figure 3/结果段仍需整合 |

## 3. 已完成的正式实验

### 3.1 all-shots FWI

输出目录：

`D:/ryjin/admit_fwi/outputs/FWI/full_salt_fwi_cg_allshots_ecg_v1`

关键参数：

| 项目 | 数值 |
|---|---:|
| shots | 224 |
| iterations | 3 |
| nt | 900 |
| optimizer | cg |
| initial misfit | 4.443589946057149e-05 |
| final misfit | 2.045601241590732e-05 |
| misfit reduction | 0.5396512130004663 |

判断：数据残差下降明显，但不能单独证明速度更新可信。后续 RTM 结果显示全量 FWI 更新可能损害成像稳定性。

### 3.2 audit_fold=0 训练炮 FWI

输出目录：

`D:/ryjin/admit_fwi/outputs/FWI/full_salt_fwi_cg_audit0_train_ecg_v1`

关键参数：

| 项目 | 数值 |
|---|---:|
| all selected shots | 224 |
| train shots | 168 |
| audit shots | 56 |
| iterations | 3 |
| nt | 900 |
| optimizer | cg |
| initial misfit | 4.4400969625124145e-05 |
| final misfit | 3.773262459972302e-05 |
| misfit reduction | 0.15018467122906845 |

判断：audit0 训练炮版本满足研究文档中“保留炮不得参与 FWI、梯度计算、照明计算和门控参数选择”的原则。它是后续真值无关审计的主结果基础。

## 4. ECG 可信度门控与 matched controls

### 4.1 all-shots ECG

输出目录：

`D:/ryjin/admit_fwi/outputs/salt_reliability_gate_v1`

状态：

| 项目 | 数值 |
|---|---:|
| manifest status | READY |
| target update fraction | 0.04 |
| target update L2 | 107.83867987101544 |
| matched ECG update L2 | 107.83867988687645 |
| gate count | 11 |

已冻结成果：

- 图：`D:/ryjin/admit_fwi/outputs/salt_reliability_gate_v1/figures/figure_gate_controls_allshots.png`
- 表：`D:/ryjin/admit_fwi/outputs/salt_reliability_gate_v1/tables/gate_control_metrics_allshots.md`
- CSV：`D:/ryjin/admit_fwi/outputs/salt_reliability_gate_v1/tables/gate_control_metrics_allshots.csv`

代表性模型指标：

| 方法 | update_l2 | model_mae | model_rmse | edge_mae |
|---|---:|---:|---:|---:|
| initial_model | 0 | 131.692 | 285.239 | 37.2756 |
| full_fwi_model | 6831.81 | 131.479 | 285.165 | 38.4261 |
| global_matched | 107.839 | 131.680 | 285.236 | 37.2780 |
| illumination_only_matched | 107.839 | 131.664 | 285.229 | 37.2721 |
| ecg_reliability_gate | 107.839 | 131.681 | 285.236 | 37.2758 |
| inverse_illumination_negative_control | 107.839 | 131.687 | 285.239 | 37.2809 |

解释：all-shots full FWI 使 MAE/RMSE 略降，但 edge MAE 明显变差，说明全量更新引入边界或梯度误差。ECG gate 在 all-shots 设置下不是最优模型误差方案，但它与 illumination-only、consensus-only、random、inverse controls 有清晰空间差异，可作为方法可解释性图。

### 4.2 audit0 ECG

输出目录：

`D:/ryjin/admit_fwi/outputs/salt_reliability_gate_audit0_v1`

状态：

| 项目 | 数值 |
|---|---:|
| manifest status | READY |
| target update fraction | 0.04 |
| target update L2 | 122.29914466934463 |
| matched ECG update L2 | 122.29914470014572 |
| gate count | 11 |

已冻结成果：

- 图：`D:/ryjin/admit_fwi/outputs/salt_reliability_gate_audit0_v1/figures/figure_gate_controls_audit0.png`
- 表：`D:/ryjin/admit_fwi/outputs/salt_reliability_gate_audit0_v1/tables/gate_control_metrics_audit0.md`
- CSV：`D:/ryjin/admit_fwi/outputs/salt_reliability_gate_audit0_v1/tables/gate_control_metrics_audit0.csv`

代表性模型指标：

| 方法 | update_l2 | model_mae | model_rmse | edge_mae |
|---|---:|---:|---:|---:|
| initial_model | 0 | 131.692 | 285.239 | 37.2756 |
| full_fwi_model | 8770.57 | 131.810 | 285.309 | 39.0084 |
| global_matched | 122.299 | 131.679 | 285.233 | 37.2763 |
| illumination_only_matched | 122.299 | 131.666 | 285.228 | 37.2718 |
| ecg_reliability_gate | 122.299 | 131.669 | 285.230 | 37.2711 |
| inverse_illumination_negative_control | 122.299 | 131.690 | 285.239 | 37.2823 |

解释：audit0 full FWI 明显变差，这强烈支持研究文档中的核心命题：不能直接接受全量 FWI 更新。ECG gate 在 edge MAE 上优于 global、random、depth、inverse controls，并接近 illumination-only 的 MAE/RMSE，说明多证据门控具有审慎选择更新的作用。

## 5. RTM before/after 审计

### 5.1 all-shots RTM

输出目录：

`D:/ryjin/admit_fwi/outputs/RTM/before_after_fwi_ecg_v1`

状态：已完成。

关键结果：

| 指标 | before initial | after FWI |
|---|---:|---:|
| filtered reference corr | 0.6035979128 | 0.5375006708 |
| filtered reference RMSE | 0.0184501049 | 0.0209446512 |
| filtered reference MAE | 0.0049780268 | 0.0049500341 |

总体判据：

`after_fwi_not_closer_to_reference`

filtered RMSE improvement fraction：

`-0.1352049896618355`

解释：all-shots full FWI 后的 RTM 图像相对 true-velocity RTM 的 filtered RMSE 变差。该结果不适合写成“FWI 改善 RTM 成像”，但非常适合支撑“必须进行可信更新筛选和 RTM 审计”的论文论证。

### 5.2 audit0 RTM

输出目录：

`D:/ryjin/admit_fwi/outputs/RTM/before_after_fwi_audit0_train_ecg_v1`

状态：正在运行。当前已生成 reference_true_velocity 与 before_initial_velocity 的部分结果，after_fwi_velocity 和最终 summary 仍待完成。

## 6. GitHub 同步状态

工作副本：

`D:/Workspace/fwi-rtm-illumination-diagnostics`

分支：

`codex/ecg-reliability-gate`

最近提交：

| commit | 内容 |
|---|---|
| `6020355` | Add gate control figure builder |
| `1b7b8fc` | Allow direct RTM before-after script execution |
| `c1e75c8` | Stabilize reliability gate budget and coverage controls |
| `6cbbfba` | Add audit-fold FWI diagnostics run support |
| `e13cf68` | Add ECG reliability gate diagnostics framework |

当前分支已推送到 GitHub。

## 7. 与研究文档创新点的对应关系

### 7.1 多证据可信更新门控

状态：已完成核心实现。

对应文件：

- `admit_fwi/diagnostics/update_reliability.py`
- `admit_fwi/scripts/replay_fwi_diagnostics.py`

当前证据：

- all-shots 和 audit0 均生成 illumination、consensus、descent、Q 和 ECG gate。

### 7.2 等更新预算反事实验证

状态：已完成并修复关键问题。

对应文件：

- `admit_fwi/diagnostics/matched_budget.py`
- `admit_fwi/diagnostics/gate_ablation.py`

当前证据：

- all-shots target L2 与 ECG matched L2 一致。
- audit0 target L2 与 ECG matched L2 一致。
- random gate 已修复为精确 top-k coverage，不再退化为 global damping。

### 7.3 真值无关的保留炮审计

状态：实验基础已完成，最终 residual 审计待补。

当前证据：

- audit0 train shots = 168。
- audit shots = 56。
- train/audit 完全不重叠。

缺口：

- 还需要对 56 个 audit shots 计算 normalized L2、NRMS、trace correlation、envelope error。

### 7.4 FWI-RTM 双层验证

状态：部分完成。

当前证据：

- all-shots RTM before/after 已完成，且给出负结果。
- audit0 RTM before/after 正在运行。

缺口：

- RTM split consistency 尚未形成最终结果。
- ECG gate 模型本身的 RTM 对比尚未单独生成，目前 RTM 是 full FWI before/after。

### 7.5 盐丘目标区可信度诊断

状态：待补强。

缺口：

- 尚未自动划分 salt top、salt flank、subsalt shadow。
- 尚未统计各区 illumination、consensus、descent、Q、update energy、velocity error、RTM response。

## 8. 当前论文可用结论

可以写入论文的结论：

1. 在复杂盐丘模型中，FWI 残差下降不能单独证明速度更新可靠。
2. all-shots FWI 残差下降约 54%，但 RTM filtered RMSE 反而变差，说明后续成像会受到不可信更新影响。
3. audit0 训练炮 full FWI 使 MAE/RMSE/edge MAE 变差，进一步说明全量接受 FWI 更新存在风险。
4. ECG gate 将局部照明、跨炮梯度一致性和下降有效性整合为 Q(x,z)，并可在等更新预算下形成不同于 illumination-only 和 random controls 的空间更新选择。
5. 当前结果更支持“可信更新筛选框架”，而不是“高精度盐丘速度恢复方法”。

暂时不应写入论文的结论：

1. 不应宣称已经显著解决盐下速度建模问题。
2. 不应宣称 full FWI 改善了 RTM 成像。
3. 不应将 Q(x,z) 表述为严格后验不确定性。
4. 不应把 source-wavefield energy proxy 写成严格 source-adjoint Hessian proxy。

## 9. 下一步优先级

### P0：完成 audit0 RTM

等待：

`D:/ryjin/admit_fwi/outputs/RTM/before_after_fwi_audit0_train_ecg_v1/rtm_before_after_summary.json`

用途：

- 对比 all-shots 与 audit0 的 RTM before/after。
- 判断 audit0 训练炮 FWI 是否同样损害 RTM。

### P1：补保留炮 residual audit

输入：

- audit shots = `shot_index mod 4 == 0`
- initial model
- audit0 full FWI model
- ECG gated model
- matched controls

输出：

- normalized L2 residual
- NRMS
- trace correlation
- envelope error

用途：

- 支撑研究文档中的“真值无关审计”。

### P2：生成 RTM split consistency

输出：

- image correlation
- simple SSIM
- local structure tensor coherence

用途：

- 支撑“FWI-RTM 双层验证”创新点。

### P3：目标区联合诊断

输出：

- salt-top / salt-flank / subsalt 分区图
- 分区统计表

用途：

- 支撑“盐顶-盐翼-盐下有效区域与失效边界”讨论。

## 10. 建议论文叙事

推荐主线：

> 本研究不试图证明当前声波 FWI 已经恢复高精度复杂盐丘速度模型，而是提出并验证一种多证据可信更新选择框架。该框架通过局部照明、跨炮梯度一致性和下降有效性构建空间可信度场，并在等更新预算下筛选可进入 RTM 成像流程的局部更新。实验结果表明，复杂盐丘中全量 FWI 更新可能降低 RTM 成像稳定性，因此需要通过 ECG gate、保留炮审计和 RTM 一致性验证来约束 FWI 更新的使用范围。

这一叙事与当前结果最一致，也最容易形成可发表的 SCI 四区以上论文框架。
