# ADMIT-FWI 新研究思路进展总结

生成日期：2026-07-08  
项目路径：`D:\ryjin\admit_fwi`  
当前主线：`ADMIT-FWI: Admissibility Diagnostics for FWI Updates`

## 1. 总体判断

当前论文不宜继续写成“ECG gate 显著改善复杂盐丘 FWI/RTM 成像”。已有证据显示，ECG 在 audit shots 和 short-record RTM 中有一定优势，但没有稳定超过 `illumination-only`，且部分 random gate 与 ECG/illumination 非常接近。因此，ECG 不能作为单点性能提升型主创新。

更稳的论文主线应改为：

> 面向复杂构造 FWI-RTM 工作流的更新可接受性审计框架，用 matched-budget counterfactual、held-out residual、RTM consistency、目标区诊断、deep-time 和 boundary diagnostics 判断一个 FWI update 是否可以进入成像解释。

也就是说，当前工作的主要价值不是“让 FWI 一定更好”，而是“识别什么时候 residual 下降不能被相信，什么时候 update 应被接受、拒绝或降级解释”。

## 2. 新研究主线

建议将方法命名为 **ADMIT-FWI**：

```text
Admissibility Diagnostics for FWI Updates
```

中文可写为：

```text
FWI 更新可接受性诊断框架
```

新主线包含四个模块：

| 模块 | 名称 | 目的 | 当前状态 |
|---|---|---|---|
| A | Selective Update Candidates | 生成 full/global/illumination/ECG/inverse/random 等候选更新 | 已完成 SEG/Salt 主案例 |
| B | Matched-Budget Counterfactual Test | 在相同更新预算下比较 gate，避免把小更新误判为可靠更新 | 已完成 SEG/Salt 主案例 |
| C | Truth-Free Reliability Audit | 用 held-out audit shots 检查数据空间泛化 | 已完成 SEG/Salt audit0 |
| D | Image and Deep-Time Admissibility | 用 RTM、目标区、深时窗和边界条件判断解释边界 | 部分完成，仍需 split consistency 和区域化补强 |

## 3. 已完成证据

### 3.1 full FWI 残差下降已跑通，但不能单独支撑成像改善

all-shots SEG/Salt FWI 已完成：

| 指标 | 数值 |
|---|---:|
| 模型尺寸 | `676 x 230` |
| 炮数 | `224` |
| 迭代次数 | `3` |
| `nt` | `900` |
| 初始 residual | `4.44358995e-05` |
| 最终 residual | `2.04560124e-05` |
| residual 下降 | `53.9651%` |

但是模型质量指标显示 full FWI 不能直接作为可靠更新接受：

| case | MAE 改善 | RMSE 改善 | Edge MAE 改善 | verdict |
|---|---:|---:|---:|---|
| `CG_allshots_v2` | `0.1119%` | `0.0100%` | `-0.0893%` | numerical improvement without gradient improvement |

结论：FWI 链路有效，数据拟合确实下降；但 residual 下降不能直接等同于速度结构或 RTM 成像可靠。

### 3.2 audit0 held-out residual 已形成数据空间审计

audit0 使用 `168` 个 train shots 和 `56` 个 held-out audit shots，二者互不重叠。held-out summary 显示：

| method | normalized L2 | NRMS | trace corr | envelope | phase |
|---|---:|---:|---:|---:|---:|
| `full_fwi` | `0.0440337` | `0.0442601` | `0.896156` | `0.031947` | `0.907543` |
| `illumination` | `0.0660274` | `0.0663682` | `0.357503` | `0.0567628` | `1.31895` |
| `ecg` | `0.0661386` | `0.0664787` | `0.354569` | `0.056894` | `1.36644` |
| `global` | `0.0664621` | `0.0668070` | `0.347597` | `0.0570738` | `1.36547` |
| `initial` | `0.0668073` | `0.0671509` | `0.337502` | `0.0570012` | `1.48094` |
| `inverse` | `0.0668588` | `0.0672045` | `0.337486` | `0.0570554` | `1.48076` |

关键判断：

- `full_fwi` 在 held-out residual 上最优，但这只能说明数据空间拟合好，不能自动证明结构和成像最可靠。
- `illumination-only` 略优于 `ecg`，说明 ECG 不能写成压倒性优于照明基线。
- `inverse` 差于 initial/global，支持 illumination 方向的物理合理性。

### 3.3 matched-budget gate 和反事实对照已形成方法支撑

旧版空间门控结果表明，`smooth_alpha0.3_thr0.5` 在真值 benchmark 下同时改善 MAE、RMSE 和 edge MAE：

| method | MAE 改善 | RMSE 改善 | Edge MAE 改善 | active fraction | mean alpha |
|---|---:|---:|---:|---:|---:|
| `smooth_alpha0.3_thr0.5` | `0.3102%` | `0.0495%` | `0.0736%` | `0.3635` | `0.0760` |

在新主线中，这个结果不应写成“最终最优 gate”，而应写成：

> matched-budget counterfactual 证明，FWI update 的可靠性需要在相同更新预算下与 global damping、illumination、inverse 和 random controls 比较；否则容易把保守小更新误判为可靠更新。

### 3.4 short-record RTM gate audit 已完成，但结论必须保守

`audit0_gate_rtm_v1` 的 gate RTM summary：

| rank | method | filtered RMSE | filtered corr | improvement vs initial |
|---:|---|---:|---:|---:|
| 1 | `full_fwi` | `0.00521785` | `0.947493` | `0.0544221` |
| 2 | `ecg` | `0.00550338` | `0.939218` | `0.0026775` |
| 3 | `random_seed_4` | `0.00550876` | `0.939094` | `0.00170333` |
| 4 | `illumination` | `0.00550883` | `0.939099` | `0.00168989` |
| 11 | `global` | `0.00551362` | `0.938984` | `0.000821833` |
| 12 | `initial` | `0.00551816` | `0.938879` | `0` |
| 13 | `inverse` | `0.00551892` | `0.938862` | `-0.000139032` |

关键判断：

- ECG 的 RTM filtered RMSE 只略优于 `illumination-only`。
- `random_seed_4` 与 ECG/illumination 非常接近，说明当前全局 RTM 指标区分度不足。
- `full_fwi` 在 short-record RTM 中排名第一，但这不能直接外推到深部/盐下解释。

### 3.5 目标区诊断已能支撑“盐下解释需降级”

已有 salt top/flank/subsalt 目标区表：

| Zone | Pixels | Src illum | Rec illum | Src-rec illum | Src-norm RTM | Lap RTM | Full update | Damped update |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `salt_top` | `5168` | `0.7223` | `0.5322` | `0.6174` | `0.5456` | `0.1744` | `0.185` | `0.018` |
| `salt_flanks` | `6389` | `0.5952` | `0.3160` | `0.4284` | `0.3907` | `0.1955` | `0.003` | `0.000` |
| `subsalt_shadow` | `16150` | `0.3689` | `0.1102` | `0.2011` | `0.1811` | `0.0880` | `0.000` | `0.000` |

结论：

- 盐顶照明和 RTM 响应相对强。
- 盐翼和盐下明显更弱，尤其 subsalt shadow 的 receiver illumination 和 RTM response 都低。
- 当前 FWI update 基本没有有效进入盐下 shadow，因此不能宣称已解决盐下速度建模或盐下成像。

### 3.6 deep-time preflight 已证明短时窗结论不能外推

time-window plan：

| 项目 | 数值 |
|---|---:|
| 当前 audit0 FWI `T` | `0.900 s` |
| 当前 P0-C RTM `T` | `0.600 s` |
| conservative required time | `3.179 s` |
| recommended `nt` | `5000` |
| 当前短时窗是否满足 | `False` |

deep wavefield smoke 使用 `nt=5000, T=5.0 s`，3 个代表炮为 `4, 340, 664`。结果显示：

- 边缘炮 `4` 和 `664` 多数为 `DEEP_WAVEFIELD_NOT_REACHED`。
- 中心炮 `340` 在 initial 和 true model 中都触发 `TIME_TRUNCATION_CONFIRMED`。
- 中心炮 `340` 同时触发 `PML_REFLECTION_RISK`，near-deep-peak boundary ratio 约为 `0.1197` initial / `0.1026` true。

结论：

> 当前 short-record RTM 排名只能作为浅部/早时窗审计结果，不能用于盐下深部解释。deep-time RTM 或 deep-time FWI 必须先解决记录时窗、PML 和炮位覆盖问题。

## 4. 与新研究思路的匹配程度

| 新研究要求 | 当前进展 | 状态 | 说明 |
|---|---|---|---|
| 将主张从 ECG 性能提升转为 update admissibility | 已明确需要调整 | 已完成思路重构 | 后续文稿需整体改写 |
| matched-budget gate comparison | 已有 full/global/illumination/ECG/random/inverse | 已完成 SEG/Salt | 需要整理成 ADMIT-FWI Table |
| held-out residual audit | audit0 56 shots 已完成 | 已完成 SEG/Salt | 数据空间证据充足 |
| RTM candidate audit | audit0 gate RTM 已完成 | 部分完成 | 需要 split consistency 和 ROI RTM |
| deep-time / boundary preflight | time planner、deep smoke、boundary audit 已完成 | 部分完成 | 结论是“深部暂不放行” |
| salt top/flank/subsalt diagnostics | 已有初版目标区指标 | 部分完成 | 需与 gate RTM 和 split consistency 对齐 |
| 多模型复杂度阶梯验证 | 尚未开始 | 待完成 | 这是新主线最大缺口 |

## 5. 当前主要缺口

### 缺口 1：单一 SEG/Salt 模型风险

如果只保留 SEG/Salt，审稿人会质疑 ADMIT-FWI 是否只是针对单一模型调出来的 QC 流程。新主线需要加入模型复杂度阶梯验证，但不应扩张成大规模 benchmark。

建议最小模型矩阵：

| Level | 模型 | 目的 | 状态 |
|---|---|---|---|
| 0 | `simple_layered` | 验证简单模型中 residual/model/RTM 应一致 | 待做 |
| 0 | `simple_fault` | 验证 edge/gradient 审计 | 待做 |
| 1 | `Marmousi crop` | 验证非盐丘复杂沉积构造 | 待做 |
| 2 | `SEG/Salt` | 主案例，复杂盐丘失效模式 | 已有 |
| 3 | `Sigsbee2A crop` | 外部盐丘/盐下 shadow 验证 | 待做 |

### 缺口 2：RTM split consistency 未形成最终结论

已有 `rtm_split_consistency.py` 诊断模块，但当前总结中尚未形成可写入论文的 split RTM 表格。该实验用于回答：

> RTM ranking 是否稳定，还是对 shot split / aperture / random seed 敏感？

这是把 short-record RTM 从单一指标升级为 image-domain audit 的关键。

### 缺口 3：ECG 与 illumination-only 的贡献边界需要重写

现有结果不能支持：

```text
ECG significantly outperforms illumination-only.
```

应改写为：

```text
ECG is one evidence-calibrated selective-update candidate. Illumination-only is a strong baseline. The main contribution is the admissibility audit protocol, not a universally superior ECG gate.
```

### 缺口 4：deep-time 只能写成 preflight negative result

当前 deep-time smoke 尚未放行 deep-time RTM/FWI。论文中可以写：

> short-record RTM cannot be extrapolated to subsalt interpretation.

但不能写：

> ADMIT-FWI improves deep subsalt imaging.

## 6. 下一步最小执行计划

### P1：补齐主案例的论文必要证据

1. 运行或汇总 `RTM split consistency`。
2. 将 gate RTM 指标按 salt top/flank/subsalt ROI 分区统计。
3. 将 deep-time preflight 结果整理为“admissibility gate failed for deep interpretation”的表格。
4. 生成 ADMIT-FWI 主结果表：data-space / model-space / image-space / deep-time-space 四列。

### P2：补模型复杂度阶梯验证

只做轻量实验，不做大 benchmark：

```text
models = simple_layered, simple_fault, Marmousi_crop, Sigsbee2A_crop
shots = 16 或 32
iterations = 2 或 3
audit_fold = small held-out split
methods = full, global, illumination, ECG, inverse, random
outputs = residual, held-out residual, model/edge metrics, simplified RTM or imaging proxy, time-window/boundary audit
```

目标不是证明某个 gate 最优，而是证明 ADMIT-FWI 能识别不同模型上的不同失效模式：

| 模型 | 预期失效模式 | ADMIT-FWI 应给出的判断 |
|---|---|---|
| `simple_layered` | full FWI 通常可接受 | residual/model/RTM 一致 |
| `simple_fault` | 边界局部更新风险 | edge/gradient audit 检出 |
| `Marmousi_crop` | 横向复杂导致局部不稳定 | held-out / model / RTM 分歧 |
| `SEG/Salt` | short-time + illumination + boundary 混杂 | deep-time/boundary preflight 限制解释 |
| `Sigsbee2A_crop` | subsalt shadow 和深部覆盖不足 | deep ROI / split RTM 检出 |

### P3：论文写作重构

建议章节结构：

1. Introduction：从 residual-driven FWI acceptance 的风险切入。
2. Method：ADMIT-FWI 四层审计框架。
3. Experimental design：模型复杂度阶梯 + matched-budget candidates。
4. SEG/Salt main case：完整展示现有 P0-B/P0-C/deep-time evidence。
5. Cross-model validation：simple / Marmousi / Sigsbee 轻量验证。
6. Discussion：ECG 边界、illumination 强基线、deep-time 不放行、truth-free field extension。
7. Conclusion：贡献是可靠性审计流程，不是单一 gate 性能提升。

## 7. 当前可写入论文的结论

可以写：

- full FWI residual 下降明显，但 residual 下降不等于 update admissible。
- matched-budget gate comparison 显示 spatial selective update 通常优于 global/inverse control。
- illumination-only 是强基线，ECG 尚未稳定超过 illumination-only。
- random controls 接近 ECG/illumination，说明全局 RTM 指标不足以单独支撑 gate 优劣。
- short-record RTM 只能支持浅部/早时窗解释，不能外推到盐下深部。
- deep-time and boundary preflight 是 ADMIT-FWI 框架中必要的解释放行条件。

不能写：

- ECG 显著改善复杂盐丘 FWI/RTM。
- 当前结果已解决盐下速度建模。
- short-record RTM 排名代表 deep subsalt imaging quality。
- full FWI held-out residual 最优就说明模型结构最可靠。

## 8. 推荐最终贡献表述

建议论文贡献写成四点：

1. 提出 **ADMIT-FWI**，一个面向 FWI-RTM 工作流的 update admissibility audit framework，统一 data-space、model-space、image-space 和 deep-time/boundary diagnostics。
2. 提出 matched-budget counterfactual gate test，在相同更新预算下比较 full update、global damping、illumination gate、ECG gate、random controls 和 inverse illumination negative control。
3. 在复杂 SEG/Salt 主案例中证明 residual、held-out residual、RTM image 和 deep-time admissibility 可以相互不一致，因此 FWI update 不能仅凭 residual 下降被接受。
4. 通过计划中的 simple/Marmousi/Sigsbee 模型阶梯验证，将方法从单一盐丘 case study 扩展为不同地质复杂度下的 FWI update failure-mode audit。

## 9. 一句话总结

当前项目已经具备从“ECG 门控性能论文”转向“ADMIT-FWI 更新可接受性审计论文”的核心证据；最重要的剩余工作不是继续调 ECG，而是补齐 RTM split/ROI 审计和最小多模型阶梯验证，让论文从单一 SEG/Salt case study 升级为可复现的 FWI update failure-mode 诊断框架。
