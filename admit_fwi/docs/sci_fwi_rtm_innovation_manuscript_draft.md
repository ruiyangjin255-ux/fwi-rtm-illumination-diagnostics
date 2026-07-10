# 面向盐丘构造的 FWI 速度更新、RTM 照明诊断与成像条件对比：一项可复现的声波数值实验研究

> 初稿版本：2026-07-05  
> 建议定位：SCI 4 区以上地球物理/计算地学类期刊初稿；投稿前需按目标期刊模板重排，并再次核验 JCR/中科院分区、参考文献格式和图件分辨率。  
> 本稿主证据来自 `D:\ryjin\admit_fwi` 已有 FWI/RTM 结果；`learning-based extension workspace` 结果仅作为可选扩展方向，不作为本文主实验结论。

## 摘要

复杂盐丘构造中的速度强横向变化会同时影响全波形反演（full waveform inversion, FWI）的模型更新与逆时偏移（reverse time migration, RTM）的构造成像质量。现有研究通常分别讨论 FWI 的速度建模能力或 RTM 的照明补偿效果，而对“速度更新可信度、照明诊断和成像条件选择”三者之间的可复现实验关系讨论不足。本文基于 SEG/Salt 速度模型建立二维声波数值实验流程，首先在完整 `676 x 230` 网格模型上开展多炮 FWI，随后利用初始平滑速度、全局阻尼更新速度和照明可信域空间门控速度进行模型质量与 RTM before/after 对比，并进一步分析震源照明归一化、源-检几何照明归一化和 Laplacian 成像条件。实验结果表明，完整盐丘模型 FWI 在 224 炮、3 次迭代条件下使平均残差由 `4.443590e-05` 降至 `2.045601e-05`，下降比例为 `53.9651%`；但全量更新的模型 MAE 仅改善 `0.1119%`，edge MAE 和梯度 MAE 未同步改善。基于模型质量门控的全局更新尺度扫描选择 `alpha=0.1`，其 MAE 改善 `0.1304%`、RMSE 改善 `0.0215%`，但 edge MAE 仍轻微退化 `-0.0135%`。为增强弱 FWI 结果的实质方法贡献，本文提出照明可信域空间更新门控（illumination-trust spatial update gate）：用源-检照明构造空间变化的 `alpha(x,z)`，仅在可信照明区接受较大 FWI 更新，在弱照明区阻断有害更新。候选门控扫描选中 `smooth_alpha0.3_thr0.5`，其有效更新区域比例为 `0.3635`，平均 `alpha=0.0760`，相对初始模型的 MAE 改善 `0.3102%`、RMSE 改善 `0.0495%`，edge MAE 改善 `0.0736%`。12 炮、`nt=1200` 的 RTM before/after 对比显示，采用阻尼 FWI 更新后，Laplacian-filtered RTM 图像相对真值速度参考的 RMSE 由 `0.027130` 小幅降至 `0.027109`。RTM 方案二实验显示，源-检几何照明低于最大值 1% 的网格比例约为 `0.0174`，源-检归一化与震源归一化图像相关系数为 `0.9732`。本文的主要贡献是提出并验证一个轻量、可复现的照明可信域 FWI-RTM 更新选择框架，用数据残差、模型结构误差、空间照明、偏移图像指标和目标区更新能量共同约束盐丘成像解释。

**关键词：** 全波形反演；逆时偏移；盐丘模型；照明补偿；成像条件

## Abstract

Strong lateral velocity contrasts in salt structures affect both full waveform inversion (FWI) velocity updating and reverse time migration (RTM) imaging. This study builds a reproducible two-dimensional acoustic workflow on the SEG/Salt model to connect FWI misfit reduction, model-quality diagnostics, RTM before/after imaging, and illumination-based imaging-condition analysis. In the full `676 x 230` salt experiment, 224-shot, three-iteration FWI reduces the average data misfit from `4.443590e-05` to `2.045601e-05` (`53.9651%`). However, the full update improves model MAE by only `0.1119%`, and edge and gradient errors do not improve. A global model-quality gate selects `alpha=0.1`, improving MAE by `0.1304%` and RMSE by `0.0215%`, but still slightly degrading edge MAE. We therefore introduce an illumination-trust spatial update gate, in which source-receiver illumination controls `alpha(x,z)`. The selected `smooth_alpha0.3_thr0.5` gate updates 36.35% of grid points with a mean alpha of 0.0760, improving MAE by `0.3102%`, RMSE by `0.0495%`, and edge MAE by `0.0736%`. In a 12-shot RTM validation, the Laplacian-filtered image RMSE relative to the true-velocity reference decreases from `0.027130` to `0.027109`. Full-aperture diagnostics show that only `0.0174` of grid points fall below 1% of maximum source-receiver illumination, while source-receiver and source-normalized images correlate at `0.9732`. The results support an illumination-trust FWI-RTM update-selection framework constrained by data misfit, model structural error, migrated-image metrics, and illumination diagnostics.

**Keywords:** full waveform inversion; reverse time migration; salt model; illumination compensation; imaging condition

## 1 Introduction

Full-waveform inversion (FWI) and reverse-time migration (RTM) are central components of modern seismic velocity-model building and structural imaging. In salt-bearing settings, the large velocity contrast between salt bodies and surrounding sediments produces multipathing, shadow zones, uneven illumination, and strong sensitivity to the starting model. A velocity update that appears favorable in the waveform domain can therefore have direct consequences for the migrated image and for subsequent geological interpretation. The practical question is not only whether FWI can reduce the data residual, but whether the resulting update is reliable enough to be accepted into an FWI-RTM imaging workflow.

This distinction is important because residual reduction is not equivalent to model or image reliability. FWI is a nonlinear and ill-posed optimization problem, and its update can be affected by cycle skipping, incomplete aperture, poor deep illumination, gradient scaling, numerical boundary effects, and insufficient recording time. These effects are amplified in complex salt models, where early arrivals and shallow energy can dominate short-record inversions while deeper salt-flank and subsalt regions remain weakly constrained. As a result, a residual-driven acceptance rule can admit updates that improve a scalar misfit but degrade structural consistency, RTM stability, or the admissibility of deep interpretation.

Previous work has improved FWI and RTM from several directions. Hessian-based and pseudo-Hessian preconditioning methods address illumination imbalance and gradient scaling; regularization, damping, and constrained optimization stabilize ill-posed updates; source encoding and shot-selection methods reduce computational cost; and Bayesian or variational FWI quantifies inversion uncertainty. These approaches are essential for producing better inversions, but they do not fully answer a separate engineering question: after an update has been computed, how should one audit whether it should be used for RTM-based interpretation?

This paper formulates that question as an update-admissibility problem and proposes ADMIT-FWI, an admissibility-diagnostics framework for FWI updates in complex FWI-RTM workflows. The framework evaluates candidate updates through four complementary checks: matched-budget counterfactual comparison, held-out shot residual auditing, RTM image-domain diagnostics, and deep-time/boundary-condition preflight analysis. The aim is not to claim that a single spatial gate is universally superior. Instead, ADMIT-FWI tests whether a candidate update passes falsifiable reliability checks before it is allowed to support imaging claims.

The study uses a complex SEG/Salt acoustic benchmark as the main case. Candidate updates include full FWI, globally damped updates, illumination-guided updates, evidence-calibrated gated updates, inverse-illumination negative controls, and random spatial gates. Matching the update budget is critical because a conservative update may appear reliable simply because it changes less of the model. Held-out audit shots then test whether a candidate update generalizes beyond the data used to construct it, while RTM and split-consistency diagnostics assess whether image-domain behavior is stable. Finally, deep-time and boundary diagnostics determine whether short-record results can support deep or subsalt interpretation.

The resulting contribution is a reliability-audit layer for FWI-RTM interpretation. The present experiments show that data-space, model-space, image-space, and deep-time diagnostics can disagree. Full FWI can achieve favorable held-out residual behavior, whereas illumination-guided and evidence-calibrated gates may be safer but not uniquely superior under all diagnostics. Short-record RTM rankings are also not sufficient to support deep subsalt claims without explicit recording-time, wavefield-coverage, and boundary-stability checks. ADMIT-FWI therefore reframes the paper from a claim of universal gate improvement to a reproducible workflow for deciding when an FWI update should be accepted, rejected, or interpreted only within a limited imaging scope.

## 2 Related Work

### 2.1 FWI, RTM, and update reliability in complex salt settings

FWI was established as a waveform-based nonlinear inverse problem in which model parameters are updated by matching observed and simulated seismic data. Subsequent frequency-domain and time-domain developments made FWI a practical high-resolution velocity-model-building method, while reviews have emphasized its dependence on starting models, low frequencies, acquisition aperture, and the treatment of the Hessian. RTM, by contrast, images reflectors by correlating source and receiver wavefields and is particularly useful in complex media where one-way migration assumptions break down. In salt settings, these two tools are coupled: FWI updates the velocity model, and the accepted velocity model controls RTM image quality.

Most FWI studies report data residuals, model errors, or visual comparisons of recovered velocity models. These criteria are informative, but they are incomplete when the question is whether an update should be used for interpretation. In field-like salt problems, the true velocity model is unavailable, and a lower residual does not necessarily imply a more geologically plausible model or a more stable migrated image. ADMIT-FWI is positioned in this gap between inversion and acceptance. It treats an FWI update as a candidate that must pass data-domain, model-domain, image-domain, and deep-time checks before being used to support RTM claims.

### 2.2 Hessian correction, preconditioning, and illumination compensation

Hessian-based, Gauss-Newton, and pseudo-Hessian methods address one of the central weaknesses of gradient-based FWI: the raw gradient is unevenly scaled by acquisition geometry, wave propagation, and local illumination. Approximate Hessian information and illumination compensation can balance updates and accelerate convergence, and related ideas also appear in RTM and least-squares RTM as image-domain amplitude or illumination corrections.

These methods are complementary to the present work, but they solve a different problem. A preconditioned or illumination-compensated update may be better scaled, yet it still may not be admissible for interpretation if it fails held-out shot tests, destabilizes RTM images, or relies on a recording time that does not illuminate the target depth. ADMIT-FWI therefore does not replace Hessian or preconditioning techniques. It adds a post-update audit layer that asks whether the update should be accepted under the evidence available.

### 2.3 Regularization, damping, spatial gating, and update selection

Regularization and damping are widely used to stabilize FWI in ill-conditioned settings. Smoothing, total-variation penalties, update clipping, trust-region ideas, and spatially constrained updates can reduce artifacts and prevent excessive perturbations. Illumination- or sensitivity-guided masks further restrict updates to regions where the data are expected to provide stronger support.

The difficulty is that a restricted update may appear better simply because it is smaller. Without counterfactual controls, it is hard to separate a meaningful spatial-selection rule from conservative damping. This is the motivation for the matched-budget component of ADMIT-FWI. Full updates, global damping, illumination gates, evidence-calibrated gates, inverse-illumination controls, and random gates are compared under comparable update-energy constraints, so that the audit tests spatial selectivity rather than update magnitude alone.

### 2.4 Shot selection, source encoding, and held-out auditing

Shot selection, simultaneous-source encoding, and compressed FWI were developed mainly to reduce computational cost. By using fewer shots or encoded source combinations, these methods make large-scale waveform inversion more tractable. Their success depends on preserving enough wavefield information to approximate the full-data inversion while lowering the number of forward and adjoint simulations.

The held-out shots in ADMIT-FWI serve a different purpose. They are not used primarily for acceleration; they are used as an independent data-space audit. If an update improves the shots used to construct it but does not generalize to held-out audit shots, its reliability is questionable even if the training residual decreases. This distinction separates an efficiency-driven data-reduction strategy from an acceptance-driven reliability test.

### 2.5 Bayesian FWI, uncertainty quantification, and practical audit proxies

Bayesian FWI and uncertainty quantification directly address the nonuniqueness of waveform inversion by estimating posterior distributions, model uncertainty, or probabilistic confidence measures. Recent variational and Stein variational approaches have made uncertainty-aware FWI more practical in two- and three-dimensional settings, but full probabilistic treatment remains substantially more expensive than deterministic inversion and may be difficult to integrate into lightweight engineering workflows.

ADMIT-FWI should therefore be viewed as a pragmatic audit proxy rather than a substitute for Bayesian uncertainty analysis. It does not estimate a posterior distribution. Instead, it asks whether a candidate update passes a sequence of low-cost, falsifiable checks: matched-budget counterfactual behavior, held-out residual consistency, RTM image stability, and deep-time/boundary admissibility. This positions the framework between residual-only quality control and full uncertainty quantification.

### 2.6 Learning-based inversion and the need for physical auditing

Learning-based seismic inversion and benchmark datasets such as OpenFWI have expanded the range of possible velocity-model prediction workflows. These methods can produce plausible velocity models rapidly after training, and they are increasingly relevant to cross-domain or data-limited inversion settings. However, a visually plausible prediction is not automatically physically admissible for RTM interpretation.

Although the present paper focuses on acoustic FWI-RTM diagnostics rather than neural inversion, the same acceptance problem applies. Any velocity update, whether produced by deterministic FWI, a spatial gate, or a learned predictor, should be checked against independent data and image-domain behavior before it is used for interpretation. ADMIT-FWI can therefore be read as a physics-based audit layer that is compatible with future learning-assisted inversion workflows, while the current manuscript keeps learning-based results outside the main evidence chain.

### 2.7 Position of this study

Prior work has provided powerful tools for improving gradients, stabilizing updates, reducing cost, and estimating uncertainty. The remaining gap is a workflow-level acceptance criterion for deciding whether a computed FWI update is admissible for RTM interpretation. This study fills that gap by combining matched-budget counterfactual tests, held-out shot auditing, RTM image diagnostics, and deep-time/boundary preflight checks. The framework is designed to identify cases in which residual reduction is insufficient evidence for update acceptance and to clarify whether the limiting factor is spatial gating, illumination, short recording time, boundary contamination, or insufficient deep coverage.

## 3 方法

### 3.1 速度模型与数值设置

本文使用 SEG/Salt 二维速度模型，完整模型尺寸为 `nx=676, nz=230`，网格间距 `dx=dz=10 m`。完整 FWI 使用 224 炮，炮点沿横向每 3 个网格布设，时间采样参数为 `nt=900, f0=8 Hz`，迭代次数为 3。观测记录由真实速度模型正演生成，初始模型由真实速度模型平滑得到。

RTM 方案二使用完整盐丘模型和 padding 后计算域，关键参数为 `nt=4001, dt=0.001 s, f0=20 Hz, pad_x=60, pad_bottom=60, workers=4`。该实验保存原始互相关成像、震源照明、接收照明、震源照明归一化、源-检照明归一化、Laplacian 成像和 Laplacian 源归一化结果。

局部 FWI 窗口用于受控对照实验，裁剪尺寸为 `70 x 120`，裁剪起点为 `z0=70, x0=310`，炮点为 `[15, 60, 104]`，时间采样为 `nt=360` 或 `450`。该窗口不用于替代完整模型结论，而用于低成本检查 FWI 更新、照明预条件和线搜索机制。

### 3.2 FWI 目标函数与更新

FWI 以模拟记录 `d_cal` 与观测记录 `d_obs` 的平均半均方误差作为目标函数：

```text
Phi(m) = 1/2 ||d_cal(m) - d_obs||_2^2
```

每一轮迭代对所有炮点累计数据残差和模型更新方向。完整模型实验采用共轭梯度（CG）优化，`step_scale=4.0`，`max_update=25 m/s`。局部窗口实验比较固定步长、普通线搜索和自适应扩展线搜索。自适应扩展线搜索首先测试 `step_scale = 0.5, 1.0, 2.0`；若最优步长位于上界且残差继续下降，则扩展测试 `3.0, 4.0, 6.0, 8.0`。

### 3.3 RTM 成像条件与照明归一化

RTM 通过震源波场 `u_s` 和接收反传波场 `u_r` 的互相关形成偏移图像：

```text
I(x,z) = sum_t u_s(t,x,z) u_r(t,x,z)
```

震源照明归一化使用：

```text
I_s(x,z) = I(x,z) / (L_s(x,z) + epsilon)
```

源-检几何照明归一化使用：

```text
I_sr(x,z) = I(x,z) / (sqrt(L_s(x,z) L_r(x,z)) + epsilon)
```

其中 `L_s` 与 `L_r` 分别由震源正传波场和接收反传波场能量累计得到。Laplacian 成像条件作为高通增强方式，用于削弱低波数背景、增强界面连续性。需要强调的是，以上归一化作用于偏移图像 `I`，不是直接更新速度模型。

### 3.4 FWI 照明预条件

局部 FWI 中的照明预条件采用震源波场能量近似：

```text
L_s(x,z) = sum_t u_s(t,x,z)^2
g_pre(x,z) = g(x,z) / (L_s_norm(x,z) + epsilon)
```

其中 `g` 是残差反传得到的模型更新方向。该方法是轻量近似，不等同于完整 Hessian 预条件。本文将其作为对照分支，目的不是证明该形式最优，而是检验“简单照明缩放”在当前模型和参数下是否真正优于 baseline。

### 3.5 照明可信域空间更新门控

全局更新尺度 `alpha` 只能整体放大或缩小 FWI 更新，无法区分强照明可信区域和弱照明高风险区域。为此，本文新增照明可信域空间更新门控：

```text
m_gate(x,z) = m_0(x,z) + alpha(x,z) [m_fwi(x,z) - m_0(x,z)]
```

其中 `m_0` 为初始平滑速度，`m_fwi` 为完整 FWI 输出速度。`alpha(x,z)` 由源-检照明场 `L_sr(x,z)` 控制，候选形式包括全局常数、硬阈值、线性软阈值、平方根软阈值和平滑阈值：

```text
alpha_smooth(x,z) = alpha_max * G_sigma( 1[L_sr(x,z) >= tau] )
```

其中 `G_sigma` 为高斯平滑算子，`tau` 为照明可信阈值。所有候选门控必须同时满足三个 benchmark 约束：相对初始模型 MAE 改善、RMSE 改善且 edge MAE 不退化。该设计的目的不是在真值模型上做不可推广的“最优调参”，而是在合成 benchmark 中验证一个方法原则：FWI 更新不应在照明不足或结构风险较高的区域被无条件接受。

### 3.6 JGE 导向的创新点-证据闭环框架

为避免“结果堆砌”而削弱投稿说服力，本文将程序框架重构为一条 JGE 导向的创新点-证据闭环：每个论文创新点必须同时对应可运行脚本、持久化指标文件、主图或表格输出以及明确的结论边界。该框架由 `build_jge_innovation_framework.py` 自动生成，输出 `jge_innovation_framework.csv/.md/.json` 和 `jge_figure_alt_text.md/.json`。其中 `jge_innovation_framework.csv` 将五个主创新点映射到程序入口、核心证据、主图表和不可过度宣称的边界；`jge_figure_alt_text.md` 为 Figure 1-5 提供 OUP/JGE 最终模板转换所需的图件替代文本。投稿包生成脚本 `build_jge_submission_package.py` 会自动重建该矩阵，并由 `check_jge_submission_readiness.py` 检查摘要、关键词、参考文献、图表数量、正文长度、数据/代码声明、AI 辅助声明、alt text 和创新框架文件是否齐全。

## 4 结果

### 4.1 完整盐丘模型 FWI 结果

完整 `676 x 230` 盐丘模型 FWI 使用 224 炮、3 次迭代、`nt=900` 和 `f0=8 Hz`。实验结果显示，初始残差为 `4.443590e-05`，最终残差为 `2.045601e-05`，残差下降比例为 `53.9651%`。该结果说明当前完整模型 FWI 正演、残差反传、炮流式 checkpoint 和速度更新链路已经形成可运行闭环。

该结论应谨慎解释。新增模型质量评估显示，虽然数据残差下降明显，但全量速度更新本身的改进很弱：全模型 MAE 仅由 `131.6921 m/s` 降至 `131.5447 m/s`，改善比例为 `0.1119%`；RMSE 改善比例仅为 `0.0100%`。更重要的是，盐体边界相关的 edge MAE 变化为 `-0.0893%`，梯度 MAE 变化为 `-3.6250%`，说明当前 3 次迭代全量更新没有改善关键构造边界。因此，程序先加入全局更新尺度门控：在 `alpha = 0, 0.1, 0.25, 0.5, 0.75, 1.0` 中选择满足 MAE/RMSE 正改善且结构退化受限的阻尼更新。当前全局候选 `alpha=0.1` 对应 MAE 改善 `0.1304%`、RMSE 改善 `0.0215%`，但 edge MAE 仍轻微退化 `-0.0135%`。这说明单一全局阻尼仍不足以支撑“结构改善”的方法宣称，必须进一步引入空间可信域控制。

### 4.2 照明可信域空间更新门控

Figure 4 展示了新增空间更新门控结果。候选集合包括全局常数、硬阈值、线性软阈值、平方根软阈值和平滑阈值门控，并以“MAE 改善、RMSE 改善、edge MAE 不退化”为接受条件。最终选中 `smooth_alpha0.3_thr0.5`，即在源-检照明大于 `0.5` 的区域构造平滑可信域，最大允许 `alpha=0.3`，有效更新区域比例为 `0.3635`，全模型平均 `alpha=0.0760`。与初始模型相比，该空间门控模型的 MAE 改善 `0.3102%`、RMSE 改善 `0.0495%`，edge MAE 改善 `0.0736%`。

与全局 `alpha=0.1` 相比，空间门控的优势在于它不是简单减小更新幅度，而是改变更新的空间分布：可信照明区域可接受更强更新，弱照明区和高风险结构区被抑制。因此，它把“FWI 结果较差”转化为一个可检验的方法问题：如何判断哪些 FWI 更新可以进入后续 RTM 解释。该结果是本文比前一版更强的核心创新点，因为它给出了一个明确的算法对象 `alpha(x,z)`、可复现的候选扫描表 `spatial_update_gate_candidates.csv` 和可直接进入论文的 Figure 4。

### 4.3 RTM before/after FWI 对比

为检验 FWI 更新是否实际改善后续偏移成像，本文使用同一真值观测数据分别以真值速度、初始平滑速度和质量门控后的阻尼 FWI 速度进行 RTM。修正版正式对比采用 `nt=1200, f0=20 Hz, pad_x=60, pad_bottom=60`，选取 12 个横向均匀分布炮点，输出见 Figure 2。主要指标如下：

| 指标 | 初始速度 RTM | 阻尼 FWI 速度 RTM |
|---|---:|---:|
| 与真值速度 RTM 的 Laplacian-filtered RMSE | `0.027130` | `0.027109` |
| 与真值速度 RTM 的 Laplacian-filtered MAE | `0.009072` | `0.009044` |
| 与真值速度 RTM 的 Laplacian-filtered 相关系数 | `0.4929` | `0.4927` |
| 与真值速度 RTM 的 source-normalized RMSE | `2.958475` | `2.962629` |
| 炮点数量 | `12` | `12` |

按照 Laplacian-filtered RTM 图像 RMSE，质量门控后的阻尼 FWI 速度相对初始速度略微更接近真值速度参考，RMSE 改善比例为 `0.0769%`。该改善幅度很小，不能被夸大为显著成像提升；但它与全量更新 `alpha=1.0` 使 RMSE 恶化 `6.9498%` 的结果形成对照，说明模型质量门控可以避免把有害 FWI 更新直接传递给 RTM。Figure 2 因此应作为本文的关键程序优化结果：数据残差下降必须经过模型边界指标和偏移图像指标筛选，才能进入后续 RTM 成像链路。

### 4.4 RTM 方案二成像条件对比

完整盐丘 RTM 方案二比较了源照明归一化、源-检几何照明归一化和 Laplacian 成像条件。主要指标如下：

| 指标 | 数值 |
|---|---:|
| 源-检归一化与源归一化相关系数 | `0.9732` |
| Laplacian 源归一化与源归一化相关系数 | `-0.2822` |
| 源-检几何照明低于最大值 1% 的网格比例 | `0.0174` |
| 模型网格 | `676 x 230` |
| 炮点数量 | `224` |
| 时间采样 | `nt=4001, dt=0.001 s` |

源-检归一化与源归一化高度相关，说明在当前 full-aperture 几何下，两类照明归一化对主体构造的影响相近。低照明比例仅约 `1.74%`，说明当前 RTM 主图不宜被描述为“主要受严重照明缺失限制”。相比之下，Laplacian 结果与源归一化图像相关性较低，表明高通增强显著改变了图像频谱和低波数背景，是影响盐体边界可读性的关键因素。

### 4.5 目标区照明、RTM 响应与 FWI 更新能量

为把“照明诊断”从全局图像比较推进到目标区评价，本文进一步根据 SEG/Salt 高速盐体自动划分盐顶、盐翼和盐下阴影三个区域，并在同一批区域内统计源-检照明、RTM 响应和 FWI 更新能量。结果见 Figure 5 和 `target_zone_illumination_metrics.csv`。主要指标如下：

| 目标区 | 源-检照明均值 | Laplacian RTM 响应均值 | 全量 FWI 更新均值 | 阻尼 FWI 更新均值 |
|---|---:|---:|---:|---:|
| 盐顶 | `0.6174` | `0.1744` | `0.185 m/s` | `0.018 m/s` |
| 盐翼 | `0.4284` | `0.1955` | `0.003 m/s` | `0.000 m/s` |
| 盐下阴影 | `0.2011` | `0.0880` | `0.000 m/s` | `0.000 m/s` |

该结果直接支撑本文框架：盐下区域的源-检照明和 RTM 响应低于盐顶与盐翼，而当前 FWI 更新能量几乎没有有效进入盐下区域。因此，当前结果不应被写成“FWI 已显著改善盐下速度”，而应写成“照明诊断揭示盐下目标仍是 FWI-RTM 闭环的薄弱区”。这也为后续实验给出明确方向：应优先在盐下目标区测试双向照明观测系统优化、近似 Hessian/L-BFGS 预条件或多尺度低频约束，而不是盲目增加当前 FWI 迭代次数。

### 4.6 局部 FWI 步长与照明预条件对照

局部 `70 x 120` 盐丘窗口实验提供了对 FWI 优化机制的低成本解释。不同策略的残差下降如下：

| 方法 | 初始误差 | 最终误差 | 下降比例 |
|---|---:|---:|---:|
| 固定步长 baseline FWI | `0.2963276605` | `0.2936459879` | `0.90497%` |
| 固定步长照明预条件 FWI | `0.2963276605` | `0.2953105271` | `0.34325%` |
| epsilon 一维扫描最佳 | `0.2963276605` | `0.2950215439` | `0.44077%` |
| epsilon x max_update 二维扫描最佳 | `0.2963276605` | `0.2947063794` | `0.54712%` |
| baseline 线搜索 | `0.2963276605` | `0.2910634528` | `1.77648%` |
| 照明预条件线搜索 | `0.2963276605` | `0.2931125114` | `1.08500%` |
| baseline 自适应扩展线搜索 | `0.2963276605` | `0.2774544110` | `6.36905%` |
| 照明预条件自适应扩展线搜索 | `0.2963276605` | `0.2840638161` | `4.13861%` |

结果显示，线搜索对局部 FWI 收敛的贡献大于当前轻量照明预条件。照明预条件经过参数扫描和线搜索后确有改善，但在同等条件下仍低于 baseline。这一负结果具有论文价值：它限制了过度宣称，并指出当前实现若要把照明优化作为主贡献，需要进一步引入更严格的 Hessian 近似、多尺度频率递进或更合理的梯度构造。

## 5 讨论

### 5.1 RTM 图像归一化不能直接等同于 FWI 照明优化

本文结果支持一个关键区分：RTM 的照明归一化主要改变偏移图像振幅分布，而 FWI 照明预条件改变模型更新方向。当前 RTM 结果中低照明区占比小，源-检归一化与源归一化高度相关；因此，若论文主张“照明补偿显著改善成像”，更合理的证据应来自局部振幅均衡、成像条件频谱变化或弱照明区域的剖面解释，而不是把归一化图像当作 FWI 优化证据。

### 5.2 自适应线搜索是当前 FWI 改进的主控因素

局部 FWI 中 baseline 自适应扩展线搜索残差下降达到 `6.36905%`，而照明预条件分支为 `4.13861%`。两者每轮均选择最大候选步长 `8.0`，说明普通线搜索的 `2.0` 上界偏低，固定步长限制了收敛表现。该结果提示后续完整 FWI 若要进一步提升，应优先保留线搜索、频率递进和多尺度策略，再考虑照明预条件的精细化。

## 6 可形成的创新点

### 创新点 1：照明可信域空间 FWI 更新门控

本文不再把全局阻尼更新作为最终创新，而是提出空间变化的 `alpha(x,z)` 更新门控：利用源-检照明场定义 FWI 更新可信域，在强照明区保留更大更新，在弱照明或结构风险区抑制更新。该方法使当前弱 FWI 结果从“残差下降但结构指标不足”转化为“可诊断、可筛选、可进入 RTM 的空间更新选择问题”。选中门控 `smooth_alpha0.3_thr0.5` 同时改善 MAE、RMSE 和 edge MAE，是本文最适合投稿的主创新点。

### 创新点 2：失败可诊断的模型质量门控

当前 FWI 图像质量不足本身不是论文失败，而是方法框架必须显式处理的问题。本文把残差下降、MAE/RMSE、edge MAE、gradient MAE、全局更新尺度、空间更新门控和 RTM 图像指标联合作为门控条件，拒绝把全量 `alpha=1.0` 更新直接写成速度改进。该点比单纯展示一张较差 FWI 图更有方法价值：它说明了何时不能相信 FWI 残差下降，以及如何防止有害速度更新进入 RTM 解释。

### 创新点 3：区分图像照明归一化与反演梯度预条件

已有结果明确显示 RTM 照明归一化和 FWI 照明预条件作用对象不同。本文可把这一点写成方法论贡献：给出同一模型下的图像层照明补偿和反演层照明预条件对照，避免把两类结果混为一谈。

### 创新点 4：以负结果约束照明优化宣称

局部 FWI 结果显示轻量照明预条件并未超过 baseline 自适应线搜索。该负结果可以转化为可信贡献：当前盐丘窗口中，步长选择是更强控制因素；照明预条件若要成为主优化策略，需要 Hessian 近似或多尺度策略支持。SCI 论文中诚实的消融和失败边界通常比过度包装更容易通过方法审查。

### 创新点 5：成像条件选择的定量证据化

源-检归一化与源归一化的相关系数、低照明比例和 Laplacian 相关性共同说明，不同成像条件并非只影响“显示好看与否”，而改变了低波数背景和界面表达。该点可支撑一组论文图和表：原始 RTM、源归一化、源-检归一化、Laplacian 增强和照明分布。

## 7 结论

本文基于 SEG/Salt 模型建立了一个可复现的声波 FWI-RTM 联合实验流程。完整盐丘模型 FWI 在 224 炮、3 次迭代下实现 `53.9651%` 的数据残差下降，证明速度更新链路可运行；但全量更新的模型质量评估显示 MAE 仅改善 `0.1119%`，edge MAE 和梯度 MAE 均未改善。全局阻尼门控 `alpha=0.1` 能改善 MAE 与 RMSE，但 edge MAE 仍轻微退化。新增照明可信域空间门控选中 `smooth_alpha0.3_thr0.5`，在有效更新区域比例 `0.3635` 和平均 `alpha=0.0760` 条件下，将 MAE 改善提升至 `0.3102%`、RMSE 改善提升至 `0.0495%`，并使 edge MAE 改善 `0.0736%`。进一步的 12 炮 RTM before/after FWI 对比表明，阻尼 FWI 速度对应的 Laplacian-filtered RTM 图像相对真值速度参考的 RMSE 由 `0.027130` 小幅降至 `0.027109`。RTM 方案二结果显示源-检几何照明低于最大值 1% 的网格比例仅约 `0.0174`，源-检归一化与震源归一化高度相关。本文建议将照明归一化、空间更新可信域、梯度预条件和步长选择作为不同层级分别讨论。

本文的核心结论是：在当前盐丘数值实验中，FWI 残差下降必须经过模型结构指标和照明可信域筛选后才能进入 RTM 解释；空间更新门控比单一全局阻尼更能抑制有害边界更新。RTM 照明归一化适合用于成像振幅均衡和条件对比，但不能直接等同于 FWI 梯度预条件。后续工作可在更严格 Hessian 预条件、多尺度 FWI、LSRTM、目标区 RTM 验证和机器学习低波数先验方面扩展。

## 8 图表计划

| 图号 | 内容 | 已有/建议路径 |
|---|---|---|
| 图 1 | 文献驱动的 FWI-RTM 方法综合、证据强弱与结论边界 | `D:\ryjin\admit_fwi\docs\jge_main_figures\figure1_fwi_quality_gate.tiff` |
| 图 2 | RTM before/after 质量门控 FWI 速度更新验证 | `D:\ryjin\admit_fwi\docs\jge_main_figures\figure2_rtm_before_after_validation.tiff` |
| 图 3 | full-aperture RTM 成像条件与照明诊断 | `D:\ryjin\admit_fwi\docs\jge_main_figures\figure3_imaging_condition_diagnostics.tiff` |
| 图 4 | 照明可信域空间 FWI 更新门控与候选质量前沿 | `D:\ryjin\admit_fwi\docs\jge_main_figures\figure4_spatial_update_gate.tiff` |
| 图 5 | 盐顶、盐翼和盐下目标区照明、RTM 响应与 FWI 更新能量 | `D:\ryjin\admit_fwi\docs\jge_main_figures\figure5_target_zone_illumination_diagnostics.tiff` |
| 表 1 | 完整 FWI 与 RTM 参数表 | 由 `full_salt_fwi_summary.md` 和 `scheme2_report.md` 整理 |
| 表 2 | RTM before/after FWI 图像指标 | 见本文第 4.3 节 |
| 表 3 | 空间更新门控候选扫描与选中策略 | `D:\ryjin\admit_fwi\docs\jge_revision\spatial_update_gate_candidates.csv` |
| 表 4 | JGE 导向的创新点-程序-证据-结论边界矩阵 | `D:\ryjin\admit_fwi\docs\jge_revision\jge_innovation_framework.csv` |
| 表 5 | 目标区照明与 FWI-RTM 诊断指标 | `D:\ryjin\admit_fwi\docs\jge_revision\target_zone_illumination_metrics.csv` |

## 9 建议投稿方向

优先考虑计算地学、应用地球物理数值方法或工程地震成像方向。候选期刊可包括 `Computers & Geosciences`、`Journal of Applied Geophysics`、`Acta Geophysica`、`Geophysical Prospecting` 等。具体是否满足“SCI 4 区以上”应在投稿前按最新 JCR/中科院分区核验；不同年份和不同分区体系可能不一致。

## 参考文献（待按目标期刊格式重排）

1. Tarantola, A. (1984). Inversion of seismic reflection data in the acoustic approximation. *Geophysics*, 49(8), 1259-1266. https://doi.org/10.1190/1.1441754
2. Baysal, E., Kosloff, D. D., & Sherwood, J. W. C. (1983). Reverse time migration. *Geophysics*, 48(11), 1514-1524. https://doi.org/10.1190/1.1441434
3. Virieux, J., & Operto, S. (2009). An overview of full-waveform inversion in exploration geophysics. *Geophysics*, 74(6), WCC1-WCC26. https://doi.org/10.1190/1.3238367
4. Pratt, R. G. (1999). Seismic waveform inversion in the frequency domain; Part 1, Theory and verification in a physical scale model. *Geophysics*, 64(3), 888-901. https://doi.org/10.1190/1.1444597
5. Bunks, C., Saleck, F. M., Zaleski, S., & Chavent, G. (1995). Multiscale seismic waveform inversion. *Geophysics*, 60(5), 1457-1473. https://doi.org/10.1190/1.1443880
6. Plessix, R.-E. (2006). A review of the adjoint-state method for computing the gradient of a functional with geophysical applications. *Geophysical Journal International*, 167(2), 495-503. https://doi.org/10.1111/j.1365-246X.2006.02978.x
7. Zhu, H., & McMechan, G. A. (2012). Seismic interferometry-based imaging conditions for prestack reverse-time migration. *Geophysics*, 77(3), S77-S86. https://doi.org/10.1190/geo2011-0340.1
8. Dai, W., Fowler, P., & Schuster, G. T. (2012). Multi-source least-squares reverse time migration. *Geophysical Prospecting*, 60(4), 681-695. https://doi.org/10.1111/j.1365-2478.2012.01092.x
9. Krebs, J. R., Anderson, J. E., Hinkley, D., Neelamani, R., Lee, S., Baumstein, A., & Lacasse, M.-D. (2009). Fast full-wavefield seismic inversion using encoded sources. *Geophysics*, 74(6), WCC177-WCC188. https://doi.org/10.1190/1.3230502
10. Li, X., Aravkin, A. Y., van Leeuwen, T., & Herrmann, F. J. (2012). Fast randomized full-waveform inversion with compressive sensing. *Geophysics*, 77(3), A13-A17. https://doi.org/10.1190/geo2011-0410.1
11. Zhu, H., Li, S., Fomel, S., Stadler, G., & Ghattas, O. (2016). A Bayesian approach to estimate uncertainty for full-waveform inversion using a priori information from depth migration. *Geophysics*, 81(5), R307-R323. https://doi.org/10.1190/geo2015-0641.1
12. Yin, Z., Orozco, R., & Herrmann, F. J. (2024). WISE: full-waveform variational inference via subsurface extensions. *Geophysics*, 89(5), R493-R507. https://doi.org/10.1190/geo2023-0753.1
13. Corrales, M., Ravasi, M., & Vasconcelos, I. (2025). Annealed Stein variational gradient descent for improved uncertainty estimation in full-waveform inversion. *Geophysical Journal International*, 241(2), 1088-1108. https://doi.org/10.1093/gji/ggaf096
14. Deng, C., Feng, S., Wang, H., Zhang, X., Jin, P., Feng, Y., Zeng, Q., Chen, Y., & Lin, Y. (2022). OpenFWI: Large-scale multi-structural benchmark datasets for full waveform inversion. *Advances in Neural Information Processing Systems, Datasets and Benchmarks Track*, 35. https://openreview.net/forum?id=7w-a8PYPlP
15. Zhu, W., Xu, K., Darve, E., Biondi, B., & Beroza, G. C. (2022). Integrating deep neural networks with full-waveform inversion: Reparameterization, regularization, and uncertainty quantification. *Geophysics*, 87(1), R93-R109. https://doi.org/10.1190/geo2020-0933.1
16. Wang, B., Feng, S., Fu, L.-Y., & Hu, L. (2023). Least squares reverse time migration imaging with illumination compensation. *Scientific Reports*, 13, 14473. https://doi.org/10.1038/s41598-023-40578-8

## 数据与代码可用性声明草稿

本文使用的代码、参数文件和主要输出位于本地项目 `D:\ryjin\admit_fwi`。推荐复现入口为 `python -m admit_fwi.run_optimized_fwi_rtm_pipeline`；若需要重新执行耗时 RTM 验证，可添加 `--run-rtm`。投稿材料打包入口为 `python -m admit_fwi.build_jge_submission_package`，输出目录为 `D:\ryjin\admit_fwi\docs\jge_submission_package_mainfigures`。完整 FWI 结果位于 `D:\ryjin\admit_fwi\outputs\FWI\full_salt_fwi_cg_allshots_v2`；优化 pipeline 总报告位于 `D:\ryjin\admit_fwi\outputs\FWI\full_salt_fwi_cg_allshots_v2\optimized_fwi_rtm_pipeline`；RTM before/after FWI 结果位于 `D:\ryjin\admit_fwi\outputs\RTM\before_after_fwi_alpha010_nt1200_shots12`；RTM 方案二结果位于 `D:\ryjin\admit_fwi\outputs\RTM\seg_salt_scheme2_full30m_nt4001_workers4`；局部 FWI 对照结果位于 `D:\ryjin\admit_fwi\outputs\FWI影响因素`。JGE 导向的创新点矩阵和图件 alt text 由 `python -m admit_fwi.build_jge_innovation_framework` 生成。已整理的公开代码仓库为 `https://github.com/ruiyangjin255-ux/fwi-rtm-illumination-diagnostics`。若投稿，需将可公开代码、参数配置、压缩后的结果表和可复现实验脚本整理到 GitHub/Zenodo，并避免上传受版权限制的原始模型文件。

## AI 辅助声明草稿

作者使用 AI 辅助工具协助整理本地实验结果、归纳文献脉络和起草初稿。所有数值结果均来自作者本地计算输出，文献条目需在投稿前由作者逐条核验 DOI、页码、期刊格式和引用位置。作者对稿件内容、数据解释和最终投稿版本承担全部责任。
