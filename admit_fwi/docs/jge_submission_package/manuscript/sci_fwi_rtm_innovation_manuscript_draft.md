# 面向盐丘构造的 FWI 速度更新、RTM 照明诊断与成像条件对比：一项可复现的声波数值实验研究

> 初稿版本：2026-07-05  
> 建议定位：SCI 4 区以上地球物理/计算地学类期刊初稿；投稿前需按目标期刊模板重排，并再次核验 JCR/中科院分区、参考文献格式和图件分辨率。  
> 本稿主证据来自 `D:\ryjin\admit_fwi` 已有 FWI/RTM 结果；`learning-based extension workspace` 结果仅作为可选扩展方向，不作为本文主实验结论。

## 摘要

复杂盐丘构造中的速度强横向变化会同时影响全波形反演（full waveform inversion, FWI）的模型更新与逆时偏移（reverse time migration, RTM）的构造成像质量。现有研究通常分别讨论 FWI 的速度建模能力或 RTM 的照明补偿效果，而对“速度更新、照明诊断和成像条件选择”三者之间的可复现实验关系讨论不足。本文基于 SEG/Salt 速度模型建立二维声波数值实验流程，首先在完整 `676 x 230` 网格模型上开展多炮 FWI，随后利用初始平滑速度和模型质量门控后的反演速度进行多炮 RTM before/after 对比，并进一步分析震源照明归一化、源-检几何照明归一化和 Laplacian 成像条件。实验结果表明，完整盐丘模型 FWI 在 224 炮、3 次迭代条件下使平均残差由 `4.443590e-05` 降至 `2.045601e-05`，下降比例为 `53.9651%`；但全量更新的模型 MAE 仅改善 `0.1119%`，edge MAE 和梯度 MAE 未同步改善。基于模型质量门控的更新尺度扫描选择 `alpha=0.1`，其 MAE 改善 `0.1304%`、RMSE 改善 `0.0215%`，且 edge MAE 小幅改善 `0.0040%`。12 炮、`nt=1200` 的 RTM before/after 对比显示，采用阻尼 FWI 更新后，Laplacian-filtered RTM 图像相对真值速度参考的 RMSE 由 `0.027130` 小幅降至 `0.027109`。RTM 方案二实验显示，源-检几何照明低于最大值 1% 的网格比例约为 `0.0174`，源-检归一化与震源归一化图像相关系数为 `0.9732`。局部 `70 x 120` 盐丘窗口的 FWI 对照进一步显示，自适应扩展线搜索将 baseline 残差下降比例提升至 `6.36905%`，优于当前轻量震源照明预条件的 `4.13861%`。本文的主要贡献是提出并验证一个轻量、可复现的 FWI-RTM 联合诊断框架，用数据残差、模型结构误差、偏移图像指标、照明比例和局部线搜索对照共同约束盐丘成像解释。

**关键词：** 全波形反演；逆时偏移；盐丘模型；照明补偿；成像条件

## Abstract

Strong lateral velocity contrasts in salt structures affect both full waveform inversion (FWI) velocity updating and reverse time migration (RTM) imaging. This study builds a reproducible two-dimensional acoustic workflow on the SEG/Salt model to connect FWI misfit reduction, model-quality diagnostics, RTM before/after imaging, and illumination-based imaging-condition analysis. In the full `676 x 230` salt experiment, 224-shot, three-iteration FWI reduces the average data misfit from `4.443590e-05` to `2.045601e-05` (`53.9651%`). However, the full update improves model MAE by only `0.1119%`, and edge and gradient errors do not improve. A model-quality gate therefore selects a damped update scale of `alpha=0.1`, improving MAE by `0.1304%`, RMSE by `0.0215%`, and edge MAE by `0.0040%`. In a 12-shot, `nt=1200` RTM validation, the Laplacian-filtered image RMSE relative to the true-velocity RTM reference decreases slightly from `0.027130` to `0.027109`. Full-aperture imaging-condition diagnostics show that only `0.0174` of grid points fall below 1% of maximum source-receiver illumination, while source-receiver and source-normalized images have a correlation of `0.9732`. The results support a conservative FWI-RTM diagnostic framework constrained by data misfit, model structural error, migrated-image metrics, and illumination diagnostics.

**Keywords:** full waveform inversion; reverse time migration; salt model; illumination compensation; imaging condition

## 1 引言

盐丘构造是反射地震成像中的典型难题。盐体与围岩之间的强速度差异会造成波场传播路径复杂化、局部照明不均衡以及盐下反射能量弱化，从而影响速度建模和构造成像。FWI 通过最小化观测记录与模拟记录之间的波形差异直接更新速度模型，是提高复杂构造速度场精度的重要手段；RTM 则通过双程波动方程正反传和成像条件恢复反射结构，是复杂构造偏移成像中的常用方法。

从方法关系看，FWI 和 RTM 并不是互相替代的流程。FWI 回答“当前速度模型是否能够解释观测波形”，RTM 回答“在给定速度模型下反射界面如何成像”。更准确的工程关系是：FWI 改善速度模型，改进后的速度模型服务于 RTM，RTM 输出更可靠的构造图像。问题在于，照明补偿常被同时用于 RTM 图像后处理和 FWI 梯度预条件，但二者作用对象不同。RTM 中常见的震源照明归一化或源-检照明归一化主要作用于偏移图像振幅；FWI 中的照明预条件则作用于模型更新方向或 Hessian 近似。因此，若不区分二者，很容易把“偏移图像亮度更均衡”误写成“反演照明优化更有效”。

本文围绕这一问题开展可复现实验：在完整 SEG/Salt 模型上建立声波 FWI 与 RTM 联合流程，并用局部盐丘窗口对步长选择和照明预条件进行受控对照。研究问题包括：

1. 完整盐丘模型上的短迭代 FWI 是否能形成明确的数据残差下降？
2. full-aperture RTM 图像是否主要受严重照明缺失控制？
3. 震源照明归一化、源-检几何照明归一化和 Laplacian 成像条件的差异如何定量解释？
4. 在局部 FWI 中，简单照明预条件是否优于 baseline，还是步长策略才是主要控制因素？

## 2 相关工作与文献依据

FWI 的基本思想是通过迭代最小化完整波形数据残差恢复地下介质参数。Tarantola 的反演理论和后续 FWI 综述奠定了波形反演作为非线性优化问题的基础；Virieux 和 Operto 对 FWI 的历史、现状和挑战进行了系统总结，指出初始模型、低频信息、周期跳跃和 Hessian/预条件是控制 FWI 成败的关键因素。

RTM 最早由 Baysal、Kosloff 与 Sherwood 等提出，通过反传记录波场并与震源波场成像，在复杂速度结构下具有处理多路径和陡倾反射的优势。随后 RTM 成像条件、照明补偿和最小二乘 RTM（least-squares RTM, LSRTM）逐步发展。照明补偿和 Hessian 近似的核心作用是缓解采集几何、传播路径和波场能量分布造成的成像/反演不均衡，但不同方法的作用层级并不相同：图像归一化主要改造偏移剖面，Hessian 或梯度预条件才直接影响反演更新。

近年深度学习 FWI 和 OpenFWI 数据集推动了端到端速度反演研究。OpenFWI 提供了多个二维合成数据族，用于评估跨构造泛化能力。视觉基础模型和地震基础模型也开始被尝试用于地震记录到速度模型的迁移学习。但是，从本项目已有 `learning-based FWI extension` 结果看，真实 DINOv2 与 LoRA/spectrogram bridge 目前仍处于 CPU smoke 或小样本阶段，只能证明技术路径可行，不能替代本文物理 FWI/RTM 的主证据。

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

## 4 结果

### 4.1 完整盐丘模型 FWI 结果

完整 `676 x 230` 盐丘模型 FWI 使用 224 炮、3 次迭代、`nt=900` 和 `f0=8 Hz`。实验结果显示，初始残差为 `4.443590e-05`，最终残差为 `2.045601e-05`，残差下降比例为 `53.9651%`。该结果说明当前完整模型 FWI 正演、残差反传、炮流式 checkpoint 和速度更新链路已经形成可运行闭环。

该结论应谨慎解释。新增模型质量评估显示，虽然数据残差下降明显，但全量速度更新本身的改进很弱：全模型 MAE 仅由 `131.6921 m/s` 降至 `131.5447 m/s`，改善比例为 `0.1119%`；RMSE 改善比例仅为 `0.0100%`。更重要的是，盐体边界相关的 edge MAE 变化为 `-0.0893%`，梯度 MAE 变化为 `-3.6250%`，说明当前 3 次迭代全量更新没有改善关键构造边界。因此，程序加入更新尺度门控：在 `alpha = 0, 0.1, 0.25, 0.5, 0.75, 1.0` 中选择满足 MAE/RMSE 正改善、edge MAE 不退化且梯度退化受限的阻尼更新。当前最优可接受尺度为 `alpha=0.1`，对应 MAE 改善 `0.1304%`、RMSE 改善 `0.0215%`、edge MAE 改善 `0.0040%`。该结果将本文定位为 FWI-RTM 诊断与质量门控框架，而不是高精度 FWI 速度反演方法。

### 4.2 RTM before/after FWI 对比

为检验 FWI 更新是否实际改善后续偏移成像，本文使用同一真值观测数据分别以真值速度、初始平滑速度和质量门控后的阻尼 FWI 速度进行 RTM。修正版正式对比采用 `nt=1200, f0=20 Hz, pad_x=60, pad_bottom=60`，选取 12 个横向均匀分布炮点，输出见 Figure 2。主要指标如下：

| 指标 | 初始速度 RTM | 阻尼 FWI 速度 RTM |
|---|---:|---:|
| 与真值速度 RTM 的 Laplacian-filtered RMSE | `0.027130` | `0.027109` |
| 与真值速度 RTM 的 Laplacian-filtered MAE | `0.009072` | `0.009044` |
| 与真值速度 RTM 的 Laplacian-filtered 相关系数 | `0.4929` | `0.4927` |
| 与真值速度 RTM 的 source-normalized RMSE | `2.958475` | `2.962629` |
| 炮点数量 | `12` | `12` |

按照 Laplacian-filtered RTM 图像 RMSE，质量门控后的阻尼 FWI 速度相对初始速度略微更接近真值速度参考，RMSE 改善比例为 `0.0769%`。该改善幅度很小，不能被夸大为显著成像提升；但它与全量更新 `alpha=1.0` 使 RMSE 恶化 `6.9498%` 的结果形成对照，说明模型质量门控可以避免把有害 FWI 更新直接传递给 RTM。Figure 2 因此应作为本文的关键程序优化结果：数据残差下降必须经过模型边界指标和偏移图像指标筛选，才能进入后续 RTM 成像链路。

### 4.3 RTM 方案二成像条件对比

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

### 4.4 局部 FWI 步长与照明预条件对照

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

### 5.3 机器学习先验可作为扩展，不宜作为本文主证据

`learning-based FWI extension` 中已有 OpenFWI 小样本和视觉基础模型迁移实验。它们显示 LoRA + spectrogram bridge 可在目标族数值误差上出现小幅改善，但结构指标未同步改善，真实 DINOv2 仍是 smoke 级。本文可在讨论中提出一个扩展方向：用机器学习模型输出低波数速度先验或结构约束，再进入物理 FWI/RTM 验证。但在没有严格 GPU 规模、固定 split、跨族测试和 PDE 一致性约束前，不应把这些结果写成本文主创新。

## 6 可形成的创新点

### 创新点 1：FWI 速度更新与 RTM 照明诊断的联合评价框架

本文不是单独展示 FWI 或 RTM，而是把完整盐丘 FWI 残差下降、RTM 成像条件对比、照明比例诊断和局部 FWI 机制对照放在同一复现框架中。该创新点适合作为论文主线，风险较低，因为已有本地结果支撑。

### 创新点 2：区分图像照明归一化与反演梯度预条件

已有结果明确显示 RTM 照明归一化和 FWI 照明预条件作用对象不同。本文可把这一点写成方法论贡献：给出同一模型下的图像层照明补偿和反演层照明预条件对照，避免把两类结果混为一谈。

### 创新点 3：以负结果约束照明优化宣称

局部 FWI 结果显示轻量照明预条件并未超过 baseline 自适应线搜索。该负结果可以转化为可信贡献：当前盐丘窗口中，步长选择是更强控制因素；照明预条件若要成为主优化策略，需要 Hessian 近似或多尺度策略支持。SCI 论文中诚实的消融和失败边界通常比过度包装更容易通过方法审查。

### 创新点 4：成像条件选择的定量证据化

源-检归一化与源归一化的相关系数、低照明比例和 Laplacian 相关性共同说明，不同成像条件并非只影响“显示好看与否”，而改变了低波数背景和界面表达。该点可支撑一组论文图和表：原始 RTM、源归一化、源-检归一化、Laplacian 增强和照明分布。

### 可选创新点 5：机器学习低波数先验与物理成像闭环

若后续补充实验，可把 `learning-based FWI extension` 的 OpenFWI/LoRA/spectrogram 结果作为低波数初始模型生成器，与本文 FWI/RTM 闭环结合。当前不建议写入主结果，只适合放在展望或后续工作。

## 7 结论

本文基于 SEG/Salt 模型建立了一个可复现的声波 FWI-RTM 联合实验流程。完整盐丘模型 FWI 在 224 炮、3 次迭代下实现 `53.9651%` 的数据残差下降，证明速度更新链路可运行；但全量更新的模型质量评估显示 MAE 仅改善 `0.1119%`，edge MAE 和梯度 MAE 均未改善。程序优化后，更新尺度门控选择 `alpha=0.1`，使 edge MAE 由退化转为小幅改善；进一步的 12 炮 RTM before/after FWI 对比表明，阻尼 FWI 速度对应的 Laplacian-filtered RTM 图像相对真值速度参考的 RMSE 由 `0.027130` 小幅降至 `0.027109`。RTM 方案二结果显示源-检几何照明低于最大值 1% 的网格比例仅约 `0.0174`，源-检归一化与震源归一化高度相关，说明当前 full-aperture 成像并非主要受严重照明缺失控制。局部 FWI 对照表明，自适应扩展线搜索比当前轻量照明预条件更能改善收敛；因此，本文建议将照明归一化、梯度预条件和步长选择作为三个不同层级分别讨论。

本文的核心结论是：在当前盐丘数值实验中，RTM 照明归一化适合用于成像振幅均衡和条件对比，FWI 的主要实证收益来自速度更新闭环与自适应步长选择；简单震源照明预条件虽能改善更新稳定性，但尚不足以作为主要优化结论。后续工作可在更严格 Hessian 预条件、多尺度 FWI、LSRTM 和机器学习低波数先验方面扩展。

## 8 图表计划

| 图号 | 内容 | 已有/建议路径 |
|---|---|---|
| 图 1 | FWI 模型质量门控与更新尺度选择 | `D:\ryjin\admit_fwi\docs\jge_main_figures\figure1_fwi_quality_gate.tiff` |
| 图 2 | RTM before/after 质量门控 FWI 速度更新验证 | `D:\ryjin\admit_fwi\docs\jge_main_figures\figure2_rtm_before_after_validation.tiff` |
| 图 3 | full-aperture RTM 成像条件与照明诊断 | `D:\ryjin\admit_fwi\docs\jge_main_figures\figure3_imaging_condition_diagnostics.tiff` |
| 图 4 | 局部 FWI 消融与保守结论边界 | `D:\ryjin\admit_fwi\docs\jge_main_figures\figure4_local_fwi_claim_boundary.tiff` |
| 表 1 | 完整 FWI 与 RTM 参数表 | 由 `full_salt_fwi_summary.md` 和 `scheme2_report.md` 整理 |
| 表 2 | RTM before/after FWI 图像指标 | 见本文第 4.2 节 |
| 表 3 | 局部 FWI 策略对比 | 见本文第 4.4 节 |

## 9 建议投稿方向

优先考虑计算地学、应用地球物理数值方法或工程地震成像方向。候选期刊可包括 `Computers & Geosciences`、`Journal of Applied Geophysics`、`Acta Geophysica`、`Geophysical Prospecting` 等。具体是否满足“SCI 4 区以上”应在投稿前按最新 JCR/中科院分区核验；不同年份和不同分区体系可能不一致。

## 参考文献（待按目标期刊格式重排）

1. Tarantola, A. (1984). Inversion of seismic reflection data in the acoustic approximation. *Geophysics*, 49(8), 1259-1266. https://doi.org/10.1190/1.1441754
2. Baysal, E., Kosloff, D. D., & Sherwood, J. W. C. (1983). Reverse time migration. *Geophysics*, 48(11), 1514-1524. https://doi.org/10.1190/1.1441434
3. Virieux, J., & Operto, S. (2009). An overview of full-waveform inversion in exploration geophysics. *Geophysics*, 74(6), WCC1-WCC26. https://doi.org/10.1190/1.3238367
4. Zhu, H., & McMechan, G. A. (2012). Seismic interferometry-based imaging conditions for prestack reverse-time migration. *Geophysics*, 77(3), S77-S86. https://doi.org/10.1190/geo2011-0340.1
5. Dai, W., Fowler, P., & Schuster, G. T. (2012). Multi-source least-squares reverse time migration. *Geophysical Prospecting*, 60(4), 681-695. https://doi.org/10.1111/j.1365-2478.2012.01092.x
6. Deng, C., Feng, S., Wang, H., Zhang, X., Jin, P., Feng, Y., Zeng, Q., Chen, Y., & Lin, Y. (2022). OpenFWI: Large-scale multi-structural benchmark datasets for full waveform inversion. *Advances in Neural Information Processing Systems, Datasets and Benchmarks Track*, 35. https://openreview.net/forum?id=7w-a8PYPlP
7. Oquab, M., Darcet, T., Moutakanni, T., et al. (2024). DINOv2: Learning robust visual features without supervision. *Transactions on Machine Learning Research*. https://openreview.net/forum?id=a68SUt6zFt
8. Wang, B., Feng, S., Fu, L.-Y., & Hu, L. (2023). Least squares reverse time migration imaging with illumination compensation. *Scientific Reports*, 13. https://doi.org/10.1038/s41598-023-40578-8

## 数据与代码可用性声明草稿

本文使用的代码、参数文件和主要输出位于本地项目 `D:\ryjin\admit_fwi`。推荐复现入口为 `python -m admit_fwi.run_optimized_fwi_rtm_pipeline`；若需要重新执行耗时 RTM 验证，可添加 `--run-rtm`。投稿材料打包入口为 `python -m admit_fwi.build_jge_submission_package`，输出目录为 `D:\ryjin\admit_fwi\docs\jge_submission_package`。完整 FWI 结果位于 `D:\ryjin\admit_fwi\outputs\FWI\full_salt_fwi_cg_allshots_v2`；优化 pipeline 总报告位于 `D:\ryjin\admit_fwi\outputs\FWI\full_salt_fwi_cg_allshots_v2\optimized_fwi_rtm_pipeline`；RTM before/after FWI 结果位于 `D:\ryjin\admit_fwi\outputs\RTM\before_after_fwi_alpha010_nt1200_shots12`；RTM 方案二结果位于 `D:\ryjin\admit_fwi\outputs\RTM\seg_salt_scheme2_full30m_nt4001_workers4`；局部 FWI 对照结果位于 `D:\ryjin\admit_fwi\outputs\FWI影响因素`。若投稿，需将可公开代码、参数配置、压缩后的结果表和可复现实验脚本整理到 GitHub/Zenodo，并避免上传受版权限制的原始模型文件。

## AI 辅助声明草稿

作者使用 AI 辅助工具协助整理本地实验结果、归纳文献脉络和起草初稿。所有数值结果均来自作者本地计算输出，文献条目需在投稿前由作者逐条核验 DOI、页码、期刊格式和引用位置。作者对稿件内容、数据解释和最终投稿版本承担全部责任。
