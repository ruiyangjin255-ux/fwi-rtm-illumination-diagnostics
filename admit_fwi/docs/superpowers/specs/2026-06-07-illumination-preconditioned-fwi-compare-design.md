# 小范围盐丘 FWI 照明预条件对比设计

## 目标

在已完成的小范围 SEG/Salt FWI 演示基础上，新增一个照明预条件 FWI 对比实验。该实验用于说明：RTM 中的照明归一化是成像域幅值补偿，而 FWI 中的照明预条件作用在模型更新方向上，二者目标和作用位置不同。

本阶段仍保持小范围、少炮、短时间步和 CPU 可运行，不扩展到完整 676 x 230 网格、224 炮生产级 FWI。

## 现有基础

- 已有基础 FWI 脚本：`D:\ryjin\admit_fwi\run_small_salt_fwi.py`。
- 已有基础输出目录：`D:\ryjin\admit_fwi\outputs\small_salt_fwi_demo`。
- 已有中文说明：`D:\ryjin\admit_fwi\docs\small_salt_fwi_and_illumination.md`。
- 基础 FWI 使用残差反传与正传波场相关构造更新方向，再做平滑、裁剪和速度范围限制。

## 设计原则

- 不改变现有 baseline FWI 默认行为，避免破坏已生成结果和测试。
- 新增照明预条件作为显式选项，而不是隐式替换原更新方向。
- 输出 baseline 与 illumination_preconditioned 两套结果，便于论文图件对比。
- 所有新增说明、图题和字段解释统一采用中文。

## 照明预条件方法

照明预条件采用正传震源波场能量近似：

`L_s(x,z) = sum_t u_s(t,x,z)^2`

对残差反传得到的更新方向进行缩放：

`g_pre(x,z) = g(x,z) / (L_s_norm(x,z) + epsilon)`

其中 `L_s_norm` 为按最大值归一化后的震源照明，`epsilon` 用于避免低照明区数值不稳定。缩放后仍执行平滑、幅值裁剪和速度范围限制。

该方法是论文照明预处理思想的轻量演示版本，不声明为完整 Hessian 预条件或工业级照明补偿 FWI。

## 输出设计

新增对比输出目录：

`D:\ryjin\admit_fwi\outputs\small_salt_fwi_illumination_compare`

预期输出：

- `baseline_summary.json`
- `preconditioned_summary.json`
- `summary_compare.json`
- `baseline_misfit_curve.png`
- `preconditioned_misfit_curve.png`
- `fwi_method_compare.png`
- `baseline_inverted_model.npy`
- `preconditioned_inverted_model.npy`

`summary_compare.json` 记录两种方法的初始误差、最终误差、误差下降比例、迭代次数、炮点和裁剪窗口。

## 中文说明更新

更新 `D:\ryjin\admit_fwi\docs\small_salt_fwi_and_illumination.md`，补充以下差异：

- RTM 照明归一化：对偏移图像 `I` 做幅值均衡。
- FWI 照明预条件：对模型更新方向 `g` 做尺度调整。
- 对比实验的判读重点不是图像是否更亮，而是数据残差是否下降、模型更新是否更稳定。

## 验证

- 单元测试验证照明场计算有限且非负。
- 单元测试验证照明预条件不会改变数组形状，且能避免除零。
- smoke 测试验证 baseline 与照明预条件对比输出 `summary_compare.json`。
- 端到端运行真实盐丘裁剪模型，检查两组输出图件和摘要存在。
- 运行已有小范围 FWI、RTM 和成像优化测试，确认向后兼容。
