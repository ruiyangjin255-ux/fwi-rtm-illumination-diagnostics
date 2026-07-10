# 照明预条件 FWI 参数扫描设计

## 目标

在已有小范围盐丘 FWI 与照明预条件对比基础上，增加参数扫描实验，检验不同照明预条件稳定项 `epsilon` 对误差下降的影响，寻找是否存在优于 baseline FWI 的预条件参数。

本阶段仍保持小范围、少炮、短时间步和 CPU 可运行，不扩展到完整 SEG/Salt 多炮生产级 FWI。

## 现有基础

- baseline FWI 已能运行并输出误差曲线。
- 照明预条件 FWI 已能运行，但当前默认参数下误差下降弱于 baseline。
- 现有对比结果说明预条件更新更保守，需要进一步调参，而不是直接作为优于 baseline 的结论。

## 扫描参数

第一阶段只扫描照明预条件稳定项：

`epsilon in [0.01, 0.02, 0.05, 0.10, 0.20]`

原因：

- `epsilon` 直接控制低照明区梯度放大程度。
- 参数数量少，适合当前 Python CPU 演示。
- 先固定其他因素，避免同时扫描步长、炮数、窗口导致结果难解释。

## 输出设计

新增默认输出目录：

`D:\ryjin\admit_fwi\outputs\small_salt_fwi_illumination_scan`

预期输出：

- `summary_scan.json`
- `scan_results.csv`
- `illumination_scan_compare.png`
- 每组预条件的摘要文件和反演模型文件

`summary_scan.json` 需要记录：

- baseline 初始误差、最终误差、下降比例。
- 每个 `epsilon` 的最终误差和下降比例。
- 最优预条件参数。
- 最优预条件是否超过 baseline。

## 判读原则

- 如果某个 `epsilon` 的下降比例高于 baseline，可作为“照明预条件在该参数下改善反演”的候选结果。
- 如果所有预条件参数仍弱于 baseline，应如实写为“当前轻量照明预条件没有优于 baseline，但证明了照明预条件会改变更新尺度，仍需更严格 Hessian/步长联合优化”。
- 不把参数扫描结果解释为完整工业级 FWI 结论。

## 验证

- 单元测试验证 `epsilon` 字符串解析。
- smoke 测试验证扫描输出 `summary_scan.json` 和 `scan_results.csv`。
- 端到端运行真实盐丘裁剪模型，检查扫描图件和摘要。
- 运行已有小范围 FWI、RTM 和优化测试，确认向后兼容。
