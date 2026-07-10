# 照明预条件 FWI 二维参数扫描设计

## 目标

在已有一维 `epsilon` 扫描基础上，新增 `epsilon × max_update` 二维联合扫描，检验照明预条件改变更新尺度后，是否能通过调整单步最大速度更新量获得优于 baseline FWI 的结果。

本阶段仍保持小范围盐丘裁剪、少炮、短时间步和 CPU 可运行，不扩展完整 SEG/Salt 多炮 FWI。

## 扫描参数

默认扫描：

- `epsilon = 0.05, 0.10, 0.20, 0.50`
- `max_update = 20, 35, 50, 80`

其中：

- `epsilon` 控制低照明区梯度缩放强弱。
- `max_update` 控制每次迭代速度更新幅度。

二维扫描的核心假设是：照明预条件可能并非无效，而是需要配套不同步长约束。

## 输出设计

新增默认输出目录：

`D:\ryjin\admit_fwi\outputs\small_salt_fwi_illumination_2d_scan`

预期输出：

- `summary_2d_scan.json`
- `scan_2d_results.csv`
- `epsilon_update_heatmap.png`
- `best_preconditioned_misfit_curve.png`
- 每组参数对应的子目录摘要和模型文件

`summary_2d_scan.json` 记录：

- baseline 误差下降比例。
- 每组 `epsilon, max_update` 的最终误差和下降比例。
- 最优预条件组合。
- 最优组合是否超过 baseline。

## 判读原则

- 如果最优组合超过 baseline，可作为“照明预条件需要配套步长调节”的实验结果。
- 如果仍未超过 baseline，应说明当前轻量预条件形式在该裁剪窗口下仍偏保守，后续应考虑更严格的 Hessian 近似、频率递进或更合理的步长线搜索。
- 不将小范围二维扫描外推为完整盐丘模型结论。

## 验证

- 单元测试验证 `max_update` 参数列表解析。
- smoke 测试验证二维扫描输出 JSON 和 CSV。
- 端到端运行真实盐丘裁剪模型，检查热力图和最优组合摘要。
- 运行已有 FWI、RTM 和优化测试，确认向后兼容。
