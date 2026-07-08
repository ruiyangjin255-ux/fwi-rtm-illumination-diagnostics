# 小范围盐丘 FWI 自适应扩展线搜索设计

## 目标

在已有自适应步长线搜索基础上，新增扩展式线搜索策略。已有结果中 baseline 和照明预条件每轮都选择 `step_scale=2.0`，而 `2.0` 是候选列表上界，说明更优步长可能位于更大范围。因此需要在每轮迭代中根据候选结果自动扩大搜索范围。

本阶段仍保持小范围盐丘裁剪、少炮、短时间步和 CPU 可运行，不扩展完整 SEG/Salt 多炮 FWI。

## 方法设计

每轮迭代先测试初始候选步长：

`initial_step_scales = 0.5, 1.0, 2.0`

如果当前最佳步长是已测试候选中的最大值，并且误差低于当前模型误差，则继续测试扩展候选：

`expanded_step_scales = 3.0, 4.0, 6.0, 8.0`

扩展过程中只要新候选继续降低误差，就更新最佳步长；如果误差不再降低，则停止扩展。若所有候选都不能降低误差，则本轮保留当前模型。

该策略同时适用于 baseline 更新方向和 illumination_preconditioned 更新方向。

## 输出设计

新增默认输出目录：

`D:\ryjin\rtm_acoustic\outputs\small_salt_fwi_adaptive_line_search`

预期输出：

- `adaptive_line_search_summary.json`
- `adaptive_line_search_compare.png`
- `baseline/adaptive_line_search_summary.json`
- `baseline/adaptive_line_search_results.csv`
- `illumination_preconditioned/adaptive_line_search_summary.json`
- `illumination_preconditioned/adaptive_line_search_results.csv`

摘要需要记录：

- 每种方法的初始误差、最终误差、下降比例。
- 每轮实际测试过的 `step_scale`。
- 每轮最终选择的 `step_scale`。
- 是否触发扩展搜索。
- 照明预条件自适应线搜索是否超过 baseline 自适应线搜索。

## 判读原则

- 如果 baseline 继续选择更大的步长并显著优于固定线搜索，说明前一阶段 `step_scale=2.0` 上界过低。
- 如果照明预条件在扩展线搜索后超过 baseline，说明预条件方向主要受步长限制。
- 如果照明预条件仍低于 baseline，说明当前轻量照明预条件方向本身仍不占优，需要改进梯度构造、Hessian 近似或频率递进。

## 验证

- smoke 测试验证自适应线搜索输出 JSON 和 CSV。
- 端到端运行真实盐丘裁剪模型，检查摘要和曲线图。
- 运行已有 FWI、RTM 和优化测试，确认向后兼容。
