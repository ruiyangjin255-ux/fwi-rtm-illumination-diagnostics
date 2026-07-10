# 小范围盐丘 FWI 自适应步长线搜索设计

## 目标

在已有小范围盐丘 FWI、照明预条件 FWI 和二维参数扫描基础上，新增自适应步长线搜索策略。该策略用于检验：固定更新尺度是否限制了 FWI 残差下降，尤其是照明预条件改变更新方向尺度后，是否需要逐迭代选择更合适的步长。

本阶段仍保持小范围盐丘裁剪、少炮、短时间步和 CPU 可运行，不扩展完整 SEG/Salt 多炮 FWI。

## 方法设计

每次迭代得到平均更新方向后，不直接更新模型，而是测试多个步长系数：

`step_scale = 0.25, 0.5, 1.0, 1.5, 2.0`

对每个候选步长：

1. 构造候选模型：`v_trial = clip(v_current + step_scale * update)`。
2. 正演所有炮点，计算总数据残差。
3. 选择误差最低的候选步长。
4. 若所有候选都不能降低误差，则保留当前模型。

该策略同时适用于 baseline 更新方向和 illumination_preconditioned 更新方向。

## 输出设计

新增默认输出目录：

`D:\ryjin\admit_fwi\outputs\small_salt_fwi_line_search`

预期输出：

- `line_search_summary.json`
- `line_search_results.csv`
- `line_search_misfit_compare.png`
- `baseline_line_search_summary.json`
- `preconditioned_line_search_summary.json`

摘要需要记录：

- 每种方法的初始误差、最终误差、下降比例。
- 每次迭代选择的最佳 `step_scale`。
- 每次迭代所有候选步长的误差。
- 照明预条件线搜索是否超过 baseline 线搜索。

## 判读原则

- 如果线搜索显著提升 baseline，说明原固定步长策略限制了收敛。
- 如果照明预条件在线搜索后超过 baseline，说明预条件方向需要合适步长配套。
- 如果照明预条件线搜索后仍低于 baseline，说明当前轻量照明预条件方向本身仍不占优，应考虑更严格 Hessian 近似、频率递进或改进梯度构造。

## 验证

- 单元测试验证 `step_scale` 参数解析。
- smoke 测试验证线搜索输出 JSON 和 CSV。
- 端到端运行真实盐丘裁剪模型，检查摘要和曲线图。
- 运行已有 FWI、RTM 和优化测试，确认向后兼容。
