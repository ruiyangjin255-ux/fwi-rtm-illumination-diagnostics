# 盐丘模型小范围全波形偏移设计

## 目标

在已有盐丘速度模型和声波 RTM/FWI 代码基础上，先实现一个小范围、可快速运行的“全波形偏移”对比实验。这里的全波形偏移定义为：使用未扣除直达波的完整炮记录进行叠前逆时偏移，并与现有反射波偏移条件进行同尺度比较。

## 范围边界

本阶段不实现最小二乘逆时偏移、Born 线性化算子或弹性波全波场偏移。实验只复用已有声波二阶有限差分正演和零延迟互相关成像条件，重点观察完整炮记录进入偏移后，对盐丘局部成像振幅、低波数背景能量和照明归一化效果的影响。

## 推荐实现

新增一个独立脚本 `run_salt_full_waveform_migration.py`。脚本读取 `seg676x230.bin`，裁剪与小范围 FWI 一致的盐丘局部窗口，构造平滑速度作为迁移速度。对同一组炮点分别运行两类成像：

1. 全波形偏移：`subtract_direct_wave=False`，保留完整炮记录。
2. 反射波偏移参考：`subtract_direct_wave=True`，使用已有直达波扣除逻辑。

两类结果都保存原始成像、震源照明归一化成像、震源-检波照明归一化成像、Laplacian 增强成像和叠加记录。脚本同时输出对比图和 `summary.json`，便于后续写论文段落。

## 输出

默认输出目录为 `D:\ryjin\admit_fwi\outputs\small_salt_full_waveform_migration`。需要包含：

- `full_waveform_image.npy`
- `full_waveform_normalized_image.npy`
- `full_waveform_source_receiver_normalized_image.npy`
- `reflection_only_image.npy`
- `reflection_only_normalized_image.npy`
- `reflection_only_source_receiver_normalized_image.npy`
- `full_waveform_stacked_record.npy`
- `reflection_only_stacked_record.npy`
- `migration_compare.png`
- `stacked_record_compare.png`
- `summary.json`
- `docs/full_waveform_migration_summary.md`

## 成功标准

脚本能在小范围盐丘窗口上完成运行，所有输出数组为有限值，图件能直接查看。`summary.json` 至少记录裁剪范围、炮点、采样参数、两类成像的振幅指标、照明指标和全波形/反射波结果差异指标。

## 测试策略

新增 `tests/test_salt_full_waveform_migration.py`。测试使用小尺寸合成模型和短时间采样，验证：

- 默认裁剪和炮点配置合法。
- 全波形与反射波成像流程都能生成有限数组。
- 输出文件和 `summary.json` 字段完整。
- 中文报告文件能生成并包含“全波形偏移”和“反射波偏移”关键词。

## 风险和解释口径

完整炮记录中直达波和潜水波能量较强，直接互相关成像容易出现浅部强振幅和低波数背景，不一定比反射波偏移更“清晰”。因此本阶段的结论应表述为“全波形记录进入 RTM 成像条件后的响应对比”，不能等同于 LSRTM 或 FWI。
