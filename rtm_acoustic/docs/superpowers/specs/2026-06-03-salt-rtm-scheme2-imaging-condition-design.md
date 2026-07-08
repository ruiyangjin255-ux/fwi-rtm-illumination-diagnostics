# 盐丘模型 RTM 方案 2 成像条件优化设计

## 目标

在不直接移植公开 C/CUDA RTM 代码、不重写传播内核的前提下，为现有 Python 声波 RTM 增加更稳健的物理成像候选：接收波照明、源-检双照明归一化和 Laplacian 成像条件对比。先通过小规模 smoke 实验判断这些物理成像改动是否值得用于 full 224 炮重跑。

## 设计原则

- 保持现有 `normalized_image` 和 `filtered_image` 语义不变，避免破坏已有结果和测试。
- 新增物理成像候选字段，而不是替换旧输出。
- full 输出不被覆盖；方案 2 输出写入新的 smoke 对比目录。
- 方案 2 第一阶段只跑小规模实验，不直接重跑 full。

## 核心改动

1. 在 `reverse_time_migrate` 反传时累计接收波照明：
   - `receiver_illumination += receiver_wavefield ** 2`
2. 新增双照明归一化：
   - `source_receiver_normalized = image / sqrt(source_illumination * receiver_illumination)`
3. 新增 Laplacian 成像候选：
   - `laplacian_image = high_order_laplacian_filter(image, dx, dz, power=1)`
   - `laplacian_normalized_image = source_normalized_image(laplacian_image, source_illumination, ...)`
4. 多炮结果累加上述候选：
   - 接收照明逐炮累加。
   - 双照明归一化在所有炮累加后计算。
   - Laplacian 成像候选在所有炮累加后计算。

## 输出

新增方案 2 对比脚本：

`D:\ryjin\rtm_acoustic\run_scheme2_imaging_condition_compare.py`

默认输出到：

`D:\ryjin\rtm_acoustic\outputs\seg_salt_scheme2_smoke`

输出内容：

- `scheme2_source_normalized.npy`
- `scheme2_source_receiver_normalized.npy`
- `scheme2_laplacian_image.npy`
- `scheme2_laplacian_source_normalized.npy`
- `scheme2_receiver_illumination.npy`
- `scheme2_compare.png`
- `scheme2_report.md`
- `scheme2_parameters.json`

## 验证

- 单元测试验证接收照明、双照明归一化和 Laplacian 候选字段存在且有限。
- 串行与并行多炮结果仍一致。
- 运行方案 2 smoke 脚本并检查图件。
- 运行现有 RTM 测试和方案 1 测试，确保向后兼容。
