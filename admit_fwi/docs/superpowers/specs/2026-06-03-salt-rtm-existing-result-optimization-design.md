# 盐丘模型现有 RTM 结果优化设计

## 目标

基于已经完成的 SEG/Salt 多炮 RTM full 输出，直接生成一组可比较的后处理优化结果，用于判断当前问题主要来自显示/后处理，还是需要进一步修改 RTM 成像条件。该阶段不重新运行 224 炮 RTM，也不修改正演和反传传播内核。

## 范围

新增独立脚本：

`D:\ryjin\admit_fwi\optimize_existing_salt_result.py`

脚本读取现有 full 输出目录：

`D:\ryjin\admit_fwi\outputs\seg_salt_multishot_rtm_padded60_full30m_workers4`

所有新结果只写入该目录下的 `optimization_compare` 子目录。

## 输入

- `migration_velocity_smooth.npy`
- `multishot_rtm_image_raw.npy`
- `multishot_rtm_illumination.npy`
- `multishot_rtm_source_normalized.npy`
- `multishot_rtm_laplacian_filtered.npy`
- `multishot_rtm_display.npy`

## 输出

- 多个候选 `.npy` 文件，包括保守显示增强、低照明区 mask、深度均衡和推荐结果。
- `optimization_compare.png`：当前结果与候选结果的总览对比图。
- `metrics.json`：振幅、深度分层、横向分区、低照明区比例和能量保留指标。
- `optimization_report.md`：中文诊断报告，说明是否应收敛到方案 1 或进入方案 2。
- `paper_figures/paper_ready_migration.png`：推荐论文图。
- `paper_figures/paper_ready_comparison.png`：当前显示、推荐论文图和增强候选对比。
- `paper_figures/paper_optimization_report.md`：中文论文图优化说明。

## 设计

优化流程保持保守。脚本把当前 RTM 输出数组作为不可变输入，只生成后处理候选：

1. 改进的对称裁剪和显示归一化。
2. 低照明区 mask，避免解释照明不足区域。
3. 深度均衡候选，用于观察深部弱同相轴，但不作为默认推荐。
4. 论文图推荐版本，采用保守显示增强，避免过度增益造成虚假连续性。

脚本同时计算简单诊断指标。如果显示候选能改善层位连续性且不改变主要反射结构，后续收敛到方案 1；如果低照明或横向能量不均仍然明显，则进入方案 2，检查成像条件和照明补偿。

## 非目标

- 不删除、覆盖或改写已有 full RTM 原始输出。
- 不重新运行 full 多炮 RTM。
- 不把公开 GitHub C/CUDA 代码直接搬进 Python 工程。
- 不修改 `acoustic_rtm.py` 的传播内核。

## 验证

- 为低照明 mask、深度均衡、指标计算、候选生成和论文图候选生成添加单元测试。
- 运行现有 RTM 测试套件。
- 用新脚本处理 full 输出目录。
- 打开 `optimization_compare.png` 和论文图 PNG 做视觉检查。
