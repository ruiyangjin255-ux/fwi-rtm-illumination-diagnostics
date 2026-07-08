# FWI Project Cleanup Design

**Date:** 2026-06-26

## Goal

在不破坏当前研究主线复现能力的前提下，显著瘦身 `D:\ryjin\fwi_visionfm`，删除无实际研究价值的历史产物、中间缓存、重复训练图和旁支实验输出。

## Current Mainline

根据 `README.md`，当前研究主线保留范围为：

- raw OpenFWI `.npy` 数据链路
- `manifest / split / train_stats`
- `run_foundation_experiment.py` 为核心的主线训练与评估程序
- `protocol_v11`
- `protocol_v12`
- `protocol_v13`
- `protocol_v14`
- `openfwi_*` 主线实验目录
- 主线复现所需 `weights / checkpoints`

以下路线视为非主线或历史兼容资产：

- `legacy npz matrix route`
- `protocol_v1` 及更早 legacy 产物
- 旁支 probe、临时归档、重复中间缓存

## Cleanup Policy

### Must Keep

- 主线代码、配置、测试、脚本
- 主线原始数据目录与主线 family 数据
- `splits / manifest / stats / index`
- 主线复现所需 `weights / checkpoints`
- 主线最终报告 `*.md`
- 主线汇总结果 `summary / aggregate / bootstrap / comparison` 类 `csv/json`
- 所有 qualitative prediction grids `*.png`

### Safe To Delete

- `feature_cache/**/*.npz`
- `predictions_*.npz`
- 训练过程中的中间数组缓存
- `*.txt` 训练日志
- 重复 seed 的非汇总图
- 训练曲线、调参图、重复诊断图
- `legacy_*`
- `ncs_probe`
- 临时 archive 和历史 research progress 报告目录
- Python 缓存目录与测试缓存目录

### Out Of Scope

- 不删除主线 `weights`
- 不删除主线 `checkpoints`
- 不修改源码逻辑
- 不删除 `v11-v14` 与 `openfwi_*` 目录中的最终报告、汇总表和 qualitative prediction grids

## Recommended Execution Strategy

采用“研究主线清理”策略：

1. 先删除全项目缓存目录，如 `__pycache__`、`.pytest_cache`
2. 删除 `outputs` 下明确非主线目录
3. 对 `protocol_v11`、`protocol_v12`、`protocol_v13`、`protocol_v14` 与 `openfwi_*` 目录执行基于文件模式的选择性清理
4. 保留最终报告、汇总结果、关键图和 qualitative prediction grids
5. 删除大体积可再生缓存，尤其 `protocol_v14_geometry_aware_trace_bridge/feature_cache`
6. 删除 `predictions_*.npz` 和训练日志
7. 做收尾校验，确认主线目录仍存在，关键 `md/csv/json/png` 保留

## Verification Criteria

清理完成后，必须满足：

- `protocol_v11-v14` 主线目录仍存在
- `openfwi_*` 主线目录仍存在
- `weights` 仍存在
- `data/splits` 仍存在
- 主线目录中的 `*.md`、汇总 `csv/json` 和 qualitative prediction grids `*.png` 仍存在
- 明确要删除的 `feature_cache/**/*.npz`、`predictions_*.npz`、训练日志已消失

## Risks

- 仅靠文件名模式区分“关键图”和“训练曲线/调参图”时，可能存在少量边界模糊文件
- `openfwi_*` 目录内部若混有不规范命名的关键图，需优先以“保留 png”而不是“按图名筛掉”来降低误删风险
- `protocol_v14` 体积主要来自 `feature_cache`，删除后若需要复算特征缓存，后续会重新生成

## Decision

执行清理时采用以下边界：

- 保留 `v14` 作为主线
- 保留主线 `weights/checkpoints`
- 主线目录中保留所有 qualitative prediction grids
- 删除训练曲线、调参图、重复 seed 非汇总图、中间 `npz` 缓存、训练日志和非主线目录
