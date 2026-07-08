# FWI Code Cleanup Design

**Date:** 2026-07-08

## Goal

清理 `D:\ryjin\fwi_visionfm` 中多余的调试、测试、试验代码，在保留 `v1-v10` 与 `raw OpenFWI + v11-v14` 两段研究主线的前提下，显著简化代码框架。

## Mainline Scope To Keep

必须保留的主线范围：

- `v1-v10` 主线 protocol 代码
- `raw OpenFWI + v11-v14` 主线代码
- 主线所需的数据转换、split、训练、评估、汇总、报告入口
- 主线结果分析与最终报告生成能力
- 最小必要测试集，用于保护主线基础模块与主要 protocol 契约

## Cleanup Target

本次清理针对三类内容：

1. 一次性调试代码
2. 临时试验/探针/availability/smoke 脚本
3. 冗余测试与阶段性包装测试

## Keep / Delete Policy

### Keep

保留以下代码资产：

- 根目录主流程文件：
  - 数据转换
  - split 构建
  - 训练入口
  - 主线评估入口
  - 汇总与报告入口
- `scripts/` 中直接服务 `v1-v14` 或 `openfwi` 主线的：
  - `build_*`
  - `run_*`
  - `report_protocol_*`
  - `summarize_protocol_*`
  - 主线 bootstrap / analysis / verify 脚本
- `tests/` 中：
  - 基础模块测试
  - protocol 最小契约测试
  - 主线 runner / matrix / summary / report 的最小保护测试

### Delete

删除以下候选：

- 名称明显表示一次性用途的脚本：
  - `*probe*`
  - `*smoke*`
  - `*preview*`
  - `*availability*`
  - `*research_progress*`
- 只服务中间包装或阶段临时导出的脚本：
  - 中文 refinement / reporting package 类
  - 临时 submission package 类
- 重复或边缘测试：
  - `*_smoke.py`
  - `*_probe.py`
  - `*_availability*.py`
  - 阶段性 audit / archive / packaging 测试
  - 明显重复的 report generation 测试
- 所有 `__pycache__` 与缓存临时文件

## Recommended Minimal Test Strategy

保留测试不再追求覆盖所有历史脚本，而是只保留三层：

1. 基础层
   - bridge
   - data
   - model
   - loss / metrics
   - split / protocol 基础约束
2. protocol 主线层
   - `v2-v14` 每段至少保留一个核心 matrix/runner/report/summary 契约测试
3. 当前主线增强层
   - `v11-v14` 的 bootstrap、report、runner、reuse/contract 类核心测试

## Expected Deletions

高概率删除候选包括但不限于：

- `foundation_smoke.py`
- `scripts/*probe*.py`
- `scripts/*smoke*.py`
- `scripts/preview_bridges.py`
- `scripts/check_*availability*.py`
- `scripts/report_v1_to_v*_research_progress.py`
- `scripts/generate_v10_*`
- `tests/test_*smoke.py`
- `tests/test_*probe.py`
- `tests/test_*availability*.py`
- `tests/test_v1_to_v*_report_generation.py`
- `tests/test_v10_*`
- `tests/__pycache__/`

## Risks

- 某些文件名虽然包含 `report` 或 `analysis`，但可能只服务一次性阶段整理，需要按主线关系人工判定
- 某些 `smoke` 名称文件可能被当作唯一入口验证用例，删除前需要确认有更正式的契约测试替代
- `v1-v10` 为历史主线，过度压缩测试可能削弱早期 protocol 的可回归性

## Execution Strategy

1. 先枚举脚本与测试文件，生成保留/删除清单
2. 人工检查边界文件，避免误删主线 runner 与主线 report
3. 删除调试/探针/一次性脚本
4. 删除冗余测试，仅保留最小主线测试集
5. 删除缓存目录
6. 对保留测试执行最小验证，确认主线结构仍可用

## Success Criteria

清理完成后应满足：

- `v1-v10` 与 `v11-v14` 主线脚本仍完整可追踪
- `scripts/` 中明显调试、probe、smoke、availability、research_progress 文件已大幅减少
- `tests/` 仅保留最小必要测试集
- 缓存目录被清掉
- 最小验证命令可以运行并通过
