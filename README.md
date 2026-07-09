# ADMIT-FWI Update Admissibility Audit Framework

本仓库当前聚焦 **ADMIT-FWI 更新可接受性审计框架**：在复杂盐丘 FWI-RTM 工作流中，不直接把全量 FWI 更新解释为可信地质改进，而是通过数据域、像域、ROI、深时窗和照明证据对更新进行可接受性审计。

核心问题是：当 full-FWI update 主要集中在浅部和盐体上方，而盐体内部及盐下区域更新很弱时，哪些更新可以进入成像解释，哪些只能作为“当前采集和频带条件下不宜接受”的证据。

## Current Scope

- SEG/Salt 复杂盐丘模型的声波正演、RTM、FWI 与诊断流程。
- 多证据更新可接受性审计，包括 ECG/admissibility gate、held-out shot audit、RTM split consistency、ROI/deep-time coverage、boundary/PML risk audit。
- 长记录、低频起始、多尺度 FWI 的生产命令框架。
- 面向论文结果组织的 figure、caption、方法和结论边界材料。

## Repository Layout

- `rtm_acoustic/`：主要 Python 实现，包含声波 RTM、完整盐丘 FWI、审计诊断和绘图脚本。
- `rtm_acoustic/configs/`：ADMIT-FWI、盐丘更新门控、深时窗和 PML padding 预检配置。
- `rtm_acoustic/diagnostics/`：更新可信度、照明、深时窗、边界能量、held-out audit 等诊断模块。
- `rtm_acoustic/scripts/`：生产/预检脚本，包括长时窗多尺度 FWI PowerShell 命令。
- `rtm_acoustic/docs/` 与 `docs/`：论文草稿、图件说明、JGE 修订材料和核心结果摘要。

大型运行输出、`npy/bin/dat` 模型数组和本地中间文件默认不提交到 GitHub；需要时按文档命令本地复现。

## Quick Checks

```powershell
python -m py_compile rtm_acoustic\run_full_salt_fwi.py rtm_acoustic\scripts\run_deep_wavefield_smoke.py
python -m pytest rtm_acoustic\tests\test_acoustic_rtm.py -q
```

## Deep-Time PML Preflight

先检查长记录时窗和 PML padding 风险：

```powershell
python rtm_acoustic\scripts\run_deep_wavefield_smoke.py --config rtm_acoustic\configs\deep_time_preflight_pml_pad_v1.yaml --shots 3
```

该配置使用 `nt=5000`、`dt=0.001 s`、`pad_x=60`、`pad_top=40`、`pad_bottom=80`，用于降低顶部/边界条带被误判为有效地质更新的风险。

## Production Multiscale FWI

完整 168 炮训练集的多尺度长时窗流程由以下脚本封装：

```powershell
powershell -ExecutionPolicy Bypass -File rtm_acoustic\scripts\run_deep_time_multiscale_fwi_production.ps1
```

脚本按 `4 Hz -> 6 Hz -> 8 Hz` 推进，并启用 PML padding、audit fold 隔离、shot-group diagnostics 和 `--workers 2` 并行炮计算。输出写入 `rtm_acoustic\outputs\FWI\...`，这些结果目录不进入 Git。

## Interpretation Boundary

当前框架的结论边界不是“ECG gate 一定提升模型”，而是：

1. full-FWI update 在短记录或高频起始条件下容易集中于浅部和盐体上方；
2. ECG/admissibility gate 在当前证据不足区域会强抑制更新；
3. 盐体内部和盐下解释必须通过 deep-time、照明、ROI、held-out shot 和 RTM consistency 审计后才能接受；
4. 若审计未通过，应把结果报告为“更新不可接受或证据不足”，而不是强行展示为深部成像改进。

## GitHub Migration Note

本次仓库主页已从旧的 `fwi_visionfm` 方向切换为 ADMIT-FWI 更新可接受性审计框架。旧项目文件如需保留，应放在单独归档分支；当前主线只保留与 ADMIT-FWI/RTM/FWI 审计相关的源码、配置、文档和可复现实验命令。
