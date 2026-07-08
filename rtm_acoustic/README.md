# SEG/Salt 模型声波逆时偏移 RTM

本目录实现了一个独立的 Python 声波逆时偏移流程，用于现有
`seg676x230.bin` 复杂盐丘速度模型。代码重点服务于 SEG/Salt 模型的
正演、反传、零延迟互相关成像、照明归一化和论文图输出。

## 理论与实现对应

当前实现参考王春艳高阶交错网格有限差分论文最后一章中的 RTM 思路：

- 使用声波方程进行震源波场正向外推；
- 将地表记录作为接收波场反传的边界/震源条件；
- 提供王式边界反传偏移结果，即把地表记录反传到地下并保存成像剖面；
- 提供现代 RTM 常用的零延迟互相关成像结果；
- 以现有复杂构造盐丘模型作为成像目标。

实现中也吸收了刘伟 VSP RTM 文献中的实用处理：

- 除原始互相关成像外，同时输出震源照明归一化成像；
- 正传和反传均使用吸收边界；
- 震源波场时间历史使用磁盘 memmap 保存，避免一次性占用大量内存；
- 提供重复拉普拉斯滤波，用于压制 RTM 低频噪声。

当前程序使用标量声压声波方程，有限差分阶数可配置。这样可以独立于
已有 C 程序运行，同时保留高阶有限差分 RTM 的主要流程。

## 主要文件

- `acoustic_rtm.py`：有限差分核、模型/记录读写、正演、反传 RTM、
  照明归一化、模型 padding、成像裁剪和拉普拉斯滤波。
- `run_seg_salt_rtm.py`：单炮/基础 SEG-Salt RTM 一键运行脚本。
- `run_multishot_rtm.py`：真正的叠前多炮 RTM。每炮独立正演，保存震源
  波场，再用对应接收记录反传并与震源波场做零延迟互相关，最后多炮叠加。
- `plot_rtm_result.py`：速度、记录、照明和 RTM 成像的绘图辅助脚本。
- `tests/test_acoustic_rtm.py`：小模型回归测试。

## 基础 RTM 运行

在 `D:\ryjin` 下运行：

```powershell
python -m rtm_acoustic.run_seg_salt_rtm --output-dir rtm_acoustic\outputs\seg_salt_rtm --fd-order 8 --laplacian-power 2
```

主要输出：

- `rtm_image_raw.npy` 和 `.bin`：零延迟互相关原始成像；
- `rtm_image_source_normalized.npy` 和 `.bin`：震源照明归一化成像；
- `wang_boundary_migration_image.npy` 和 `.bin`：王式边界反传偏移剖面；
- `wang_boundary_migration_laplacian_filtered.npy` 和 `.bin`：滤波后的边界反传剖面；
- `rtm_image_laplacian_filtered.npy` 和 `.bin`：滤波后的互相关 RTM 成像；
- `seg_salt_rtm_panel.png`：汇总图；
- `source_wavefield_float32.dat`：磁盘保存的震源波场时间历史。

运行测试：

```powershell
python -m pytest rtm_acoustic\tests\test_acoustic_rtm.py -q
```

## 多炮零延迟互相关 RTM

论文式边界反传流程适合复现王式图件，但不是严格的叠前 RTM。若要运行
真正的多炮零延迟互相关 RTM，可使用：

```powershell
python -m rtm_acoustic.run_multishot_rtm --output-dir rtm_acoustic\outputs\seg_salt_multishot_rtm --shot-spacing 30 --fd-order 8 --laplacian-power 1
```

快速 smoke test：

```powershell
python -m rtm_acoustic.run_multishot_rtm --output-dir rtm_acoustic\outputs\seg_salt_multishot_rtm_smoke --nt 600 --max-shots 3 --shot-spacing 300 --fd-order 8 --laplacian-power 1
```

## 全物理模型成像 padding 版本

如果直接在 `676 x 230` 原始模型上运行，`absorb_cells=40` 会占用左右和底部
边界，导致这些区域照明不足，不能作为有效成像范围解释。为了让原始物理模型
尽量完整成像，应把吸收边界放到模型外侧 padding 区中，再把最终结果裁回原始
物理窗口。

推荐命令：

```powershell
python -m rtm_acoustic.run_multishot_rtm --output-dir rtm_acoustic\outputs\seg_salt_multishot_rtm_padded60 --shot-spacing 30 --fd-order 8 --laplacian-power 1 --pad-x 60 --pad-bottom 60 --workers 4 --smooth-radius-x 20 --smooth-radius-z 20 --smooth-passes 3 --min-illumination-fraction 0.02 --direct-mute-padding 0.02 --migration-depth-power 0.0 --migration-clip-percentile 99.0 --migration-trace-balance 0.0 --migration-output-clip 0.95
```

该命令会：

- 左右各加 `60` 个网格、底部加 `60` 个网格；
- padding 区使用边界速度复制，避免产生人工速度突变；
- 在扩展后的 `796 x 290` 网格上做正演和 RTM；
- 将炮点和检波点横向坐标整体右移 `60` 个网格；
- 保存标准输出前，将 RTM 图像和记录裁回原始 `676 x 230` 物理模型范围；
- 额外保存 `migration_velocity_smooth_padded.npy`，用于检查扩展后的迁移速度。

## 加速 full 版本

`run_multishot_rtm.py` 支持 `--workers` 并行多炮 RTM：

```powershell
python -m rtm_acoustic.run_multishot_rtm --output-dir rtm_acoustic\outputs\seg_salt_multishot_rtm_padded60_parallel --shot-spacing 30 --fd-order 8 --laplacian-power 1 --pad-x 60 --pad-bottom 60 --workers 4 --smooth-radius-x 20 --smooth-radius-z 20 --smooth-passes 3 --min-illumination-fraction 0.02 --direct-mute-padding 0.02 --migration-depth-power 0.0 --migration-clip-percentile 99.0 --migration-trace-balance 0.0 --migration-output-clip 0.95
```

并行方式说明：

- `--workers 1` 使用原串行流程；
- `--workers 4` 同时计算 4 炮，通常能显著缩短 full 224 炮运行时间；
- 每个 worker 会使用独立的临时震源波场文件，完成该炮成像后自动删除；
- 日志会输出 `Finished shot 当前/总数`，便于估计剩余时间；
- worker 数越多，磁盘 I/O、内存和临时波场空间压力越大。若机器卡顿，可改为
  `--workers 2` 或 `--workers 3`。

## 重要参数说明

- 默认震源和检波器深度为 `z=4`，不是 `z=1`。这是因为 8 阶有限差分需要
  4 个网格的模板半径。
- `--pad-x`、`--pad-top`、`--pad-bottom` 用边界速度复制扩展模型。正演和反传
  在扩展模型上进行，标准 RTM 输出会裁回原始物理模型尺寸。
- 默认会对直达波进行 mute。只有明确需要完整炮记录时，才使用 `--no-direct-mute`。
- `--shot-spacing 30` 对当前模型约对应 `224` 炮，完整多炮 RTM 很耗时。
- `--max-shots` 可用于预览或调参。例如 `--max-shots 60` 会从全炮集中均匀抽取
  60 炮。
- `--workers` 用于并行计算不同炮点。建议 full padding 版本先用 `4`；如果内存或
  磁盘 I/O 压力较大，降到 `2`。

## 多炮 RTM 主要输出

- `multishot_rtm_image_raw.npy`：多炮叠加的零延迟互相关原始成像；
- `multishot_rtm_illumination.npy`：累计震源照明；
- `multishot_rtm_source_normalized.npy`：照明归一化成像；
- `multishot_rtm_laplacian_filtered.npy`：拉普拉斯滤波后的最终 RTM 成像；
- `multishot_rtm_display.npy`：显示增强后的迁移剖面；
- `multishot_record_and_rtm.png`：叠加记录与 RTM 剖面对比图；
- `run_parameters.json`：本次运行的模型、padding、炮点和显示参数记录。
