# 小规模盐丘模型验证总结与全范围成像建议

## 1. 验证目的

本轮小规模验证的目标是：在进入 `676 x 230` 全范围盐丘模型成像之前，先用 `70 x 120` 局部窗口检查已有 RTM、FWI、照明归一化、全波形偏移和直达波静音流程是否稳定，并据此给出全范围成像的参数建议。

当前局部窗口统一采用：

- 裁剪位置：`z0=70, x0=310`
- 裁剪尺寸：`nz=70, nx=120`
- 炮点：`[15, 60, 104]`
- 时间采样：`nt=450, dt=0.001 s`
- 震源主频：`f0=10 Hz`
- 震源/检波点深度：`z=4`

## 2. 小规模 RTM/FWI 验证结论

### 2.1 小规模 FWI 可运行，但基础步长偏保守

基础小规模 FWI 的数据残差从 `0.2963276605` 降到 `0.2936459879`，下降比例为 `0.90497%`。这说明正演、残差反传和速度更新链路已经打通，但固定更新幅度下收敛较慢。

自适应扩展线搜索明显改善收敛，基础更新分支残差下降比例达到 `6.36905%`，最终残差为 `0.2774544110`。三次迭代均选择最大测试步长 `8.0`，说明当前小规模模型仍处在可接受较大步长的范围内。

结论：后续若做全范围 FWI 或迭代式成像，应优先保留线搜索机制，而不是只依赖固定 `max_update`。

### 2.2 照明预条件改善有限，不能替代线搜索

照明预条件扫描中，`epsilon=0.2` 的残差下降约 `0.44077%`，二维扫描最佳组合为 `epsilon=0.5, max_update=20`，下降约 `0.54712%`，仍低于基础固定更新分支的 `0.90497%`。

在线搜索条件下，照明预条件分支下降比例提升到 `1.08500%`；自适应扩展线搜索下提升到 `4.13861%`。但同条件下基础更新分支分别为 `1.77648%` 和 `6.36905%`。

结论：照明预条件可以作为梯度均衡手段保留，但当前实现不应作为主控收敛策略。全范围任务中应先做好 RTM 成像、静音和显示参数，再考虑把照明预条件用于 FWI。

### 2.3 全波形偏移与反射波偏移差异明确

小规模全波形偏移定义为：完整炮记录不扣除直达波，直接进入叠前声波 RTM。反射波偏移参考为：扣除平滑模型直达波后再 RTM。

主要指标：

| 指标 | 全波形偏移 | 反射波偏移 |
| --- | ---: | ---: |
| 原始成像 99% 振幅 | `4.841339e+04` | `6.852943e+04` |
| 源照明归一化 99% 振幅 | `6.230579e+01` | `8.202710e+01` |
| 源-检波照明归一化 99% 振幅 | `6.970317e-01` | `5.882304e-01` |
| 低照明比例 | `0.401190` | `0.401190` |
| 叠加记录 RMS | `4.405694e-01` | `4.874512e-01` |

全波形与反射波原始成像差异 L2 为 `1.417003e+04`，全波形/反射波 99% 振幅比为 `0.70646`。

结论：完整记录进入 RTM 后会改变浅部和低波数背景响应，但当前小规模窗口中，反射波参考的界面振幅更强。全范围成像不建议只保留“未扣直达波”的全波形结果，应同时输出反射波偏移、照明归一化和 Laplacian 增强结果。

### 2.4 直达波静音有效但作用温和

直达波静音扫描采用 `padding_time=[0,0.01,0.02,0.03]` 和 `taper_time=[0.01,0.02]`，共 8 组。最佳组合为：

- `padding_time=0.030 s`
- `taper_time=0.020 s`
- 估计直达波速度：`2070.47 m/s`

关键指标：

| 指标 | 未静音全波形 | 最佳静音全波形 | 反射波参考 |
| --- | ---: | ---: | ---: |
| 原始成像 99% 振幅 | `4.841339e+04` | `4.791420e+04` | `6.852943e+04` |
| 源照明归一化 99% 振幅 | `6.230579e+01` | `6.153365e+01` | `8.202710e+01` |
| 叠加记录 RMS | `4.405694e-01` | `4.394718e-01` | `4.874512e-01` |
| 浅部平均绝对振幅 | `1.395039e+04` | `1.381380e+04` | `1.646751e+04` |

结论：当前静音参数会稳定降低浅部和整体强能量，但变化幅度不大。它适合作为全范围 RTM 的默认保护参数，而不是主要成像增强方法。照明归一化和 Laplacian 仍必须输出。

## 3. 对全范围盐丘模型成像的参数建议

### 3.1 推荐先做三阶段全范围成像

第一阶段：全模型少炮预览。

目的：确认 padding、炮点、直达波静音和显示参数没有明显问题。

```powershell
cd D:\ryjin
python -m rtm_acoustic.run_multishot_rtm --output-dir rtm_acoustic\outputs\seg_salt_full_preview_60shots --shot-spacing 30 --max-shots 60 --fd-order 8 --laplacian-power 1 --pad-x 60 --pad-bottom 60 --workers 4 --smooth-radius-x 20 --smooth-radius-z 20 --smooth-passes 3 --min-illumination-fraction 0.02 --direct-mute-padding 0.03 --direct-mute-taper 0.02 --migration-depth-power 0.0 --migration-clip-percentile 99.0 --migration-trace-balance 0.0 --migration-output-clip 0.95
```

第二阶段：全炮反射波 RTM。

目的：作为论文主成像结果候选。保留直达波扣除和静音，使用 padding 后裁回物理模型。

```powershell
cd D:\ryjin
python -m rtm_acoustic.run_multishot_rtm --output-dir rtm_acoustic\outputs\seg_salt_full_reflection_rtm_padded60 --shot-spacing 30 --fd-order 8 --laplacian-power 1 --pad-x 60 --pad-bottom 60 --workers 4 --smooth-radius-x 20 --smooth-radius-z 20 --smooth-passes 3 --min-illumination-fraction 0.02 --direct-mute-padding 0.03 --direct-mute-taper 0.02 --migration-depth-power 0.0 --migration-clip-percentile 99.0 --migration-trace-balance 0.0 --migration-output-clip 0.95
```

第三阶段：全炮全波形 RTM 对照。

目的：对比完整记录参与成像时的浅部和背景响应。该结果不建议单独作为主图，应作为方法对照。

```powershell
cd D:\ryjin
python -m rtm_acoustic.run_multishot_rtm --output-dir rtm_acoustic\outputs\seg_salt_full_fullwave_rtm_padded60 --shot-spacing 30 --fd-order 8 --laplacian-power 1 --pad-x 60 --pad-bottom 60 --workers 4 --smooth-radius-x 20 --smooth-radius-z 20 --smooth-passes 3 --min-illumination-fraction 0.02 --no-direct-subtract --direct-mute-padding 0.03 --direct-mute-taper 0.02 --migration-depth-power 0.0 --migration-clip-percentile 99.0 --migration-trace-balance 0.0 --migration-output-clip 0.95
```

### 3.2 推荐保留的输出图件

全范围成像至少保留以下结果：

- 原始互相关成像：用于判断真实振幅分布和噪声背景。
- 震源照明归一化成像：用于比较照明补偿效果。
- 源-检波照明归一化成像：用于观察双向照明补偿是否过度压制弱照明区。
- Laplacian 或高阶 Laplacian 成像：用于论文图展示和界面解释。
- 叠加记录图：用于说明直达波、反射波和静音参数是否合理。
- `run_parameters.json`：用于论文复现实验参数。

### 3.3 参数选择依据

- `pad-x=60, pad-bottom=60`：避免吸收边界侵占物理成像区域。
- `workers=4`：已有代码支持并行多炮，适合作为默认；若磁盘或内存压力大，降为 `2`。
- `smooth-radius-x=20, smooth-radius-z=20, smooth-passes=3`：保持迁移速度平滑，降低直达波扣除和 RTM 反传的不稳定。
- `min-illumination-fraction=0.02`：小规模验证中低照明比例约 `0.401`，需要低照明门限避免归一化放大边界伪影。
- `direct-mute-padding=0.03, direct-mute-taper=0.02`：来自小规模静音扫描最佳组，适合作为全范围初始值。
- `laplacian-power=1`：全范围 RTM 建议先用 1，避免过强高通导致深部弱反射被削弱。

## 4. 下一步建议

优先执行第一阶段 `60` 炮预览。预览通过后，再跑全炮反射波 RTM。全波形 RTM 对照应放在反射波主结果之后，用于解释完整记录、直达波静音和照明补偿之间的差异。

如果全范围预览中出现浅部强背景或盐丘下方阴影明显，下一轮应只扫描以下少量参数：

- `direct-mute-padding`: `0.02,0.03,0.04`
- `direct-mute-taper`: `0.02,0.03`
- `min-illumination-fraction`: `0.01,0.02,0.05`
- `laplacian-power`: `1,2`

暂不建议立即进入全范围 FWI。小规模验证显示，FWI 的有效改进主要来自线搜索，而照明预条件本身不是当前最强收益来源。应先获得稳定、可解释的全范围 RTM 主图，再决定是否做更昂贵的全范围 FWI 或 LSRTM。

## 5. 对应输出路径

- 小规模 FWI：`D:\ryjin\rtm_acoustic\outputs\small_salt_fwi_demo`
- 自适应线搜索 FWI：`D:\ryjin\rtm_acoustic\outputs\small_salt_fwi_adaptive_line_search`
- 小规模全波形偏移：`D:\ryjin\rtm_acoustic\outputs\small_salt_full_waveform_migration`
- 小规模直达波静音扫描：`D:\ryjin\rtm_acoustic\outputs\small_salt_full_waveform_mute_scan`
- 综合图：`D:\ryjin\rtm_acoustic\outputs\paper_summary\paper_fwi_rtm_summary.png`
