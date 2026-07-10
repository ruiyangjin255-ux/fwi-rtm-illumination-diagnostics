# ADMIT-FWI 更新可接受性审计框架

## 目标

ADMIT-FWI 的目标不是让每一次 FWI 更新都通过，而是在复杂盐丘和盐下成像中回答一个更严格的问题：当前数据、频带、时窗、照明和边界条件是否足以支持把某个空间位置的速度更新用于解释。

当 full-FWI update 主要集中在浅部和盐体上方，ECG-gated update 几乎没有进入盐体及盐下区域时，框架应输出“深部更新证据不足或不可接受”，而不是把弱更新包装成有效盐下改进。

## 审计证据链

1. 数据域：检查训练炮和 held-out 炮的残差变化是否一致，避免只对参与反演的炮过拟合。
2. 像域：检查 RTM before/after 与 split consistency，确认速度更新是否改善稳定成像而不是放大噪声。
3. ROI：将盐体上方、盐体内部和盐下区域分开统计，避免浅部更新掩盖深部失败。
4. 深时窗：记录时长至少覆盖 deep-time proxy，当前目标约 `3.2-5.0 s`。
5. 边界/PML：通过 padding、顶部吸收和边界能量审计排除早到波、source/receiver imprint 和边界条带伪影。
6. 照明：检查炮数、offset、孔径和深部波场覆盖，低照明区域默认不接受强解释。

## 长时窗与多尺度生产设置

当前生产命令采用：

- `nt=5000`, `dt=0.001 s`，覆盖约 `5.0 s`；
- `4 Hz -> 6 Hz -> 8 Hz` 多尺度推进；
- `pad_x=60`, `pad_top=40`, `pad_bottom=80`，先降低 PML 风险；
- audit fold 隔离，避免用审计炮参与训练；
- shot-group diagnostics，用于检查不同炮组证据是否一致；
- `--workers 2` 并行炮计算，兼顾本机 CPU 和磁盘压力。

脚本入口：

```powershell
powershell -ExecutionPolicy Bypass -File admit_fwi\scripts\run_deep_time_multiscale_fwi_production.ps1
```

## Figure 解释规范

若 Figure 1 中 full-FWI update 主要集中在浅部和盐体上方，caption 应明确：

> The full-FWI update is concentrated mainly in the shallow section and around the upper salt region, while the subsalt region remains weakly updated.

若 ECG-gated update 接近全灰，应明确：

> The ECG-gated update is strongly suppressed under the current admissibility criteria.

这类图的作用是支持 ADMIT-FWI 的审计结论：计算出的更新不能自动进入深部或盐下解释，必须先通过数据域、像域、ROI、深时窗和照明审计。

## 当前结论边界

- 可支持：ADMIT-FWI 能识别当前更新主要停留在浅部/盐上，并阻止证据不足的盐下更新被过度解释。
- 不应声称：当前 ECG gate 已经显著改善盐下速度模型。
- 下一步证据：完整 168 炮长时窗多尺度 FWI、更多模型适用性测试和最终图件结果展示。
