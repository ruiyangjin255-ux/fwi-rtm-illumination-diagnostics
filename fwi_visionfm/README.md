# FWI-VisionFM 研究原型

当前仓库包含两条并行链路：

1. NumPy-only 原型：用于快速验证 `多炮记录 -> bridge -> placeholder backbone -> 聚合 -> 速度解码 -> loss/metrics -> smoke` 主流程。
2. PyTorch baseline：用于验证最小可训练闭环，后续作为接入 DINOv2 / MAE / SAM / LoRA 的前置骨架。
3. Frozen Vision Foundation baseline：用于验证 `pseudo image -> frozen DINOv2 encoder -> 聚合 -> 可训练速度 decoder` 这一路是否可以在 CPU smoke 上闭环。

## 当前推荐主流程：OpenFWI raw .npy foundation transfer

当前主线不再是 `convert_openfwi -> npz -> matrix`，而是直接基于原始 OpenFWI `.npy` 批文件做小样本视觉基础模型迁移实验：

1. `build_openfwi_index.py` 生成 `openfwi_manifest.csv`
2. `make_openfwi_splits.py` 生成 `train/val/test_in_family/test_cross_family/smoke` split
3. `compute_openfwi_stats.py` 只基于 `train.csv` 计算 `train_stats.json`
4. `run_foundation_experiment.py` 直接消费 raw OpenFWI `.npy + manifest + split + stats`
5. 在同一入口下运行 `dummy / timm scratch / frozen / adapter / lora`

当前已验证数据口径：

- 真实数据根目录示例：`D:\data\openfwi`
- 当前使用 family：`FlatVel_A`、`CurveVel-A`
- 当前 manifest 样本数：`2000`
- data shape：`(5,1000,70)`
- model shape：`(1,70,70)`
- train family：`FlatVel_A`
- cross-family test family：`CurveVel-A`

建议把这个流程视为当前推荐主线：

- `Current recommended`：
  raw OpenFWI `.npy` + manifest/split/stats + `run_foundation_experiment`
- `Legacy / compatibility`：
  `convert_openfwi -> npz -> run_experiment_matrix`
- `Post-training / no new training`：
  result indexing、checkpoint evaluation、prediction visualization、stage report

重要说明：

- `dummy_dinov2` 不代表真实 DINOv2。
- 当前 1 epoch CPU 小样本结果只代表工程闭环和小样本验证，不代表最终科研精度。
- `pretrained=true` 若下载失败，先用 `pretrained=false` 验证链路。
- 不要把 validation 指标直接等同于严格 target test 指标。
- cross-family 结论必须来自固定 `test_cross_family` split。

## Protocol route decision

当前项目保留两条历史路线，但定位不同：

- `legacy npz matrix route`：`convert_openfwi -> npz -> run_experiment_matrix`，保留为工程验证资产，用于追溯早期数据转换、shape guard、matrix runner、Protocol v1 后处理和 smoke 结果。
- `raw OpenFWI route`：`raw OpenFWI .npy -> manifest/split/stats -> run_foundation_experiment -> checkpoint-only evaluation -> report/figures/comparison`，作为当前正式研究主线。

详细决策记录见：

```powershell
D:\ryjin\fwi_visionfm\docs\protocol_route_decision.md
```

legacy results should not be directly compared with main protocol results unless split/stats/test protocol are verified consistent. 查看 legacy 资产索引用：

```powershell
python -m fwi_visionfm.index_legacy_outputs
```

该命令只生成 `outputs/legacy_index.csv` 和 `outputs/legacy_index.md`，不会把 legacy 指标并入正式 `experiment_comparison.csv`。

## Verified OpenFWI small transfer results

以下结果来自当前已经验证通过的 raw OpenFWI 小样本 CPU 迁移实验，只能作为工程闭环和小样本验证，不作为最终科研结论。

### 指标结果

| experiment | final train loss | best val loss | val MAE | val RMSE |
| --- | ---: | ---: | ---: | ---: |
| dummy_openfwi_smoke | 0.1102607673 | 0.0581603569 | 0.1895457860 | 0.2347925939 |
| timm_vit_tiny_scratch_openfwi | 0.0383925565 | 0.0260466797 | 0.1327375144 | 0.1565880176 |
| timm_vit_tiny_frozen_openfwi | 0.0328172026 | 0.0141496381 | 0.0912285304 | 0.1156941409 |
| timm_vit_tiny_adapter_openfwi | 0.0299904421 | 0.0112967854 | 0.0794445442 | 0.1023873095 |
| timm_vit_tiny_lora_openfwi | 0.0277607361 | 0.0132719045 | 0.0864513382 | 0.1106666981 |

### 参数量结果

| experiment | total params | trainable params | trainable ratio |
| --- | ---: | ---: | ---: |
| dummy_openfwi_smoke | 1523876 | 1292324 | 0.8480506288 |
| timm_vit_tiny_scratch_openfwi | 6804900 | 6804900 | 1.0 |
| timm_vit_tiny_frozen_openfwi | 6804900 | 1308708 | 0.1923184764 |
| timm_vit_tiny_adapter_openfwi | 6955044 | 1458852 | 0.2097545321 |
| timm_vit_tiny_lora_openfwi | 6952356 | 1456164 | 0.2094489983 |

最直接的结论是：

- raw OpenFWI `.npy` 小样本迁移链路已经跑通；
- frozen / Adapter / LoRA 已经形成可复现实验骨架；
- 当前结果还不能当作最终论文性能，只能作为 CPU 条件下的小样本工程验证。

## Latest stage result: bridge × LoRA interaction

最新阶段结果已经单独归档到：

- [D:\ryjin\fwi_visionfm\outputs\final_openfwi_vision_transfer_stage_report.md](D:\ryjin\fwi_visionfm\outputs\final_openfwi_vision_transfer_stage_report.md)

当前最重要的更新是：

- `LoRA + raw_spectrogram` 在 `3 seed` 下改善了 target-family `MAE / RMSE`。
- 这种改善没有同步体现在 `edge / laplacian / gradient` 结构指标上。
- Qualitative prediction grids suggest that the gain mainly comes from lower target-family numerical/background error rather than improved boundary or gradient recovery.
- 因此现阶段只能表述为目标域数值误差改善，不能表述为全面结构恢复能力提升。
- 这也不是最终 DINOv2 结论。

## Real DINOv2 smoke

当前已经完成真实 `timm DINOv2` 最小接口验证：

- attempted backbone: `vit_small_patch14_dinov2.lvd142m`
- frozen smoke: success
- LoRA smoke: success

需要注意：

- 该模型不能直接使用 `224`，当前实际 smoke 配置切换到了 `518`
- smoke 只说明真实 DINOv2 backbone 已经成功接入当前 raw OpenFWI `.npy` foundation transfer 框架
- smoke 不等于正式 DINOv2 实验
- 当前正式结论仍然来自 `vit_tiny_patch16_224` 小样本协议
- Frozen / LoRA DINOv2 checkpoints were also evaluated on small fixed in-family and cross-family subsets as checkpoint-only smoke; these results verify evaluation compatibility but are not formal DINOv2 generalization results.
- Full fixed-test checkpoint-only evaluation was also run on the smoke-trained frozen / LoRA DINOv2 checkpoints. Because the checkpoints were trained only on smoke_train=32, these results validate evaluation compatibility over the full fixed test split but should not be interpreted as formal DINOv2 performance.
- Real DINOv2 + LoRA + raw_spectrogram smoke was completed on CPU within 2 hours. It slightly reduced cross-family MAE/RMSE compared with raw_repeat3, but structural metrics became worse; the result is a smoke-level confirmation of the LoRA x spectrogram trend, not a formal DINOv2 benchmark.

详细记录见：

- [D:\ryjin\fwi_visionfm\outputs\real_dinov2_smoke\real_dinov2_smoke_report.md](D:\ryjin\fwi_visionfm\outputs\real_dinov2_smoke\real_dinov2_smoke_report.md)

## Final CPU-stage archive

- final archive path:
  `D:\ryjin\fwi_visionfm\outputs\final_stage_archive`
- CPU-stage conclusion:
  `LoRA + raw_spectrogram shows target-family numerical error gain but not structural recovery gain.`
- real DINOv2 result remains smoke-level.

## PyTorch CPU 安装

当前 baseline 只要求 CPU 版 PyTorch，不依赖 CUDA。

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

如需后续接入 Vision FM，再额外安装：

```powershell
pip install timm transformers
```

## 运行命令

NumPy smoke：

```powershell
python -m fwi_visionfm.cli
```

PyTorch smoke：

```powershell
python -m fwi_visionfm.cli torch-smoke
```

## Current recommended: raw OpenFWI small transfer workflow

当前推荐直接使用原始 OpenFWI `.npy` 数据，而不是先转为 `.npz`。典型流程如下：

```powershell
python -m fwi_visionfm.data.build_openfwi_index --root D:\data\openfwi --output-dir D:\ryjin\fwi_visionfm\outputs\openfwi_index

python -m fwi_visionfm.data.make_openfwi_splits --manifest D:\ryjin\fwi_visionfm\outputs\openfwi_index\openfwi_manifest.csv --output-dir D:\ryjin\fwi_visionfm\splits\openfwi_small --train-family auto --train-size 500 --val-size 100 --test-size 100 --cross-family-size 100 --seed 2026

python -m fwi_visionfm.data.compute_openfwi_stats --manifest D:\ryjin\fwi_visionfm\outputs\openfwi_index\openfwi_manifest.csv --split D:\ryjin\fwi_visionfm\splits\openfwi_small\train.csv --output D:\ryjin\fwi_visionfm\outputs\openfwi_index\train_stats.json --max-samples 500
```

五组已验证的 1 epoch CPU 小样本入口都走同一个训练器：

```powershell
python -m fwi_visionfm.run_foundation_experiment --openfwi-root D:\data\openfwi --manifest D:\ryjin\fwi_visionfm\outputs\openfwi_index\openfwi_manifest.csv --train-split D:\ryjin\fwi_visionfm\splits\openfwi_small\smoke_train.csv --val-split D:\ryjin\fwi_visionfm\splits\openfwi_small\smoke_val.csv --stats-file D:\ryjin\fwi_visionfm\outputs\openfwi_index\train_stats.json --model-type foundation_fwi --backbone-type dummy --backbone-name dummy_dinov2 --pretrained false --transfer-mode frozen --image-size 112 --epochs 1 --batch-size 2 --max-train-samples 32 --max-val-samples 16 --device cpu --output-dir D:\ryjin\fwi_visionfm\outputs\dummy_openfwi_smoke
```

把 `backbone-type / backbone-name / transfer-mode / peft` 替换为 `timm scratch / frozen / adapter / lora` 即可复用同一流程。

## CPU-only research workflow

当前仓库已经提供一套只依赖 CPU 的最小研究工作流，用于产出 baseline/smoke 级结果、曲线图和阶段报告，而不是追求最终精度。

推荐按下面顺序执行：

```powershell
python -m fwi_visionfm.cli smoke
python -m fwi_visionfm.cli torch-smoke
python -m fwi_visionfm.cli torch-cpu-experiment
python -m fwi_visionfm.plot_torch_experiment --input outputs/torch_cpu_experiment
python -m fwi_visionfm.report_torch_cpu_experiment --input outputs/torch_cpu_experiment
python scripts/validate_fwi_visionfm.py
```

说明：

- `torch-cpu-experiment` 会输出 `training_history.csv`、`metrics.json`、`experiment_summary.json`、`prediction.npy`、`target.npy`、`input_seismic.npy`。
- `plot_torch_experiment` 会输出 `loss_curve.png` 与 `prediction_vs_target.png`。
- `report_torch_cpu_experiment` 会输出可直接归档的 `report.md`。
- `scripts/validate_fwi_visionfm.py` 只校验 `fwi_visionfm` 自己的 smoke 与相关测试，不会调用全局 `python -m pytest -q`。

当前验收口径需要明确：

- `D:\ryjin` 根目录下的全局 `python -m pytest -q` 可能因为兄弟项目 `D:\ryjin\rtm_acoustic` 的 matplotlib/OpenMP 崩溃而失败。
- 这不属于 `fwi_visionfm` 当前 CPU-only 验收范围。
- `dummy_dinov2` 仍只是工程占位接口，不代表真实 Vision Foundation Model。
- 当前所有 CPU-only 输出都只代表 subset/baseline/smoke 级工程验证。

## Smoke-scale ablation workflow

当前仓库还提供一条 CPU 条件下的小规模模块对照实验流程，用于比较 bridge / aggregation / decoder 的最小设计差异。

```powershell
python -m fwi_visionfm.cli torch-ablation
python -m fwi_visionfm.plot_ablation --input outputs/torch_ablation
python -m fwi_visionfm.report_ablation --input outputs/torch_ablation
```

说明：

- 该实验用于 CPU 条件下的小规模模块对照。
- 它只服务于 smoke-scale ablation 和阶段报告，不代表最终 OpenFWI 大规模结果。
- 当前流程不下载任何 foundation model 权重。

## OpenFWI small-scale CPU workflow

当前仓库提供一个与 `torch_cpu_experiment` 同口径的 OpenFWI-style 小规模 CPU 实验入口。它使用本地已有的 OpenFWI raw data、sample-level split CSV 和 train stats 文件，不会自动下载数据，也不会下载 foundation model 权重。

```powershell
python -m fwi_visionfm.cli openfwi-small-experiment --data-root <path> --split-dir <path> --stats-file <path>
python -m fwi_visionfm.plot_torch_experiment --input outputs/openfwi_small_experiment
python -m fwi_visionfm.report_torch_cpu_experiment --input outputs/openfwi_small_experiment
python -m fwi_visionfm.summarize_experiments
```

说明：

- `openfwi-small-experiment` 默认使用 ablation 阶段选出的配置：`bridge=channel_stack`、`aggregation=max`、`decoder=bounded`。
- 它适用于 CPU 条件下的小样本真实数据链路验证。
- 它不应与大规模 GPU 训练结果等同。
- 如果本地缺少 `data-root`、`split-dir` 或 `stats-file`，程序会直接报清晰错误，不会自动下载 OpenFWI。

## OpenFWI scale study on CPU

当前仓库提供同口径的 OpenFWI 规模递增实验入口，用于在 CPU 条件下观察训练样本数变化时的误差趋势。它基于 `openfwi-small-experiment` 的同一配置，不自动下载 OpenFWI，不下载 foundation model，也不处理 `D:\ryjin\rtm_acoustic`。

```powershell
python -m fwi_visionfm.cli openfwi-scale-study --data-root D:\data\openfwi --split-dir D:\ryjin\fwi_visionfm\splits\openfwi_small --stats-file D:\ryjin\fwi_visionfm\outputs\openfwi_index\train_stats.json --epochs 3 --sizes 8:2:2,32:8:8,64:8:8

python -m fwi_visionfm.plot_openfwi_scale_study --input outputs/openfwi_scale_study

python -m fwi_visionfm.report_openfwi_scale_study --input outputs/openfwi_scale_study

python -m fwi_visionfm.summarize_experiments
```

说明：

- 该实验用于 CPU 条件下观察样本规模变化趋势。
- 所有配置固定为 smoke-scale ablation 的最优组合：`channel_stack + max + bounded`。
- 该实验只用于趋势观察和流程验证，不代表大规模 GPU OpenFWI 训练结果。

## Evaluation metrics

当前 CPU-only 实验与 OpenFWI small-scale / scale-study 流程统一输出以下指标：

- `MAE`：速度值绝对误差，直接反映预测速度与真值的平均偏差。
- `RMSE`：速度值均方根误差，对较大偏差更敏感。
- `relative MAE`：相对绝对误差，便于比较不同速度量级下的误差比例。
- `relative RMSE`：相对均方根误差，反映归一化后的整体偏差。
- `PSNR`：峰值信噪比，从图像重建角度衡量速度图相似性。
- `SSIM`：结构相似性，反映速度结构、层状边界和整体纹理的一致性。
- `velocity gradient error`：速度梯度误差，反映界面、断层和构造边界的预测偏差。

说明：

- `MAE/RMSE` 更偏重数值误差。
- `PSNR/SSIM` 更偏重图像结构相似性。
- `gradient error` 更偏重速度界面与构造边界恢复质量。
- 当前这些指标用于 CPU 小规模实验、smoke-scale ablation 和 OpenFWI small-scale / scale-study 的统一比较，不代表最终大规模科研结论。

## Legacy / compatibility: npz baseline and matrix workflow

下面这些命令保留用于兼容早期 `npz`-based baseline、shape guard、matrix 和转换工具，但它们不是当前 raw OpenFWI foundation transfer 的推荐主线。

更接近旧版数据集实验的 PyTorch baseline：

```powershell
python -m fwi_visionfm.run_torch_experiment --data-dir D:\ryjin\fwi_visionfm\outputs\converted_tiny --output-dir D:\ryjin\fwi_visionfm\outputs\torch_baseline --depth 6 --width 7 --channels raw,offset --aggregation source_attention --batch-size 2 --epochs 3 --learning-rate 1e-3
```

## Legacy foundation baseline examples

这一部分主要保留早期 foundation baseline 示例命令，方便兼容旧输出目录与早期 `npz` 数据流程。当前真正推荐的主线仍然是上面的 raw OpenFWI `.npy` transfer 流程。

这一阶段已经补齐“真实视觉基础模型迁移”最小实验骨架，统一支持：

- `dummy`：离线占位 backbone，用于 CPU smoke 与无网环境。
- `timm`：真实 ViT / DINOv2 类 backbone，例如 `vit_tiny_patch16_224`、`vit_small_patch16_224`、`vit_small_patch14_dinov2.lvd142m`。
- `hf_dinov2`：Hugging Face `facebook/dinov2-small` 路径，支持 `--local-files-only`。

当前仍未接入 SAM、MAE、可微 physics consistency。

可选依赖安装：

```powershell
pip install timm
```

`transformers` 不是项目硬依赖，只在 Hugging Face DINOv2 路径中需要：

```powershell
pip install transformers
```

CPU 推荐先跑 `dummy` 和 `timm vit_tiny`。DINOv2 权重较大，CPU 只用于接口验证；无网络时使用 `--pretrained false` 或 `--local-files-only true`。再次强调：`dummy_dinov2` 只是工程占位，不代表真实 DINOv2 迁移结论。

当前建议输入尺寸：

- `dummy_dinov2`：推荐 `--image-size 64`，用于 CPU smoke。
- 真实 `timm` DINOv2：默认建议 `--image-size 518`。
- CPU 上真实 DINOv2 只建议做 `1 epoch` 接口验证，不建议作为常规训练配置。

离线 CPU smoke 推荐使用 `dummy_dinov2`，不依赖联网下载真实权重：

```powershell
python -m fwi_visionfm.run_foundation_experiment --data-dir D:\ryjin\fwi_visionfm\outputs\converted_tiny --output-dir D:\ryjin\fwi_visionfm\outputs\dinov2_frozen_smoke --depth 6 --width 7 --foundation-backbone dummy_dinov2 --no-pretrained --freeze-backbone --image-size 64 --aggregation source_attention --batch-size 2 --epochs 2 --learning-rate 1e-3 --device cpu
```

ViT from scratch CPU smoke：

```powershell
python -m fwi_visionfm.run_foundation_experiment --data-dir D:\ryjin\fwi_visionfm\outputs\converted_tiny --output-dir D:\ryjin\fwi_visionfm\outputs\vit_tiny_from_scratch --depth 6 --width 7 --backbone-type timm --model-name vit_tiny_patch16_224 --no-pretrained --image-size 224 --aggregation source_attention --batch-size 2 --epochs 1 --learning-rate 1e-3 --device cpu
```

如果本地 `timm` 可用，并且你接受真实 DINOv2 权重加载，则可以运行：

```powershell
python -m fwi_visionfm.run_foundation_experiment --data-dir D:\ryjin\fwi_visionfm\outputs\converted_tiny --output-dir D:\ryjin\fwi_visionfm\outputs\dinov2_frozen --depth 6 --width 7 --foundation-backbone vit_small_patch14_dinov2.lvd142m --pretrained --freeze-backbone --image-size 518 --aggregation source_attention --batch-size 1 --epochs 1 --learning-rate 1e-4 --device cpu
```

Hugging Face DINOv2 本地缓存接口 smoke：

```powershell
python -m fwi_visionfm.run_foundation_experiment --data-dir D:\ryjin\fwi_visionfm\outputs\converted_tiny --output-dir D:\ryjin\fwi_visionfm\outputs\hf_dinov2_smoke --depth 6 --width 7 --backbone-type hf_dinov2 --model-name facebook/dinov2-small --pretrained --freeze-backbone --local-files-only --image-size 224 --aggregation source_attention --batch-size 1 --epochs 1 --learning-rate 1e-4 --device cpu
```

## Legacy LoRA PEFT Baseline

当前已实现标准 LoRA 与最小 Adapter，不实现 SAM、MAE、QLoRA、rsLoRA、AdaLoRA，也不接 physics consistency 到训练。

dummy_dinov2 + LoRA CPU smoke：

```powershell
python -m fwi_visionfm.run_foundation_experiment --data-dir D:\ryjin\fwi_visionfm\outputs\converted_tiny --output-dir D:\ryjin\fwi_visionfm\outputs\dummy_dinov2_lora_smoke --depth 6 --width 7 --foundation-backbone dummy_dinov2 --no-pretrained --freeze-backbone --image-size 64 --aggregation source_attention --batch-size 2 --epochs 3 --learning-rate 1e-3 --device cpu --peft lora --lora-rank 4 --lora-alpha 8 --lora-dropout 0.0 --lora-target-modules qkv,proj,fc1,fc2
```

timm DINOv2 + LoRA no-pretrained 1 epoch：

```powershell
python -m fwi_visionfm.run_foundation_experiment --data-dir D:\ryjin\fwi_visionfm\outputs\converted_tiny --output-dir D:\ryjin\fwi_visionfm\outputs\dinov2_lora_518_no_pretrained_1ep --depth 6 --width 7 --foundation-backbone vit_small_patch14_dinov2.lvd142m --no-pretrained --freeze-backbone --image-size 518 --aggregation source_attention --batch-size 1 --epochs 1 --learning-rate 1e-4 --device cpu --peft lora --lora-rank 4 --lora-alpha 8 --lora-dropout 0.0 --lora-target-modules qkv,proj,fc1,fc2
```

dummy_dinov2 + Adapter CPU smoke：

```powershell
python -m fwi_visionfm.run_foundation_experiment --data-dir D:\ryjin\fwi_visionfm\outputs\converted_tiny --output-dir D:\ryjin\fwi_visionfm\outputs\dummy_dinov2_adapter_smoke --depth 6 --width 7 --foundation-backbone dummy_dinov2 --no-pretrained --freeze-backbone --image-size 64 --aggregation source_attention --batch-size 2 --epochs 1 --learning-rate 1e-3 --device cpu --peft adapter --adapter-bottleneck-dim 8 --adapter-dropout 0.0
```

frozen vs LoRA 对比建议：

```powershell
python -m fwi_visionfm.run_foundation_experiment --data-dir D:\ryjin\fwi_visionfm\outputs\converted_tiny --output-dir D:\ryjin\fwi_visionfm\outputs\dummy_dinov2_frozen --depth 6 --width 7 --foundation-backbone dummy_dinov2 --no-pretrained --freeze-backbone --image-size 64 --aggregation source_attention --batch-size 2 --epochs 3 --learning-rate 1e-3 --device cpu

python -m fwi_visionfm.run_foundation_experiment --data-dir D:\ryjin\fwi_visionfm\outputs\converted_tiny --output-dir D:\ryjin\fwi_visionfm\outputs\dummy_dinov2_lora --depth 6 --width 7 --foundation-backbone dummy_dinov2 --no-pretrained --freeze-backbone --image-size 64 --aggregation source_attention --batch-size 2 --epochs 3 --learning-rate 1e-3 --device cpu --peft lora --lora-rank 4 --lora-alpha 8 --lora-dropout 0.0 --lora-target-modules qkv,proj,fc1,fc2

python -m fwi_visionfm.compare_experiments --experiment-dirs D:\ryjin\fwi_visionfm\outputs\dummy_dinov2_frozen D:\ryjin\fwi_visionfm\outputs\dummy_dinov2_lora --output D:\ryjin\fwi_visionfm\outputs\comparison_dummy_lora
```

## 实验对比与曲线绘制

PyTorch CNN baseline 10 epoch：

```powershell
python -m fwi_visionfm.run_torch_experiment --data-dir D:\ryjin\fwi_visionfm\outputs\converted_tiny --output-dir D:\ryjin\fwi_visionfm\outputs\torch_baseline_10ep --depth 6 --width 7 --channels raw,offset --aggregation source_attention --batch-size 2 --epochs 10 --learning-rate 1e-3 --device cpu
```

dummy_dinov2 frozen 5 epoch：

```powershell
python -m fwi_visionfm.run_foundation_experiment --data-dir D:\ryjin\fwi_visionfm\outputs\converted_tiny --output-dir D:\ryjin\fwi_visionfm\outputs\dummy_dinov2_5ep --depth 6 --width 7 --foundation-backbone dummy_dinov2 --no-pretrained --freeze-backbone --image-size 64 --aggregation source_attention --batch-size 2 --epochs 5 --learning-rate 1e-3 --device cpu
```

实验对比：

```powershell
python -m fwi_visionfm.compare_experiments --experiment-dirs D:\ryjin\fwi_visionfm\outputs\torch_baseline_10ep D:\ryjin\fwi_visionfm\outputs\dummy_dinov2_5ep --output D:\ryjin\fwi_visionfm\outputs\comparison
```

训练曲线绘制：

```powershell
python -m fwi_visionfm.plot_training_curves --history D:\ryjin\fwi_visionfm\outputs\torch_baseline_10ep\torch_training_history.csv --output D:\ryjin\fwi_visionfm\outputs\comparison\torch_baseline_10ep_loss.png
```

Windows / Anaconda 环境如果遇到：

```text
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
```

可以临时在 PowerShell 设置：

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:NUMEXPR_NUM_THREADS="1"
```

也可以直接在绘图命令后追加：

```powershell
python -m fwi_visionfm.plot_training_curves --history D:\ryjin\fwi_visionfm\outputs\torch_baseline_10ep\torch_training_history.csv --output D:\ryjin\fwi_visionfm\outputs\comparison\torch_baseline_10ep_loss.png --allow-duplicate-openmp
```

`KMP_DUPLICATE_LIB_OK` 只建议用于绘图等后处理，不建议长期用于正式训练。

## Legacy 标准实验矩阵

可以用标准矩阵一次性运行三组 tiny/smoke 对照实验：

```powershell
python -m fwi_visionfm.run_experiment_matrix --data-dir D:\ryjin\fwi_visionfm\outputs\converted_tiny --output-root D:\ryjin\fwi_visionfm\outputs\matrix_tiny --depth 6 --width 7 --device cpu
```

该矩阵默认包含：

1. `torch_cnn_baseline`
2. `dummy_dinov2_frozen`
3. `dummy_dinov2_lora`

注意：

- 该矩阵只用于 tiny/smoke 工程验证。
- 它不能作为最终科研结论。
- 后续真实 OpenFWI 实验会替换 `data-dir` 和 split 策略。

## Legacy OpenFWI tiny subset

OpenFWI tiny subset 只用于格式验证和 smoke，不下载数据，不直接作为科研结论。

dry-run：

```powershell
python -m fwi_visionfm.convert_openfwi --records D:\data\openfwi\data.npy --velocity D:\data\openfwi\model.npy --output-dir D:\ryjin\fwi_visionfm\data\openfwi_tiny --dataset-name flatvel_a_tiny --records-layout samples_shots_time_receivers --max-samples 16 --dry-run
```

转换：

```powershell
python -m fwi_visionfm.convert_openfwi --records D:\data\openfwi\data.npy --velocity D:\data\openfwi\model.npy --output-dir D:\ryjin\fwi_visionfm\data\openfwi_tiny --dataset-name flatvel_a_tiny --records-layout samples_shots_time_receivers --max-samples 16
```

验证：

```powershell
python -m fwi_visionfm.validate_npz_dataset --data-dir D:\ryjin\fwi_visionfm\data\openfwi_tiny --max-checks 16
```

跑 matrix：

```powershell
python -m fwi_visionfm.run_experiment_matrix --data-dir D:\ryjin\fwi_visionfm\data\openfwi_tiny --output-root D:\ryjin\fwi_visionfm\outputs\matrix_openfwi_tiny --auto-shape --device cpu
```

说明：

- OpenFWI tiny subset 只用于格式验证和 smoke。
- 真实科研结果至少需要更大 subset 和固定 train/val/test split。
- 后续 cross-family 需要 FlatVel / CurveVel / Fault 等不同 family 分开转换。

## Legacy OpenFWI in-domain / cross-family split

先在转换阶段写入 family / split / subset 元数据，例如：

```powershell
python -m fwi_visionfm.convert_openfwi --records D:\data\openfwi\data.npy --velocity D:\data\openfwi\model.npy --output-dir D:\ryjin\fwi_visionfm\data\flatvel_a_tiny --dataset-name flatvel_a_tiny --family flatvel_a --split-name tiny --subset-name flatvel_a_tiny16 --records-layout samples_shots_time_receivers --max-samples 16
```

FlatVel-A tiny in-domain split：

```powershell
python -m fwi_visionfm.make_split_manifest --data-dirs D:\ryjin\fwi_visionfm\data\flatvel_a_tiny --output D:\ryjin\fwi_visionfm\data\splits\flatvel_a_tiny_split.json --train-fraction 0.7 --val-fraction 0.15 --seed 2026
```

FlatVel + CurveVel -> Fault cross-family：

```powershell
python -m fwi_visionfm.make_cross_family_split --train-dirs D:\ryjin\fwi_visionfm\data\flatvel_a_tiny D:\ryjin\fwi_visionfm\data\curvevel_a_tiny --test-dirs D:\ryjin\fwi_visionfm\data\flatfault_a_tiny --output D:\ryjin\fwi_visionfm\data\splits\flat_curve_to_fault_tiny.json --train-fraction 0.8 --seed 2026
```

验证 split manifest：

```powershell
python -m fwi_visionfm.validate_split_manifest --split-manifest D:\ryjin\fwi_visionfm\data\splits\flatvel_a_tiny_split.json
```

使用 split manifest 跑 matrix：

```powershell
python -m fwi_visionfm.run_experiment_matrix --split-manifest D:\ryjin\fwi_visionfm\data\splits\flatvel_a_tiny_split.json --output-root D:\ryjin\fwi_visionfm\outputs\matrix_flatvel_tiny_split --auto-shape --device cpu
```

说明：

- 这些 split 主要用于 tiny/smoke 与格式验证，不直接作为科研结论。
- 真实科研实验应固定 train/val/test split，并扩大 subset 规模。
- 后续可分别准备 `flatvel_a`、`curvevel_a`、`flatfault_a`，再进入 in-domain 与 cross-family 对照。

## Legacy Shape Guard / Auto-Shape

当前仓库里同时存在两类数据：

- smoke 数据可能是 `6x7` 或 `5x7` 这类小尺寸 velocity。
- 真实 OpenFWI 数据通常是 `70x70` velocity。

为避免把 smoke 结果误当成 OpenFWI 结果，`run_experiment_matrix` 现在支持 shape guard：

- 可用 `--auto-shape` 从 `data-dir` 或 `split-manifest` 自动推断 `depth/width`。
- 如果显式传入 `--depth --width`，程序仍会检查与数据真实 shape 是否一致。
- 如果不一致，会直接报错，不继续训练。

示例：

```powershell
python -m fwi_visionfm.run_experiment_matrix --data-dir D:\ryjin\fwi_visionfm\data\openfwi_tiny --output-root D:\ryjin\fwi_visionfm\outputs\matrix_openfwi_tiny --auto-shape --device cpu
```

```powershell
python -m fwi_visionfm.run_experiment_matrix --split-manifest D:\ryjin\fwi_visionfm\data\splits\flatvel_a_tiny_split.json --output-root D:\ryjin\fwi_visionfm\outputs\matrix_flatvel_tiny_split --auto-shape --device cpu
```

如果你明确传入：

```powershell
python -m fwi_visionfm.run_experiment_matrix --data-dir D:\ryjin\fwi_visionfm\outputs\converted_tiny --output-root D:\ryjin\fwi_visionfm\outputs\matrix_bad_shape --depth 70 --width 70 --device cpu
```

而数据实际是 `6x7`，程序会直接报错，避免误报结果。

## 下载真实 OpenFWI FlatVel-A

真实数据不包含在 OpenFWI GitHub benchmark 代码仓库中。`GitHub` 主要提供基准代码、说明文档和数据入口；真正的 `FlatVel-A` 数据需要从 OpenFWI 数据页或其 Google Drive 文件夹下载。

参考链接：

- [OpenFWI GitHub](https://github.com/lanl/OpenFWI/)
- [OpenFWI 数据页](https://openfwi-lanl.github.io/docs/data.html)
- [FlatVel-A Google Drive folder](https://drive.google.com/drive/folders/1NIdjiYhjWSV9NHn7ZEFYTpJxzvzxqYRb?usp=sharing)

下载前建议先确认磁盘空间。`FlatVel-A` 大约 `43GB`，建议 `D` 盘至少保留 `60GB` 可用空间，再开始下载和解压。

可先检查 `D` 盘剩余空间：

```powershell
Get-PSDrive D
```

推荐优先使用 `gdown` 直接下载整个 Google Drive 文件夹：

```powershell
cd D:\ryjin

python -m pip install --upgrade gdown

New-Item -ItemType Directory -Force D:\data\openfwi\FlatVel_A

gdown --folder "https://drive.google.com/drive/folders/1NIdjiYhjWSV9NHn7ZEFYTpJxzvzxqYRb?usp=sharing" -O D:\data\openfwi\FlatVel_A
```

可选说明：

- 如果 `gdown --folder` 失败，通常是 Google Drive 访问限制、网络代理或 `gdown` 版本问题。
- 这种情况下，直接在浏览器打开上面的 Google Drive 文件夹手动下载，再解压到 `D:\data\openfwi\FlatVel_A`。
- 后续 README 中所有 `<真实路径>\data1.npy`、`<真实路径>\model1.npy` 都应替换成你本机下载后的实际路径。

## Dataset Download Plan

当前项目需要严格区分两类数据来源：

1. `OpenFWI FlatVel-A / CurveVel-A / FlatFault-A` 是 FWI 主任务数据。
2. `Figshare 30702569` 先作为待检查的 cross-domain seismic interpretation 数据。
3. 如果 Figshare 不包含 `shot records + velocity maps`，则不能作为 FWI 主训练集。
4. Figshare 可用于 `seismic-to-vision bridge`、`auxiliary pretraining`、`seismic-domain adaptation`。
5. OpenFWI `tiny16` 只用于接线验证，不作为科研结论。
6. `subset500 / subset2k` 仍是阶段性实验，最终需要更大固定 split。

推荐执行顺序：

- 先检查 `D:\data\openfwi` 下是否已有 `data1.npy / model1.npy`，避免重复下载。
- 对 `FlatVel-A / CurveVel-A / FlatFault-A` 分别做 shape 验证，确认是否为 `(500, 5, 1000, 70)` 和 `(500, 1, 70, 70)`。
- 先拉取 Figshare article metadata，再根据文件结构判断它属于解释/属性回归还是 FWI 主任务。
- 只有在 OpenFWI family 真正落盘后，才继续做 `tiny16 / subset500 / split manifest / matrix`。

## OpenFWI First-File Subset500 Experiments

每个 OpenFWI family 的第一个 records/model 文件通常包含 `500` 个样本。本阶段只使用 first-file `subset500`，用于快速验证 in-domain 与 cross-family 泛化流程，不直接启动全量训练。

当前 first-file 检查命令：

```powershell
python tools\inspect_openfwi_first_files.py --root D:\data\openfwi --output-md D:\ryjin\fwi_visionfm\outputs\openfwi_first_files_summary.md --output-json D:\ryjin\fwi_visionfm\outputs\openfwi_first_files_summary.json
```

当前 CPU 阶段使用：

- `epochs=3`
- `batch_size=2`
- `device=cpu`
- `--auto-shape`

subset500 结果仍然只是阶段性工程验证，不是最终论文结论。`dummy_dinov2 / dummy_lora` 只代表 foundation-style 接口与 LoRA 工程闭环，不代表真实 DINOv2 预训练迁移效果。后续只有在 cross-family 中出现稳定、有意义的差异后，才应扩展到 `subset2k / subset5k` 或迁移到 GPU。

批量绘制 subset500 matrix 曲线：

```powershell
python tools\plot_all_matrix_curves.py --root D:\ryjin\fwi_visionfm\outputs --allow-duplicate-openmp
```

生成 cross-family subset500 总汇报：

```powershell
python -m fwi_visionfm.summarize_cross_family --matrix-dirs D:\ryjin\fwi_visionfm\outputs\matrix_flatvel_a_subset500 D:\ryjin\fwi_visionfm\outputs\matrix_curvevel_a_subset500 D:\ryjin\fwi_visionfm\outputs\matrix_flatfault_a_subset500 D:\ryjin\fwi_visionfm\outputs\matrix_flatvel_to_curvevel_subset500 D:\ryjin\fwi_visionfm\outputs\matrix_curvevel_to_flatvel_subset500 D:\ryjin\fwi_visionfm\outputs\matrix_flat_curve_to_flatfault_subset500 --output D:\ryjin\fwi_visionfm\outputs\cross_family_subset500_summary.md
```

## 真实 OpenFWI FlatVel-A tiny 接入流程

根据 OpenFWI 官方数据页与官方代码仓库，`FlatVel-A` 单个 family 数据量约 `43GB`，输入 shape 为 `(5,1000,70)`，输出 shape 为 `(1,70,70)`。[OpenFWI 数据页](https://openfwi-lanl.github.io/docs/data.html) [OpenFWI GitHub](https://github.com/lanl/OpenFWI/)

注意：

- 你必须先从 OpenFWI 官网下载并解压 `FlatVel-A`。
- README 中的 `D:\data\openfwi\...` 只是示例路径，必须替换为本机真实路径。
- 推荐先只用 `data1.npy / model1.npy` 的前 `16` 个样本做 `tiny` 格式验证。
- `tiny16` 只用于接线和 smoke，不能作为最终科研结论。

A. 查找本机 OpenFWI 文件：

```powershell
Get-ChildItem D:\ -Recurse -Filter data1.npy -ErrorAction SilentlyContinue | Select-Object FullName
Get-ChildItem D:\ -Recurse -Filter model1.npy -ErrorAction SilentlyContinue | Select-Object FullName
```

B. dry-run，先替换为真实路径：

```powershell
python -m fwi_visionfm.convert_openfwi --records <真实路径>\data1.npy --velocity <真实路径>\model1.npy --output-dir D:\ryjin\fwi_visionfm\data\flatvel_a_tiny --dataset-name flatvel_a_tiny --family flatvel_a --split-name tiny --subset-name flatvel_a_tiny16 --records-layout samples_shots_time_receivers --max-samples 16 --dry-run
```

C. 正式转换：

```powershell
python -m fwi_visionfm.convert_openfwi --records <真实路径>\data1.npy --velocity <真实路径>\model1.npy --output-dir D:\ryjin\fwi_visionfm\data\flatvel_a_tiny --dataset-name flatvel_a_tiny --family flatvel_a --split-name tiny --subset-name flatvel_a_tiny16 --records-layout samples_shots_time_receivers --max-samples 16
```

D. validate：

```powershell
python -m fwi_visionfm.validate_npz_dataset --data-dir D:\ryjin\fwi_visionfm\data\flatvel_a_tiny --max-checks 16
```

真实 FlatVel-A tiny 的 validate 结果必须确认：

- `inferred_depth = 70`
- `inferred_width = 70`
- `velocity_shape_set = [(70,70)]`
- `records_shape_set = [(5,70,1000)]`

如果这里不是 `70x70`，不要继续跑 matrix。

E. 生成 split：

```powershell
python -m fwi_visionfm.make_split_manifest --data-dirs D:\ryjin\fwi_visionfm\data\flatvel_a_tiny --output D:\ryjin\fwi_visionfm\data\splits\flatvel_a_tiny_split.json --train-fraction 0.7 --val-fraction 0.15 --seed 2026
```

F. validate split：

```powershell
python -m fwi_visionfm.validate_split_manifest --split-manifest D:\ryjin\fwi_visionfm\data\splits\flatvel_a_tiny_split.json
```

G. 使用 `--auto-shape` 跑 matrix：

```powershell
python -m fwi_visionfm.run_experiment_matrix --split-manifest D:\ryjin\fwi_visionfm\data\splits\flatvel_a_tiny_split.json --output-root D:\ryjin\fwi_visionfm\outputs\matrix_flatvel_a_tiny_split --auto-shape --device cpu
```

H. 可选显式 `70x70` shape guard：

```powershell
python -m fwi_visionfm.run_experiment_matrix --split-manifest D:\ryjin\fwi_visionfm\data\splits\flatvel_a_tiny_split.json --output-root D:\ryjin\fwi_visionfm\outputs\matrix_flatvel_a_tiny_split_70x70 --depth 70 --width 70 --device cpu
```

如果后续增加 `generate_openfwi_tiny_report.py`，可再补：

```powershell
python -m fwi_visionfm.generate_openfwi_tiny_report --data-dir D:\ryjin\fwi_visionfm\data\flatvel_a_tiny --split-manifest D:\ryjin\fwi_visionfm\data\splits\flatvel_a_tiny_split.json --matrix-dir D:\ryjin\fwi_visionfm\outputs\matrix_flatvel_a_tiny_split --output D:\ryjin\fwi_visionfm\outputs\matrix_flatvel_a_tiny_split\openfwi_tiny_report.md
```

## FlatVel-A Tiny Report

真实 `FlatVel-A tiny16` 接线完成后，可以把数据验证、split 验证和 matrix 对比汇总成一个 Markdown 报告：

```powershell
python -m fwi_visionfm.generate_openfwi_tiny_report --data-dir D:\ryjin\fwi_visionfm\data\flatvel_a_tiny --split-manifest D:\ryjin\fwi_visionfm\data\splits\flatvel_a_tiny_split.json --matrix-dir D:\ryjin\fwi_visionfm\outputs\matrix_flatvel_a_tiny_split --output D:\ryjin\fwi_visionfm\outputs\matrix_flatvel_a_tiny_split\openfwi_tiny_report.md
```

报告会汇总：

- 数据集元数据：`dataset_name`、`family`、`split_name`、`subset_name`
- `validation_summary.json` 中的 shape 与统计
- `split_validation_summary.json` 中的 train/val/test 统计与 family 分布
- `comparison_summary.json/csv` 中的三组实验指标表

如果以下文件缺失，报告生成器会直接报错：

- `D:\ryjin\fwi_visionfm\data\flatvel_a_tiny\validation_summary.json`
- `D:\ryjin\fwi_visionfm\data\splits\split_validation_summary.json`
- `D:\ryjin\fwi_visionfm\outputs\matrix_flatvel_a_tiny_split\comparison\comparison_summary.json`
- 或 `D:\ryjin\fwi_visionfm\outputs\matrix_flatvel_a_tiny_split\comparison\comparison_summary.csv`

## FlatVel-A subset500

`subset500` 直接复用你已经下载好的 `data1.npy / model1.npy`：

- `D:\data\openfwi\FlatVel_A\data\data1.npy`
- `D:\data\openfwi\FlatVel_A\model\model1.npy`

这一步不需要继续下载 `data2.npy / model2.npy`。

建议输出目录与命名：

- 数据目录：`D:\ryjin\fwi_visionfm\data\flatvel_a_subset500`
- split：`D:\ryjin\fwi_visionfm\data\splits\flatvel_a_subset500_split.json`
- matrix：`D:\ryjin\fwi_visionfm\outputs\matrix_flatvel_a_subset500`
- report：`D:\ryjin\fwi_visionfm\outputs\matrix_flatvel_a_subset500\openfwi_subset500_report.md`

1. 转换前 500 个样本：

```powershell
python -m fwi_visionfm.convert_openfwi --records D:\data\openfwi\FlatVel_A\data\data1.npy --velocity D:\data\openfwi\FlatVel_A\model\model1.npy --output-dir D:\ryjin\fwi_visionfm\data\flatvel_a_subset500 --dataset-name flatvel_a_data1_500 --family flatvel_a --split-name subset500 --subset-name flatvel_a_data1_500 --records-layout samples_shots_time_receivers --max-samples 500
```

2. 验证数据：

```powershell
python -m fwi_visionfm.validate_npz_dataset --data-dir D:\ryjin\fwi_visionfm\data\flatvel_a_subset500 --max-checks 500
```

3. 生成 split：

```powershell
python -m fwi_visionfm.make_split_manifest --data-dirs D:\ryjin\fwi_visionfm\data\flatvel_a_subset500 --output D:\ryjin\fwi_visionfm\data\splits\flatvel_a_subset500_split.json --train-fraction 0.7 --val-fraction 0.15 --seed 2026
```

4. 验证 split：

```powershell
python -m fwi_visionfm.validate_split_manifest --split-manifest D:\ryjin\fwi_visionfm\data\splits\flatvel_a_subset500_split.json
```

5. 跑 matrix：

```powershell
python -m fwi_visionfm.run_experiment_matrix --split-manifest D:\ryjin\fwi_visionfm\data\splits\flatvel_a_subset500_split.json --output-root D:\ryjin\fwi_visionfm\outputs\matrix_flatvel_a_subset500 --auto-shape --device cpu
```

6. 生成 subset500 报告：

```powershell
python -m fwi_visionfm.generate_openfwi_tiny_report --data-dir D:\ryjin\fwi_visionfm\data\flatvel_a_subset500 --split-manifest D:\ryjin\fwi_visionfm\data\splits\flatvel_a_subset500_split.json --matrix-dir D:\ryjin\fwi_visionfm\outputs\matrix_flatvel_a_subset500 --output D:\ryjin\fwi_visionfm\outputs\matrix_flatvel_a_subset500\openfwi_subset500_report.md
```

当前报告标题会根据 `manifest.json` 中的 `subset_name` 自动生成，例如：

- `OpenFWI FlatVel-A flatvel_a_tiny16 Report`
- `OpenFWI FlatVel-A flatvel_a_data1_500 Report`

## FlatVel-A subset2k 计划

`subset2k` 继续使用同一批已经下载好的 `FlatVel_A` 数据，但扩展到：

- `D:\data\openfwi\FlatVel_A\data\data1.npy`
- `D:\data\openfwi\FlatVel_A\data\data2.npy`
- `D:\data\openfwi\FlatVel_A\data\data3.npy`
- `D:\data\openfwi\FlatVel_A\data\data4.npy`

以及对应的：

- `D:\data\openfwi\FlatVel_A\model\model1.npy`
- `D:\data\openfwi\FlatVel_A\model\model2.npy`
- `D:\data\openfwi\FlatVel_A\model\model3.npy`
- `D:\data\openfwi\FlatVel_A\model\model4.npy`

总样本数为 `2000`。这仍然属于小规模 in-domain baseline，不是最终科研结论。

推荐使用多文件转换入口：

dry-run：

```powershell
python -m fwi_visionfm.convert_openfwi_multi --records D:\data\openfwi\FlatVel_A\data\data1.npy D:\data\openfwi\FlatVel_A\data\data2.npy D:\data\openfwi\FlatVel_A\data\data3.npy D:\data\openfwi\FlatVel_A\data\data4.npy --velocity D:\data\openfwi\FlatVel_A\model\model1.npy D:\data\openfwi\FlatVel_A\model\model2.npy D:\data\openfwi\FlatVel_A\model\model3.npy D:\data\openfwi\FlatVel_A\model\model4.npy --output-dir D:\ryjin\fwi_visionfm\data\flatvel_a_subset2k --dataset-name flatvel_a_subset2k --family flatvel_a --split-name subset2k --subset-name flatvel_a_data1_4_2000 --records-layout samples_shots_time_receivers --dry-run
```

正式转换：

```powershell
python -m fwi_visionfm.convert_openfwi_multi --records D:\data\openfwi\FlatVel_A\data\data1.npy D:\data\openfwi\FlatVel_A\data\data2.npy D:\data\openfwi\FlatVel_A\data\data3.npy D:\data\openfwi\FlatVel_A\data\data4.npy --velocity D:\data\openfwi\FlatVel_A\model\model1.npy D:\data\openfwi\FlatVel_A\model\model2.npy D:\data\openfwi\FlatVel_A\model\model3.npy D:\data\openfwi\FlatVel_A\model\model4.npy --output-dir D:\ryjin\fwi_visionfm\data\flatvel_a_subset2k --dataset-name flatvel_a_subset2k --family flatvel_a --split-name subset2k --subset-name flatvel_a_data1_4_2000 --records-layout samples_shots_time_receivers
```

后续流程与 subset500 一致：

1. `validate_npz_dataset --max-checks 2000`
2. `make_split_manifest --train-fraction 0.7 --val-fraction 0.15 --seed 2026`
3. `validate_split_manifest`
4. `run_experiment_matrix --auto-shape`
5. `generate_openfwi_tiny_report`

subset2k 报告命令示例：

```powershell
python -m fwi_visionfm.generate_openfwi_tiny_report --data-dir D:\ryjin\fwi_visionfm\data\flatvel_a_subset2k --split-manifest D:\ryjin\fwi_visionfm\data\splits\flatvel_a_subset2k_split.json --matrix-dir D:\ryjin\fwi_visionfm\outputs\matrix_flatvel_a_subset2k --output D:\ryjin\fwi_visionfm\outputs\matrix_flatvel_a_subset2k\openfwi_subset2k_report.md
```

## FlatVel-A subset2k 10ep 稳定性复验

3 epoch 只用于快速验证训练闭环。要观察三组模型在 `subset2k` 上的曲线稳定性，可再跑一轮 10 epoch：

```powershell
python -m fwi_visionfm.run_experiment_matrix --split-manifest D:\ryjin\fwi_visionfm\data\splits\flatvel_a_subset2k_split.json --output-root D:\ryjin\fwi_visionfm\outputs\matrix_flatvel_a_subset2k_10ep --auto-shape --device cpu --epochs 10 --learning-rate 1e-3 --batch-size 2
```

10 epoch 报告：

```powershell
python -m fwi_visionfm.generate_openfwi_tiny_report --data-dir D:\ryjin\fwi_visionfm\data\flatvel_a_subset2k --split-manifest D:\ryjin\fwi_visionfm\data\splits\flatvel_a_subset2k_split.json --matrix-dir D:\ryjin\fwi_visionfm\outputs\matrix_flatvel_a_subset2k_10ep --output D:\ryjin\fwi_visionfm\outputs\matrix_flatvel_a_subset2k_10ep\openfwi_subset2k_10ep_report.md
```

三组训练曲线：

```powershell
python -m fwi_visionfm.plot_training_curves --history D:\ryjin\fwi_visionfm\outputs\matrix_flatvel_a_subset2k_10ep\torch_cnn_baseline\torch_training_history.csv --output D:\ryjin\fwi_visionfm\outputs\matrix_flatvel_a_subset2k_10ep\torch_cnn_baseline_loss.png --allow-duplicate-openmp

python -m fwi_visionfm.plot_training_curves --history D:\ryjin\fwi_visionfm\outputs\matrix_flatvel_a_subset2k_10ep\dummy_dinov2_frozen\foundation_training_history.csv --output D:\ryjin\fwi_visionfm\outputs\matrix_flatvel_a_subset2k_10ep\dummy_dinov2_frozen_loss.png --allow-duplicate-openmp

python -m fwi_visionfm.plot_training_curves --history D:\ryjin\fwi_visionfm\outputs\matrix_flatvel_a_subset2k_10ep\dummy_dinov2_lora\foundation_training_history.csv --output D:\ryjin\fwi_visionfm\outputs\matrix_flatvel_a_subset2k_10ep\dummy_dinov2_lora_loss.png --allow-duplicate-openmp
```

scale summary：

```powershell
python -m fwi_visionfm.compare_scale_reports --reports D:\ryjin\fwi_visionfm\outputs\matrix_flatvel_a_tiny_split\openfwi_tiny_report.md D:\ryjin\fwi_visionfm\outputs\matrix_flatvel_a_subset500\openfwi_subset500_report.md D:\ryjin\fwi_visionfm\outputs\matrix_flatvel_a_subset2k\openfwi_subset2k_report.md D:\ryjin\fwi_visionfm\outputs\matrix_flatvel_a_subset2k_10ep\openfwi_subset2k_10ep_report.md --output D:\ryjin\fwi_visionfm\outputs\flatvel_a_scale_summary.md
```

说明：

- 3 epoch 只用于快速验证。
- 10 epoch 用于观察曲线稳定性。
- 如果三组仍然接近，下一阶段应进入 `CurveVel-A` cross-family，而不是继续无限调 `FlatVel-A`。

## 当前 PyTorch backend 边界

当前 `fwi_visionfm/torch_backend/` 只实现最小 baseline：

- `data.py`：`.npz` 样本读取与 `DataLoader`。
- `model.py`：`SeismicToVisionTorchBridge`、`PseudoVisionImageBridge`、轻量 CNN backbone、统一 vision backbone 接口、跨炮聚合、bounded velocity decoder。
- `train.py`：CPU 训练 loop、checkpoint、history、prediction dump、torch smoke。

`models/vision_backbones.py` 与 `models/seismic_bridge.py` 则补上了真实视觉基础模型迁移的统一 backbone/bridge 抽象；`foundation_train.py` 和 `run_foundation_experiment.py` 提供了 frozen / Adapter / LoRA 的数据集级训练入口。

## CurveVel-A and Cross-Family Experiments

当前推进重点已经从 `FlatVel-A` 单 family in-domain baseline，转向 `CurveVel-A / Fault` 的 in-domain 与 cross-family 泛化验证。

约束：

1. `FlatVel-A subset2k` 已作为 in-domain baseline，不继续反复微调 tiny/subset 超参。
2. `CurveVel-A tiny16` 只用于格式验证。
3. `CurveVel-A subset500` 用于第二个 family 的 in-domain baseline。
4. `FlatVel -> CurveVel` 用于 cross-family 泛化测试。
5. `FlatVel + CurveVel -> Fault` 用于更强的 cross-family 测试。
6. 当前 `dummy_dinov2 / dummy_lora` 仍然是工程验证，不是真实 DINOv2 预训练结论。
7. 真实科研结论需要真实 DINOv2 / MAE / SAM backbone 与更大固定 split。

先检查本机是否已有 `CurveVel-A / Fault` 数据，不要假设路径存在：

```powershell
Get-ChildItem D:\data\openfwi -Directory -ErrorAction SilentlyContinue
Get-ChildItem D:\data\openfwi -Recurse -Filter data1.npy -ErrorAction SilentlyContinue | Select-Object FullName
Get-ChildItem D:\data\openfwi -Recurse -Filter model1.npy -ErrorAction SilentlyContinue | Select-Object FullName
```

如果未找到 `CurveVel-A / Fault`，当前机器上只能先停在检测与脚本准备阶段；不要伪造 cross-family 结果。

### CurveVel-A tiny16

```powershell
python -m fwi_visionfm.convert_openfwi --records <CurveVel_A真实路径>\data\data1.npy --velocity <CurveVel_A真实路径>\model\model1.npy --output-dir D:\ryjin\fwi_visionfm\data\curvevel_a_tiny --dataset-name curvevel_a_tiny --family curvevel_a --split-name tiny --subset-name curvevel_a_tiny16 --records-layout samples_shots_time_receivers --max-samples 16 --dry-run

python -m fwi_visionfm.convert_openfwi --records <CurveVel_A真实路径>\data\data1.npy --velocity <CurveVel_A真实路径>\model\model1.npy --output-dir D:\ryjin\fwi_visionfm\data\curvevel_a_tiny --dataset-name curvevel_a_tiny --family curvevel_a --split-name tiny --subset-name curvevel_a_tiny16 --records-layout samples_shots_time_receivers --max-samples 16

python -m fwi_visionfm.validate_npz_dataset --data-dir D:\ryjin\fwi_visionfm\data\curvevel_a_tiny --max-checks 16
python -m fwi_visionfm.make_split_manifest --data-dirs D:\ryjin\fwi_visionfm\data\curvevel_a_tiny --output D:\ryjin\fwi_visionfm\data\splits\curvevel_a_tiny_split.json --train-fraction 0.7 --val-fraction 0.15 --seed 2026
python -m fwi_visionfm.validate_split_manifest --split-manifest D:\ryjin\fwi_visionfm\data\splits\curvevel_a_tiny_split.json
python -m fwi_visionfm.run_experiment_matrix --split-manifest D:\ryjin\fwi_visionfm\data\splits\curvevel_a_tiny_split.json --output-root D:\ryjin\fwi_visionfm\outputs\matrix_curvevel_a_tiny --auto-shape --device cpu --epochs 3 --learning-rate 1e-3 --batch-size 2
python -m fwi_visionfm.generate_openfwi_tiny_report --data-dir D:\ryjin\fwi_visionfm\data\curvevel_a_tiny --split-manifest D:\ryjin\fwi_visionfm\data\splits\curvevel_a_tiny_split.json --matrix-dir D:\ryjin\fwi_visionfm\outputs\matrix_curvevel_a_tiny --output D:\ryjin\fwi_visionfm\outputs\matrix_curvevel_a_tiny\openfwi_curvevel_a_tiny_report.md
```

### CurveVel-A subset500

```powershell
python -m fwi_visionfm.convert_openfwi --records <CurveVel_A真实路径>\data\data1.npy --velocity <CurveVel_A真实路径>\model\model1.npy --output-dir D:\ryjin\fwi_visionfm\data\curvevel_a_subset500 --dataset-name curvevel_a_subset500 --family curvevel_a --split-name subset500 --subset-name curvevel_a_data1_500 --records-layout samples_shots_time_receivers --max-samples 500
python -m fwi_visionfm.validate_npz_dataset --data-dir D:\ryjin\fwi_visionfm\data\curvevel_a_subset500 --max-checks 500
python -m fwi_visionfm.make_split_manifest --data-dirs D:\ryjin\fwi_visionfm\data\curvevel_a_subset500 --output D:\ryjin\fwi_visionfm\data\splits\curvevel_a_subset500_split.json --train-fraction 0.7 --val-fraction 0.15 --seed 2026
python -m fwi_visionfm.validate_split_manifest --split-manifest D:\ryjin\fwi_visionfm\data\splits\curvevel_a_subset500_split.json
python -m fwi_visionfm.run_experiment_matrix --split-manifest D:\ryjin\fwi_visionfm\data\splits\curvevel_a_subset500_split.json --output-root D:\ryjin\fwi_visionfm\outputs\matrix_curvevel_a_subset500 --auto-shape --device cpu --epochs 5 --learning-rate 1e-3 --batch-size 2
python -m fwi_visionfm.generate_openfwi_tiny_report --data-dir D:\ryjin\fwi_visionfm\data\curvevel_a_subset500 --split-manifest D:\ryjin\fwi_visionfm\data\splits\curvevel_a_subset500_split.json --matrix-dir D:\ryjin\fwi_visionfm\outputs\matrix_curvevel_a_subset500 --output D:\ryjin\fwi_visionfm\outputs\matrix_curvevel_a_subset500\openfwi_curvevel_a_subset500_report.md
```

### FlatVel -> CurveVel

```powershell
python -m fwi_visionfm.make_cross_family_split --train-dirs D:\ryjin\fwi_visionfm\data\flatvel_a_subset2k --test-dirs D:\ryjin\fwi_visionfm\data\curvevel_a_subset500 --output D:\ryjin\fwi_visionfm\data\splits\flatvel_to_curvevel_subset.json --train-fraction 0.85 --seed 2026
python -m fwi_visionfm.validate_split_manifest --split-manifest D:\ryjin\fwi_visionfm\data\splits\flatvel_to_curvevel_subset.json
python -m fwi_visionfm.run_experiment_matrix --split-manifest D:\ryjin\fwi_visionfm\data\splits\flatvel_to_curvevel_subset.json --output-root D:\ryjin\fwi_visionfm\outputs\matrix_flatvel_to_curvevel --auto-shape --device cpu --epochs 5 --learning-rate 1e-3 --batch-size 2
python -m fwi_visionfm.generate_openfwi_tiny_report --data-dir D:\ryjin\fwi_visionfm\data\curvevel_a_subset500 --split-manifest D:\ryjin\fwi_visionfm\data\splits\flatvel_to_curvevel_subset.json --matrix-dir D:\ryjin\fwi_visionfm\outputs\matrix_flatvel_to_curvevel --output D:\ryjin\fwi_visionfm\outputs\matrix_flatvel_to_curvevel\cross_family_flatvel_to_curvevel_report.md
```

### FlatVel + CurveVel -> Fault

```powershell
python -m fwi_visionfm.convert_openfwi --records <FlatFault_A真实路径>\data\data1.npy --velocity <FlatFault_A真实路径>\model\model1.npy --output-dir D:\ryjin\fwi_visionfm\data\flatfault_a_subset500 --dataset-name flatfault_a_subset500 --family flatfault_a --split-name subset500 --subset-name flatfault_a_data1_500 --records-layout samples_shots_time_receivers --max-samples 500
python -m fwi_visionfm.validate_npz_dataset --data-dir D:\ryjin\fwi_visionfm\data\flatfault_a_subset500 --max-checks 500
python -m fwi_visionfm.make_cross_family_split --train-dirs D:\ryjin\fwi_visionfm\data\flatvel_a_subset2k D:\ryjin\fwi_visionfm\data\curvevel_a_subset500 --test-dirs D:\ryjin\fwi_visionfm\data\flatfault_a_subset500 --output D:\ryjin\fwi_visionfm\data\splits\flat_curve_to_fault_subset.json --train-fraction 0.85 --seed 2026
python -m fwi_visionfm.validate_split_manifest --split-manifest D:\ryjin\fwi_visionfm\data\splits\flat_curve_to_fault_subset.json
python -m fwi_visionfm.run_experiment_matrix --split-manifest D:\ryjin\fwi_visionfm\data\splits\flat_curve_to_fault_subset.json --output-root D:\ryjin\fwi_visionfm\outputs\matrix_flat_curve_to_fault --auto-shape --device cpu --epochs 5 --learning-rate 1e-3 --batch-size 2
python -m fwi_visionfm.generate_openfwi_tiny_report --data-dir D:\ryjin\fwi_visionfm\data\flatfault_a_subset500 --split-manifest D:\ryjin\fwi_visionfm\data\splits\flat_curve_to_fault_subset.json --matrix-dir D:\ryjin\fwi_visionfm\outputs\matrix_flat_curve_to_fault --output D:\ryjin\fwi_visionfm\outputs\matrix_flat_curve_to_fault\cross_family_flat_curve_to_fault_report.md
```

### Cross-family Summary

```powershell
python -m fwi_visionfm.summarize_cross_family --reports D:\ryjin\fwi_visionfm\outputs\matrix_flatvel_a_subset2k\openfwi_subset2k_report.md D:\ryjin\fwi_visionfm\outputs\matrix_curvevel_a_subset500\openfwi_curvevel_a_subset500_report.md D:\ryjin\fwi_visionfm\outputs\matrix_flatvel_to_curvevel\cross_family_flatvel_to_curvevel_report.md --output D:\ryjin\fwi_visionfm\outputs\cross_family_summary.md
```

如果 Fault 实验也完成，再追加：

```powershell
D:\ryjin\fwi_visionfm\outputs\matrix_flat_curve_to_fault\cross_family_flat_curve_to_fault_report.md
```

## Protocol v1: Matched Target-Test Cross-Family Evaluation

旧版 `cross_family_subset500_summary.md` 汇总的是 `final_val_mae` / `final_val_rmse`，不等价于严格 target test 指标。Protocol v1 固定每个目标 family 的 test set，并保证 in-domain 与 cross-family 的训练样本数一致，用于避免把 smoke 结果误读为公平泛化结论。

这里需要和当前主线明确区分：

- 当前 raw OpenFWI small transfer 是 foundation transfer 方向的新入口。
- Protocol v1 是此前 matrix-based cross-family 评估协议。
- 两条线后续应通过统一的 `evaluate / summary / report` 脚本收敛到同一结果表。

先审计旧结果：

```powershell
python -m fwi_visionfm.audit_cross_family_results --experiment-dirs D:\ryjin\fwi_visionfm\outputs\matrix_flatvel_a_subset500 D:\ryjin\fwi_visionfm\outputs\matrix_curvevel_a_subset500 D:\ryjin\fwi_visionfm\outputs\matrix_flatfault_a_subset500 D:\ryjin\fwi_visionfm\outputs\matrix_flatvel_to_curvevel_subset500 D:\ryjin\fwi_visionfm\outputs\matrix_curvevel_to_flatvel_subset500 D:\ryjin\fwi_visionfm\outputs\matrix_flat_curve_to_flatfault_subset500 --split-manifests D:\ryjin\fwi_visionfm\data\splits\flatvel_a_subset500_split.json D:\ryjin\fwi_visionfm\data\splits\curvevel_a_subset500_split.json D:\ryjin\fwi_visionfm\data\splits\flatfault_a_subset500_split.json D:\ryjin\fwi_visionfm\data\splits\flatvel_to_curvevel_subset500.json D:\ryjin\fwi_visionfm\data\splits\curvevel_to_flatvel_subset500.json D:\ryjin\fwi_visionfm\data\splits\flat_curve_to_flatfault_subset500.json --output D:\ryjin\fwi_visionfm\outputs\cross_family_audit.md
```

生成 Protocol v1 split：

```powershell
python -m fwi_visionfm.make_target_test_protocol --flatvel-dir D:\ryjin\fwi_visionfm\data\flatvel_a_subset500 --curvevel-dir D:\ryjin\fwi_visionfm\data\curvevel_a_subset500 --flatfault-dir D:\ryjin\fwi_visionfm\data\flatfault_a_subset500 --output-dir D:\ryjin\fwi_visionfm\data\splits\protocol_v1 --train-count 350 --val-count 50 --test-count 100 --seed 2026
```

逐个验证 split：

```powershell
python -m fwi_visionfm.validate_split_manifest --split-manifest D:\ryjin\fwi_visionfm\data\splits\protocol_v1\protocol_v1_curvevel_indomain.json
python -m fwi_visionfm.validate_split_manifest --split-manifest D:\ryjin\fwi_visionfm\data\splits\protocol_v1\protocol_v1_flatvel_to_curvevel.json
python -m fwi_visionfm.validate_split_manifest --split-manifest D:\ryjin\fwi_visionfm\data\splits\protocol_v1\protocol_v1_flatfault_indomain.json
python -m fwi_visionfm.validate_split_manifest --split-manifest D:\ryjin\fwi_visionfm\data\splits\protocol_v1\protocol_v1_flat_curve_to_flatfault.json
```

运行必需的四组 3 epoch CPU matrix：

```powershell
python -m fwi_visionfm.run_experiment_matrix --split-manifest D:\ryjin\fwi_visionfm\data\splits\protocol_v1\protocol_v1_curvevel_indomain.json --output-root D:\ryjin\fwi_visionfm\outputs\protocol_v1_curvevel_indomain --auto-shape --device cpu --epochs 3 --learning-rate 1e-3 --batch-size 2
python -m fwi_visionfm.run_experiment_matrix --split-manifest D:\ryjin\fwi_visionfm\data\splits\protocol_v1\protocol_v1_flatvel_to_curvevel.json --output-root D:\ryjin\fwi_visionfm\outputs\protocol_v1_flatvel_to_curvevel --auto-shape --device cpu --epochs 3 --learning-rate 1e-3 --batch-size 2
python -m fwi_visionfm.run_experiment_matrix --split-manifest D:\ryjin\fwi_visionfm\data\splits\protocol_v1\protocol_v1_flatfault_indomain.json --output-root D:\ryjin\fwi_visionfm\outputs\protocol_v1_flatfault_indomain --auto-shape --device cpu --epochs 3 --learning-rate 1e-3 --batch-size 2
python -m fwi_visionfm.run_experiment_matrix --split-manifest D:\ryjin\fwi_visionfm\data\splits\protocol_v1\protocol_v1_flat_curve_to_flatfault.json --output-root D:\ryjin\fwi_visionfm\outputs\protocol_v1_flat_curve_to_flatfault --auto-shape --device cpu --epochs 3 --learning-rate 1e-3 --batch-size 2
```

可选补充 FlatVel target 的两组：

```powershell
python -m fwi_visionfm.run_experiment_matrix --split-manifest D:\ryjin\fwi_visionfm\data\splits\protocol_v1\protocol_v1_flatvel_indomain.json --output-root D:\ryjin\fwi_visionfm\outputs\protocol_v1_flatvel_indomain --auto-shape --device cpu --epochs 3 --learning-rate 1e-3 --batch-size 2
python -m fwi_visionfm.run_experiment_matrix --split-manifest D:\ryjin\fwi_visionfm\data\splits\protocol_v1\protocol_v1_curvevel_to_flatvel.json --output-root D:\ryjin\fwi_visionfm\outputs\protocol_v1_curvevel_to_flatvel --auto-shape --device cpu --epochs 3 --learning-rate 1e-3 --batch-size 2
```

汇总 Protocol v1：

```powershell
python -m fwi_visionfm.summarize_protocol_v1 --matrix-root D:\ryjin\fwi_visionfm\outputs --split-dir D:\ryjin\fwi_visionfm\data\splits\protocol_v1 --output D:\ryjin\fwi_visionfm\outputs\protocol_v1_summary.md
```

生成预测示例图：

```powershell
python -m fwi_visionfm.generate_prediction_examples --split-manifest D:\ryjin\fwi_visionfm\data\splits\protocol_v1\protocol_v1_curvevel_indomain.json --experiment-dir D:\ryjin\fwi_visionfm\outputs\protocol_v1_curvevel_indomain\torch_cnn_baseline --model-type torch_cnn_baseline --output-dir D:\ryjin\fwi_visionfm\outputs\protocol_v1_curvevel_indomain\prediction_examples\torch_cnn_baseline --num-samples 3 --device cpu
```

真实 DINOv2 frozen 只建议 CPU 1 epoch 接口 smoke：

```powershell
python -m fwi_visionfm.run_foundation_experiment --data-dir D:\ryjin\fwi_visionfm\data\curvevel_a_subset500 --output-dir D:\ryjin\fwi_visionfm\outputs\real_dinov2_curvevel_frozen_1ep_smoke --depth 70 --width 70 --foundation-backbone vit_small_patch14_dinov2.lvd142m --pretrained --freeze-backbone --image-size 518 --aggregation source_attention --batch-size 1 --epochs 1 --learning-rate 1e-4 --device cpu
```

如果真实 DINOv2 下载或 CPU 推理太慢，应把错误或超时记录到 `D:\ryjin\fwi_visionfm\outputs\real_dinov2_curvevel_frozen_1ep_smoke\README_result.md`，不要用 dummy_dinov2 结果替代真实 DINOv2 结论。

## CPU-Only Post-Training Program Construction

小规模 Protocol v1 训练已经完成。后续默认不再启动大规模训练，CPU 阶段重点转为：

- result indexing
- existing checkpoint evaluation
- prediction/error visualization
- protocol comparison plots
- stage report generation
- config-based pipeline
- bridge/backbone registry
- unit tests

这里也需要和当前主线明确区分：

- 当前 raw OpenFWI small transfer 是 foundation transfer 方向的新入口。
- Protocol v1 是此前 matrix-based cross-family 评估协议。
- 两者后续应通过统一的 `evaluate / summary / report` 脚本收敛到同一结果表。

当前约束：

- `stage=all` 默认不训练。
- `metric_consistency_audit.md` 用于核查 `protocol_v1_summary.md`、`all_test_metrics.csv`、`prediction_examples/test_metrics.csv` 和 `stage_report_cpu_protocol_v1.md` 的指标一致性。
- RMSE 口径必须明确；当前推荐以完整 test split 的 global MSE 后开方作为统一口径。
- 真实 DINOv2 只做 interface smoke。
- 所有结论都必须保留 guardrails。
- `dummy_dinov2` 只是工程接口，不是真实 Vision Foundation Model 结论。
- 当前结果只代表 CPU subset500 Protocol v1 工程验证。

推荐命令：

```powershell
python -m fwi_visionfm.pipeline --config D:\ryjin\configs\cpu_protocol_v1.yaml --stage all
```

This does not launch training by default.

结果索引：

```powershell
python -m fwi_visionfm.index_results --outputs-root D:\ryjin\fwi_visionfm\outputs --output-json D:\ryjin\fwi_visionfm\outputs\results_index.json --output-md D:\ryjin\fwi_visionfm\outputs\results_index.md
```

已有 checkpoint 测试集评估：

```powershell
python -m fwi_visionfm.evaluate_existing_checkpoints --split-dir D:\ryjin\fwi_visionfm\data\splits\protocol_v1 --outputs-root D:\ryjin\fwi_visionfm\outputs --output-dir D:\ryjin\fwi_visionfm\outputs\protocol_v1_eval --device cpu
```

批量预测图：

```powershell
python -m fwi_visionfm.generate_prediction_examples --protocol-dir D:\ryjin\fwi_visionfm\data\splits\protocol_v1 --outputs-root D:\ryjin\fwi_visionfm\outputs --output-subdir prediction_examples --num-samples 8 --device cpu --batch
```

Protocol v1 对比图：

```powershell
python -m fwi_visionfm.plot_protocol_comparison --summary D:\ryjin\fwi_visionfm\outputs\protocol_v1_summary.md --eval-csv D:\ryjin\fwi_visionfm\outputs\protocol_v1_eval\all_test_metrics.csv --output-dir D:\ryjin\fwi_visionfm\outputs\protocol_v1_figures
```

阶段报告：

```powershell
python -m fwi_visionfm.build_stage_report --protocol-summary D:\ryjin\fwi_visionfm\outputs\protocol_v1_summary.md --results-index D:\ryjin\fwi_visionfm\outputs\results_index.json --eval-csv D:\ryjin\fwi_visionfm\outputs\protocol_v1_eval\all_test_metrics.csv --figures-dir D:\ryjin\fwi_visionfm\outputs\protocol_v1_figures --output D:\ryjin\fwi_visionfm\outputs\stage_report_cpu_protocol_v1.md
```

Bridge smoke：

```powershell
python -m fwi_visionfm.bridges.registry --smoke
```

Foundation smoke：

```powershell
python -m fwi_visionfm.foundation_smoke --config D:\ryjin\configs\real_dinov2_smoke.yaml --device cpu
```

## Next step

下一轮更合理的推进顺序是：

1. 修正无 `test-split` 时 `metrics_test.json` fallback 行为；
2. 增加 `evaluate_foundation_checkpoint.py`；
3. 对五组模型统一运行 `test_cross_family.csv`；
4. 生成 `summary_metrics.csv`、`summary_report.md`；
5. 生成 Adapter in-family 和 cross-family `prediction_grid.png`。
