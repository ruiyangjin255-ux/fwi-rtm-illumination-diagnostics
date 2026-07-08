# Protocol V12 频谱 DINOv2 确认性评测设计

## 目标与边界

V12 在 V11 统一模型接口上，将 source train 从 100 增加到 200，固定 3 个 transfer、3 个 seed、5 个方法和 2 epochs。主矩阵只比较 CNN、random ViT、DINOv2 frozen、DINOv2-LoRA 与 spectrogram-DINOv2-LoRA，不引入 NCS2D、边界辅助、geometry 或 fusion。

## 架构

1. `build_protocol_v12_manifests.py` 为三个 family 生成固定 200/50/50 CSV，并写入内容哈希和隔离报告。
2. `build_protocol_v12_matrix.py` 读取锁定清单，生成 45 条矩阵、availability、预注册文件和 bridge 样例。
3. V12 runner 复用 V11 模型构建、训练、评估和预测契约；新增 manifest hash 前后核验、config hash、optimizer 参数登记和严格 resume 完整性门禁。
4. bootstrap 对齐 sample_id 后，对 MAE、RMSE、gradient_error、edge_MAE 分别执行 2000 次配对重采样，并生成 seed 一致性表。
5. summary 按五个预注册比较分别判定一致、混合或无一致证据；report 生成逐 transfer 表、8 张中文图与 claims/integrity 文件。

## 数据流与完整性

family CSV 是唯一 sample 清单来源。runner 由 source family 的 train/val/test CSV 和 target family 的 test CSV构造运行 manifest。运行前后重新计算全量 manifest hash；不一致时标记 `FAILED_MANIFEST_MISMATCH`。target test 不进入 dataloader 的 train/val，也不参与模型选择。

## 公平性

M2-M5 使用同一 ViT 尺度、224 输入和共同 decoder。M3/M4 仅 transfer mode 不同；M4/M5 仅 bridge 不同。模型前向初始化 lazy decoder 后再构建 optimizer，并核对 decoder 参数集合完全包含于 optimizer 参数集合。model card 单独记录 encoder、decoder、trainable 和 optimizer 参数量。

## 错误处理与续跑

数据或 backbone 不可用时单 run skip；manifest hash 变化时失败；其他异常写入 `exception.txt`。只有所有必需文件、config hash、manifest hash和 optimizer registration 均有效的成功 run 才可被 resume 复用。

## 测试

测试覆盖 manifest hash/隔离、45 条矩阵、decoder optimizer 注册、target test 隔离、sample_id 配对 bootstrap，以及中文报告与结论边界。真实训练按 Stage A 15 run、Stage B 45 run 顺序执行，Stage A 不用于调参。

