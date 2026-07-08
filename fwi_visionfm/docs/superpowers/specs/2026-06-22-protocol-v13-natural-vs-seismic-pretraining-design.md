# Protocol V13 自然图像与地震域预训练确认设计

## 范围

V13 完全继承 V12 的 200/50/50、三 transfer、三 seed、2 epochs、5 炮、70×70 physical velocity、mean aggregation、共同 decoder 和 default L1。M1–M5 仅在严格复用门禁通过后进入 V13；新增训练仅限 M6 NCS2D frozen 的 9 个 run。

## 复用门禁

门禁逐 run 比较 manifest combined SHA256、source/target、sample_id、seed、shot count、bridge、image size、decoder/loss/epochs/metric space、optimizer registration 与 target-test isolation。V13 与 V12 的协议名称和输出目录不同，因此另计算只包含科学配置字段的 reuse-critical hash；该 hash 必须一致。通过的文件除 `config.json` 外使用硬链接，config 复制后只增加 `reused_from` 元数据，不修改 V12 源文件。

## NCS2D 数据流

V13 复用 V12 锁定 manifest。真实 NCS2D 通过本地权重加载，`is_real_feature=True` 是硬门禁。特征按 family/split 共享缓存，缓存写入 sample_id、target、特征和真实特征标志；每个 run 再写独立 `feature_cache_metadata.json`。decoder 在首次真实 feature 前向后构建 optimizer，并核验所有 decoder 参数进入 optimizer。

## 统计与报告

54 个 run 统一汇总绝对指标。泛化差距对 MAE/RMSE/gradient/edge 使用 cross-in，对 SSIM 使用 in-cross。A-E 五个预注册比较按 target sample_id 执行 2000 次 paired bootstrap，并按 transfer 汇总 seed 一致性。中文报告同时讨论绝对跨构造误差和泛化差距，不作 benchmark 或工程级结论。

## 错误处理

任何 V12 关键字段不一致的 run 不得复用，并进入非复用清单。NCS fallback/dummy cache 直接拒绝；manifest hash 变化标记失败；单 run 失败不影响其他 run，支持 resume。

