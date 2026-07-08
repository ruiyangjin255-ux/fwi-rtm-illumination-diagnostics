# 盐丘模型现有 RTM 结果优化实施计划

> **给 agentic workers：** 必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans` 按任务执行。步骤使用 checkbox (`- [ ]`) 语法跟踪。

**目标：** 构建一个独立脚本，对现有 SEG/Salt RTM full 输出做后处理候选、诊断指标和论文图优化。

**架构：** 新增一个独立 Python 模块，包含纯 NumPy 处理函数、matplotlib 图件输出、中文 Markdown 报告和 CLI。测试覆盖核心数组处理逻辑，绘图和文件读写保持薄层。

**技术栈：** Python 3.9、NumPy、matplotlib、pytest。

---

### 任务 1：添加优化辅助函数测试

**文件：**
- 新增：`D:\ryjin\rtm_acoustic\tests\test_optimize_existing_salt_result.py`
- 新增：`D:\ryjin\rtm_acoustic\optimize_existing_salt_result.py`

- [x] **步骤 1：写失败测试**

测试覆盖：

- 低照明 mask 会清零照明不足的样点。
- 对称显示裁剪保持有限数值并限制输出范围。
- 深度均衡能增强深部弱事件，但保留数组形状。
- 诊断指标包含横向、深度、低照明区和 Laplacian 能量保留信息。
- 候选生成返回命名产品和诊断结论。
- 论文图候选返回保守版本、增强版本和推荐版本。

- [x] **步骤 2：运行测试确认失败**

命令：`python -m pytest rtm_acoustic\tests\test_optimize_existing_salt_result.py -q`

预期：在实现前失败，原因是 `rtm_acoustic.optimize_existing_salt_result` 或新函数不存在。

### 任务 2：实现处理函数

**文件：**
- 新增：`D:\ryjin\rtm_acoustic\optimize_existing_salt_result.py`

- [x] **步骤 1：实现最小处理函数**

实现：

- `robust_symmetric_display`
- `mask_low_illumination`
- `depth_balanced_display`
- `soft_threshold_display`
- `compute_diagnostics`
- `build_candidate_products`
- `make_paper_ready_products`

- [x] **步骤 2：运行辅助函数测试**

命令：`python -m pytest rtm_acoustic\tests\test_optimize_existing_salt_result.py -q`

预期：通过。

### 任务 3：添加 CLI 和输出文件

**文件：**
- 修改：`D:\ryjin\rtm_acoustic\optimize_existing_salt_result.py`

- [x] **步骤 1：添加 CLI**

CLI 读取 full 输出目录，写入 `optimization_compare`，保存候选 `.npy`、`metrics.json`、中文 `optimization_report.md`、总览对比图和论文图目录。

- [x] **步骤 2：在现有 full 输出上运行脚本**

命令：`python -m rtm_acoustic.optimize_existing_salt_result --input-dir rtm_acoustic\outputs\seg_salt_multishot_rtm_padded60_full30m_workers4`

预期：在 `optimization_compare` 下生成对比产物。

### 任务 4：完整验证

**文件：**
- 验证新增和现有测试。

- [x] **步骤 1：运行 focused tests**

命令：`python -m pytest rtm_acoustic\tests\test_optimize_existing_salt_result.py -q`

预期：通过。

- [x] **步骤 2：运行现有 RTM tests**

命令：`python -m pytest rtm_acoustic\tests\test_acoustic_rtm.py -q`

预期：通过。

- [x] **步骤 3：检查生成图件**

打开 `optimization_compare\optimization_compare.png`、`paper_figures\paper_ready_migration.png` 和 `paper_figures\paper_ready_comparison.png`，确认图件存在且内容可读。
