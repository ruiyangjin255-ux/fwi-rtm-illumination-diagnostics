# 盐丘模型 RTM 方案 2 成像条件优化实施计划

> **给 agentic workers：** 必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans` 按任务执行。步骤使用 checkbox (`- [ ]`) 语法跟踪。

**目标：** 为现有 Python 声波 RTM 增加接收照明、源-检双照明归一化和 Laplacian 成像候选，并生成方案 2 小规模对比实验。

**架构：** 在 `acoustic_rtm.py` 中扩展结果 dataclass 和成像函数，保持旧字段兼容；新增一个薄 CLI 脚本负责小规模运行、保存候选数组、绘图和中文报告。

**技术栈：** Python 3.9、NumPy、matplotlib、pytest。

---

### 任务 1：测试新增成像候选字段

**文件：**
- 修改：`D:\ryjin\admit_fwi\tests\test_acoustic_rtm.py`

- [x] **步骤 1：写失败测试**

新增测试断言：

- `reverse_time_migrate` 返回 `receiver_illumination`、`source_receiver_normalized_image`、`laplacian_image` 和 `laplacian_normalized_image`。
- 新字段形状为 `(nz, nx)`，且数值有限。
- 多炮串行与并行结果的新字段一致。

- [x] **步骤 2：运行测试确认失败**

命令：`python -m pytest admit_fwi\tests\test_acoustic_rtm.py -q`

预期：失败，因为 dataclass 和函数尚未实现新字段。

### 任务 2：实现成像候选

**文件：**
- 修改：`D:\ryjin\admit_fwi\acoustic_rtm.py`

- [x] **步骤 1：扩展 dataclass**

给 `RTMResult` 和 `MultishotRTMResult` 增加字段：

- `receiver_illumination`
- `source_receiver_normalized_image`
- `laplacian_image`
- `laplacian_normalized_image`

- [x] **步骤 2：扩展 `reverse_time_migrate`**

反传循环中累计接收波照明，并在循环后计算双照明归一化和 Laplacian 候选。

- [x] **步骤 3：扩展串行和并行多炮函数**

逐炮累加接收照明，最终生成多炮双照明归一化和 Laplacian 候选。

- [x] **步骤 4：运行测试**

命令：`python -m pytest admit_fwi\tests\test_acoustic_rtm.py -q`

预期：通过。

### 任务 3：新增方案 2 smoke 对比脚本

**文件：**
- 新增：`D:\ryjin\admit_fwi\run_scheme2_imaging_condition_compare.py`

- [x] **步骤 1：实现 CLI**

默认使用 SEG/Salt 模型，小规模参数运行多炮 RTM，并保存四类成像候选、接收照明、对比图、中文报告和参数 JSON。

- [x] **步骤 2：运行 smoke**

命令：`python -m admit_fwi.run_scheme2_imaging_condition_compare --output-dir admit_fwi\outputs\seg_salt_scheme2_smoke --max-shots 6 --nt 500 --workers 1`

预期：生成 `scheme2_compare.png` 和 `scheme2_report.md`。

### 任务 4：完整验证

**文件：**
- 验证所有相关测试与输出。

- [x] **步骤 1：运行 focused/full tests**

命令：

- `python -m pytest admit_fwi\tests\test_acoustic_rtm.py -q`
- `python -m pytest admit_fwi\tests\test_optimize_existing_salt_result.py -q`

预期：全部通过。

- [x] **步骤 2：视觉检查**

打开 `scheme2_compare.png`，确认四类成像候选均可读，没有空白或明显绘图错误。
