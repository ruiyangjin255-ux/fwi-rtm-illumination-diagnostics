# FWI Code Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 删除 `fwi_visionfm` 中多余的调试、测试、试验代码，同时保留 `v1-v10` 与 `raw OpenFWI + v11-v14` 主线代码和最小必要测试。

**Architecture:** 先按命名模式删除一次性脚本与缓存，再压缩 `tests/` 到基础模块测试加 protocol 最小契约测试，最后验证主线脚本和保留测试仍存在。

**Tech Stack:** PowerShell, filesystem pattern matching

---

### Task 1: Remove One-Off Scripts

**Files:**
- Modify: filesystem only

- [ ] **Step 1: Delete one-off scripts by reviewed patterns**

Run:

```powershell
$targets = @(
  'D:\ryjin\fwi_visionfm\foundation_smoke.py',
  'D:\ryjin\fwi_visionfm\index_legacy_outputs.py',
  'D:\ryjin\fwi_visionfm\generate_openfwi_tiny_report.py',
  'D:\ryjin\fwi_visionfm\plot_ablation.py',
  'D:\ryjin\fwi_visionfm\plot_openfwi_scale_study.py',
  'D:\ryjin\fwi_visionfm\plot_torch_experiment.py',
  'D:\ryjin\fwi_visionfm\plot_training_curves.py',
  'D:\ryjin\fwi_visionfm\report_ablation.py',
  'D:\ryjin\fwi_visionfm\report_openfwi_scale_study.py',
  'D:\ryjin\fwi_visionfm\report_torch_cpu_experiment.py',
  'D:\ryjin\fwi_visionfm\run_torch_experiment.py',
  'D:\ryjin\fwi_visionfm\compare_scale_reports.py',
  'D:\ryjin\fwi_visionfm\compare_experiments.py'
)
foreach($p in $targets){ if(Test-Path $p){ Remove-Item -LiteralPath $p -Force } }
```

- [ ] **Step 2: Delete script-layer debug/probe/smoke/availability/research-progress files**

Run:

```powershell
$patterns = @('*probe*.py','*smoke*.py','preview_*.py','check_*availability*.py','report_v1_to_v*_research_progress.py','generate_v10_*.py','build_computers_geosciences_submission_package.py')
$files = foreach($pattern in $patterns){ Get-ChildItem 'D:\ryjin\fwi_visionfm\scripts' -File -Filter $pattern -ErrorAction SilentlyContinue }
$files | Sort-Object FullName -Unique | ForEach-Object { Remove-Item -LiteralPath $_.FullName -Force }
```

### Task 2: Reduce Tests To Minimal Mainline Set

**Files:**
- Modify: filesystem only

- [ ] **Step 1: Delete caches from tests**

Run:

```powershell
Get-ChildItem 'D:\ryjin\fwi_visionfm\tests' -Directory -Recurse -Filter __pycache__ -ErrorAction SilentlyContinue | ForEach-Object { Remove-Item -LiteralPath $_.FullName -Recurse -Force }
```

- [ ] **Step 2: Delete redundant tests by reviewed patterns**

Run:

```powershell
$patterns = @(
  'test_*smoke.py',
  'test_*probe.py',
  'test_*availability*.py',
  'test_v1_to_v*_report_generation.py',
  'test_v10_*.py',
  'test_*framework_alignment*.py',
  'test_*archive*.py',
  'test_*packag*.py'
)
$files = foreach($pattern in $patterns){ Get-ChildItem 'D:\ryjin\fwi_visionfm\tests' -File -Filter $pattern -ErrorAction SilentlyContinue }
$files | Sort-Object FullName -Unique | ForEach-Object { Remove-Item -LiteralPath $_.FullName -Force }
```

- [ ] **Step 3: Delete additional non-core historical tests**

Run:

```powershell
$targets = @(
  'D:\ryjin\fwi_visionfm\tests\test_bridge_preview.py',
  'D:\ryjin\fwi_visionfm\tests\test_local_mae_ablation_report.py',
  'D:\ryjin\fwi_visionfm\tests\test_local_mae_ablation_summary.py',
  'D:\ryjin\fwi_visionfm\tests\test_local_mae_report.py',
  'D:\ryjin\fwi_visionfm\tests\test_ncs_probe_report.py',
  'D:\ryjin\fwi_visionfm\tests\test_pasd_edge_mask_audit.py',
  'D:\ryjin\fwi_visionfm\tests\test_pasd_gradient_metric_audit.py',
  'D:\ryjin\fwi_visionfm\tests\test_stage_report_protocol_v2_v3_v4.py',
  'D:\ryjin\fwi_visionfm\tests\test_structure_diagnostics_plot.py'
)
foreach($p in $targets){ if(Test-Path $p){ Remove-Item -LiteralPath $p -Force } }
```

### Task 3: Verify Remaining Mainline Structure

**Files:**
- Modify: filesystem only

- [ ] **Step 1: Verify critical mainline scripts still exist**

Run:

```powershell
$targets = @(
  'D:\ryjin\fwi_visionfm\run_experiment_matrix.py',
  'D:\ryjin\fwi_visionfm\summarize_protocol_v1.py',
  'D:\ryjin\fwi_visionfm\run_foundation_experiment.py',
  'D:\ryjin\fwi_visionfm\scripts\build_protocol_v11_matrix.py',
  'D:\ryjin\fwi_visionfm\scripts\run_protocol_v11_visionfm_crossfamily.py',
  'D:\ryjin\fwi_visionfm\scripts\report_protocol_v11_visionfm_crossfamily.py',
  'D:\ryjin\fwi_visionfm\scripts\build_protocol_v14_matrix.py',
  'D:\ryjin\fwi_visionfm\scripts\run_protocol_v14_geometry_aware_trace_bridge.py',
  'D:\ryjin\fwi_visionfm\scripts\report_protocol_v14_geometry_aware_trace_bridge.py'
)
$targets | ForEach-Object { [pscustomobject]@{Path=$_; Exists=Test-Path $_} }
```

- [ ] **Step 2: Verify minimal representative tests still exist**

Run:

```powershell
$targets = @(
  'D:\ryjin\fwi_visionfm\tests\test_bridge.py',
  'D:\ryjin\fwi_visionfm\tests\test_data.py',
  'D:\ryjin\fwi_visionfm\tests\test_model.py',
  'D:\ryjin\fwi_visionfm\tests\test_loss_metrics.py',
  'D:\ryjin\fwi_visionfm\tests\test_protocol_v2_metrics.py',
  'D:\ryjin\fwi_visionfm\tests\test_protocol_v3_selected_matrix.py',
  'D:\ryjin\fwi_visionfm\tests\test_protocol_v4_integrated_report.py',
  'D:\ryjin\fwi_visionfm\tests\test_protocol_v5_final_report.py',
  'D:\ryjin\fwi_visionfm\tests\test_protocol_v6_geometry_aggregation_report.py',
  'D:\ryjin\fwi_visionfm\tests\test_protocol_v7_boundary_auxiliary_report.py',
  'D:\ryjin\fwi_visionfm\tests\test_protocol_v8_report.py',
  'D:\ryjin\fwi_visionfm\tests\test_protocol_v9_report.py',
  'D:\ryjin\fwi_visionfm\tests\test_protocol_v11_report.py',
  'D:\ryjin\fwi_visionfm\tests\test_protocol_v12_report.py',
  'D:\ryjin\fwi_visionfm\tests\test_protocol_v13_report.py',
  'D:\ryjin\fwi_visionfm\tests\test_protocol_v14_report.py'
)
$targets | ForEach-Object { [pscustomobject]@{Path=$_; Exists=Test-Path $_} }
```
