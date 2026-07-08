# FWI Cleanup Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在保留当前研究主线程序、数据索引、主线权重和关键成果图的前提下，清理 `fwi_visionfm` 中无用缓存、旁支输出和主线目录中的中间产物。

**Architecture:** 先删除全局缓存，再清理 `outputs` 中非主线目录，随后对 `protocol_v11-v14` 和 `openfwi_*` 主线目录按文件模式执行保留式清理。最后做目录与关键文件校验，确保主线可追溯资产仍在。

**Tech Stack:** PowerShell, filesystem pattern matching

---

### Task 1: Remove Global Cache

**Files:**
- Modify: filesystem only

- [ ] **Step 1: Remove Python and pytest cache directories**

Run:

```powershell
$targets = @()
$targets += Get-ChildItem -Directory -Recurse -Filter __pycache__ -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName
$targets += Get-ChildItem -Directory -Recurse -Filter .pytest_cache -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName
$targets = $targets | Sort-Object -Unique
foreach($p in $targets){ if(Test-Path $p){ Remove-Item -LiteralPath $p -Recurse -Force } }
```

- [ ] **Step 2: Verify caches are gone**

Run:

```powershell
"__pycache__ count: $((Get-ChildItem -Directory -Recurse -Filter __pycache__ -ErrorAction SilentlyContinue).Count)"
".pytest_cache count: $((Get-ChildItem -Directory -Recurse -Filter .pytest_cache -ErrorAction SilentlyContinue).Count)"
```

Expected: both counts are `0`

### Task 2: Remove Non-Mainline Output Directories

**Files:**
- Modify: filesystem only

- [ ] **Step 1: Delete explicit non-mainline output directories**

Run:

```powershell
$targets = @(
  'D:\ryjin\fwi_visionfm\outputs\legacy_matrix',
  'D:\ryjin\fwi_visionfm\outputs\legacy_reports',
  'D:\ryjin\fwi_visionfm\outputs\ncs_probe',
  'D:\ryjin\fwi_visionfm\outputs\research_progress_report',
  'D:\ryjin\fwi_visionfm\outputs\final_stage_archive',
  'D:\ryjin\fwi_visionfm\outputs\stage_reports'
)
foreach($p in $targets){ if(Test-Path $p){ Remove-Item -LiteralPath $p -Recurse -Force } }
```

- [ ] **Step 2: Verify non-mainline directories are gone**

Run:

```powershell
$targets = @(
  'D:\ryjin\fwi_visionfm\outputs\legacy_matrix',
  'D:\ryjin\fwi_visionfm\outputs\legacy_reports',
  'D:\ryjin\fwi_visionfm\outputs\ncs_probe',
  'D:\ryjin\fwi_visionfm\outputs\research_progress_report',
  'D:\ryjin\fwi_visionfm\outputs\final_stage_archive',
  'D:\ryjin\fwi_visionfm\outputs\stage_reports'
)
$targets | ForEach-Object { [pscustomobject]@{Path=$_; Exists=Test-Path $_} }
```

Expected: every `Exists` is `False`

### Task 3: Clean Mainline Output Directories

**Files:**
- Modify: filesystem only

- [ ] **Step 1: Delete regenerable caches and logs inside mainline directories**

Run:

```powershell
$roots = @(
  'D:\ryjin\fwi_visionfm\outputs\protocol_v11_visionfm_crossfamily',
  'D:\ryjin\fwi_visionfm\outputs\protocol_v12_spectrogram_dinov2_confirmation',
  'D:\ryjin\fwi_visionfm\outputs\protocol_v13_natural_vs_seismic_pretraining',
  'D:\ryjin\fwi_visionfm\outputs\protocol_v14_geometry_aware_trace_bridge',
  'D:\ryjin\fwi_visionfm\outputs\openfwi_small_vision_transfer_3ep',
  'D:\ryjin\fwi_visionfm\outputs\openfwi_small_vision_transfer_3ep_multiseed',
  'D:\ryjin\fwi_visionfm\outputs\openfwi_lora_bridge_multiseed_3ep',
  'D:\ryjin\fwi_visionfm\outputs\openfwi_bridge_ablation_3ep',
  'D:\ryjin\fwi_visionfm\outputs\openfwi_bridge_transfer_interaction_3ep',
  'D:\ryjin\fwi_visionfm\outputs\real_dinov2_smoke'
)
foreach($root in $roots){
  if(Test-Path $root){
    Get-ChildItem -LiteralPath $root -Directory -Recurse -ErrorAction SilentlyContinue | Where-Object { $_.Name -eq 'feature_cache' } | ForEach-Object { Remove-Item -LiteralPath $_.FullName -Recurse -Force }
    Get-ChildItem -LiteralPath $root -File -Recurse -ErrorAction SilentlyContinue | Where-Object {
      $_.Name -like 'predictions_*.npz' -or $_.Extension -eq '.txt'
    } | ForEach-Object { Remove-Item -LiteralPath $_.FullName -Force }
  }
}
```

- [ ] **Step 2: Verify mainline reports and png files still exist**

Run:

```powershell
$roots = @(
  'D:\ryjin\fwi_visionfm\outputs\protocol_v11_visionfm_crossfamily',
  'D:\ryjin\fwi_visionfm\outputs\protocol_v12_spectrogram_dinov2_confirmation',
  'D:\ryjin\fwi_visionfm\outputs\protocol_v13_natural_vs_seismic_pretraining',
  'D:\ryjin\fwi_visionfm\outputs\protocol_v14_geometry_aware_trace_bridge'
)
foreach($root in $roots){
  if(Test-Path $root){
    [pscustomobject]@{
      Root = $root
      MdCount = (Get-ChildItem -LiteralPath $root -File -Recurse -Filter *.md -ErrorAction SilentlyContinue).Count
      CsvCount = (Get-ChildItem -LiteralPath $root -File -Recurse -Filter *.csv -ErrorAction SilentlyContinue).Count
      JsonCount = (Get-ChildItem -LiteralPath $root -File -Recurse -Filter *.json -ErrorAction SilentlyContinue).Count
      PngCount = (Get-ChildItem -LiteralPath $root -File -Recurse -Filter *.png -ErrorAction SilentlyContinue).Count
    }
  }
}
```

Expected: each mainline root still has non-zero `md/csv/json/png` counts where applicable

### Task 4: Final Mainline Integrity Check

**Files:**
- Modify: filesystem only

- [ ] **Step 1: Verify required roots still exist**

Run:

```powershell
$targets = @(
  'D:\ryjin\fwi_visionfm\data\splits',
  'D:\ryjin\fwi_visionfm\weights',
  'D:\ryjin\fwi_visionfm\outputs\protocol_v11_visionfm_crossfamily',
  'D:\ryjin\fwi_visionfm\outputs\protocol_v12_spectrogram_dinov2_confirmation',
  'D:\ryjin\fwi_visionfm\outputs\protocol_v13_natural_vs_seismic_pretraining',
  'D:\ryjin\fwi_visionfm\outputs\protocol_v14_geometry_aware_trace_bridge'
)
$targets | ForEach-Object { [pscustomobject]@{Path=$_; Exists=Test-Path $_} }
```

Expected: every `Exists` is `True`

- [ ] **Step 2: Verify deleted cache patterns are gone**

Run:

```powershell
$roots = @(
  'D:\ryjin\fwi_visionfm\outputs\protocol_v11_visionfm_crossfamily',
  'D:\ryjin\fwi_visionfm\outputs\protocol_v12_spectrogram_dinov2_confirmation',
  'D:\ryjin\fwi_visionfm\outputs\protocol_v13_natural_vs_seismic_pretraining',
  'D:\ryjin\fwi_visionfm\outputs\protocol_v14_geometry_aware_trace_bridge',
  'D:\ryjin\fwi_visionfm\outputs\openfwi_small_vision_transfer_3ep',
  'D:\ryjin\fwi_visionfm\outputs\openfwi_small_vision_transfer_3ep_multiseed',
  'D:\ryjin\fwi_visionfm\outputs\openfwi_lora_bridge_multiseed_3ep',
  'D:\ryjin\fwi_visionfm\outputs\openfwi_bridge_ablation_3ep',
  'D:\ryjin\fwi_visionfm\outputs\openfwi_bridge_transfer_interaction_3ep',
  'D:\ryjin\fwi_visionfm\outputs\real_dinov2_smoke'
)
[pscustomobject]@{
  FeatureCacheDirCount = (Get-ChildItem $roots -Directory -Recurse -ErrorAction SilentlyContinue | Where-Object { $_.Name -eq 'feature_cache' }).Count
  PredictionNpzCount = (Get-ChildItem $roots -File -Recurse -ErrorAction SilentlyContinue | Where-Object { $_.Name -like 'predictions_*.npz' }).Count
  TxtLogCount = (Get-ChildItem $roots -File -Recurse -Filter *.txt -ErrorAction SilentlyContinue).Count
}
```

Expected: all counts are `0`
