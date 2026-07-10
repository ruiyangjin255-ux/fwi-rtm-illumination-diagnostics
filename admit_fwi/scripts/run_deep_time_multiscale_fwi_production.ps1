$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root

$LogDir = Join-Path $Root "admit_fwi\outputs\FWI\deep_time_multiscale_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Model = "admit_fwi\outputs\generated_inputs\seg676x230_from_fwi_true.bin"
$Common = @(
  "--model", $Model,
  "--nt", "5000",
  "--iterations", "2",
  "--max-shots", "0",
  "--optimizer", "cg",
  "--audit-fold", "0",
  "--audit-num-folds", "4",
  "--save-iteration-diagnostics",
  "--num-shot-groups", "4",
  "--workers", "2",
  "--pad-x", "60",
  "--pad-top", "40",
  "--pad-bottom", "80"
)

function Run-Stage {
  param(
    [string]$Name,
    [string]$F0,
    [string]$OutputDir,
    [string]$InitialModel = ""
  )

  $stdout = Join-Path $LogDir "$Name.stdout.log"
  $stderr = Join-Path $LogDir "$Name.stderr.log"
  $args = @("admit_fwi\run_full_salt_fwi.py") + $Common + @("--f0", $F0, "--output-dir", $OutputDir)
  if ($InitialModel -ne "") {
    $args += @("--initial-model", $InitialModel)
  }
  if (Test-Path (Join-Path $OutputDir "checkpoint")) {
    $args += "--resume"
  }

  "[$(Get-Date -Format o)] START $Name f0=$F0 output=$OutputDir" | Tee-Object -FilePath (Join-Path $LogDir "production_status.log") -Append
  $p = Start-Process -FilePath "python" -ArgumentList $args -NoNewWindow -Wait -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
  if ($p.ExitCode -ne 0) {
    "[$(Get-Date -Format o)] FAIL $Name exit=$($p.ExitCode)" | Tee-Object -FilePath (Join-Path $LogDir "production_status.log") -Append
    exit $p.ExitCode
  }
  "[$(Get-Date -Format o)] DONE $Name" | Tee-Object -FilePath (Join-Path $LogDir "production_status.log") -Append
}

$StageA = "admit_fwi\outputs\FWI\full_salt_fwi_deep_time_f4_audit0_train_pmlpad_v1"
$StageB = "admit_fwi\outputs\FWI\full_salt_fwi_deep_time_f6_audit0_train_pmlpad_v1"
$StageC = "admit_fwi\outputs\FWI\full_salt_fwi_deep_time_f8_audit0_train_pmlpad_v1"

Run-Stage -Name "stageA_f4" -F0 "4.0" -OutputDir $StageA
Run-Stage -Name "stageB_f6" -F0 "6.0" -OutputDir $StageB -InitialModel (Join-Path $StageA "full_salt_inverted_model.npy")
Run-Stage -Name "stageC_f8" -F0 "8.0" -OutputDir $StageC -InitialModel (Join-Path $StageB "full_salt_inverted_model.npy")

"[$(Get-Date -Format o)] ALL_DONE" | Tee-Object -FilePath (Join-Path $LogDir "production_status.log") -Append
