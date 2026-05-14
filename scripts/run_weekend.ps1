<#
.SYNOPSIS
    Weekend training queue: trains 6 advanced-GNN variants sequentially,
    evaluates each, captures all logs, and hibernates the PC at the end.

.DESCRIPTION
    Runs three advanced GNN architectures (multi_horizon_gat,
    spatiotemporal_gnn, seq2seq_gnn) with a small and a large variant
    each, all with the weather block (Block H) enabled and
    temporal_window_hours=1. Configs live under configs/weekend/.

    For each run:
      * stdout+stderr of train and evaluate go to logs/<RUN_ID>/<name>_*.log
      * the trained checkpoint is moved from outputs/best_<model_name>.pt
        to outputs/runs/<RUN_ID>/<name>.pt to prevent the same-model
        sibling from clobbering it.
      * a row is added to outputs/runs/<RUN_ID>/summary.{txt,json}.

    A failure in one run logs the error and continues to the next run.

    After all 6 runs, the script hibernates with 'shutdown /h' (unless
    NO_HIBERNATE=1 or -DryRun was passed).

.PARAMETER DryRun
    Print the train/evaluate commands that would be executed, create the
    run/log directories so we know they are writable, and exit. Does NOT
    invoke Python or hibernate.

.PARAMETER NoHibernate
    Skip the final hibernate call. Equivalent to setting
    $env:NO_HIBERNATE=1 before invocation.

.PARAMETER PythonExe
    Override the Python executable used to launch train.py / evaluate.py.
    Defaults to "python" (resolved from PATH).

.EXAMPLE
    # Dry-run preflight: verify the script parses and dirs are writable.
    $env:NO_HIBERNATE=1; .\scripts\run_weekend.ps1 -DryRun

.EXAMPLE
    # Real run, no hibernate (for an interactive test).
    .\scripts\run_weekend.ps1 -NoHibernate

.EXAMPLE
    # Real run, hibernate at the end (the weekend default).
    .\scripts\run_weekend.ps1

.NOTES
    Targets Windows PowerShell 5.1+. Does NOT use && / || / ternary.
    Run from the project root (the directory containing scripts/, configs/, src/).
#>

[CmdletBinding()]
param(
    [switch]$DryRun,
    [switch]$NoHibernate,
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
$RunId       = Get-Date -Format "yyyyMMdd-HHmmss"
$ProjectRoot = (Get-Location).Path
$RunDir      = Join-Path $ProjectRoot "outputs\runs\$RunId"
$LogDir      = Join-Path $ProjectRoot "logs\$RunId"
$OutputsDir  = Join-Path $ProjectRoot "outputs"

New-Item -ItemType Directory -Force -Path $RunDir   | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir   | Out-Null
New-Item -ItemType Directory -Force -Path $OutputsDir | Out-Null

# The 6 runs to execute, in order: small before large so the GPU warms up
# on a smaller model first (easier debug if the first run crashes).
$Runs = @(
    @{ Name = "multi_horizon_gat_small";  ModelName = "multi_horizon_gat" }
    @{ Name = "multi_horizon_gat_large";  ModelName = "multi_horizon_gat" }
    @{ Name = "spatiotemporal_gnn_small"; ModelName = "spatiotemporal_gnn" }
    @{ Name = "spatiotemporal_gnn_large"; ModelName = "spatiotemporal_gnn" }
    @{ Name = "seq2seq_gnn_small";        ModelName = "seq2seq_gnn" }
    @{ Name = "seq2seq_gnn_large";        ModelName = "seq2seq_gnn" }
)

# Resolve NoHibernate from either the switch or the env var so a dry-run
# inside the same shell that set $env:NO_HIBERNATE doesn't trip hibernate.
$HibernateRequested = -not ($NoHibernate -or $DryRun -or $env:NO_HIBERNATE)

# -----------------------------------------------------------------------------
# Pre-flight banner (teed into logs/<RUN_ID>/preflight.log)
# -----------------------------------------------------------------------------
$PreflightLog = Join-Path $LogDir "preflight.log"

function Write-Banner {
    param([string]$Message, [string]$LogPath)
    $line = "=" * 70
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $block = "$line`n[$stamp] $Message`n$line"
    Write-Host $block
    if ($LogPath) {
        Add-Content -Path $LogPath -Value $block
    }
}

function Write-Log {
    param([string]$Message, [string]$LogPath)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$stamp] $Message"
    Write-Host $line
    if ($LogPath) {
        Add-Content -Path $LogPath -Value $line
    }
}

Write-Banner "Weekend training queue -- RUN_ID = $RunId" $PreflightLog
Write-Log "Project root : $ProjectRoot" $PreflightLog
Write-Log "Run dir      : $RunDir" $PreflightLog
Write-Log "Log dir      : $LogDir" $PreflightLog
Write-Log "Python exe   : $PythonExe" $PreflightLog
Write-Log "DryRun       : $DryRun" $PreflightLog
Write-Log "Hibernate    : $HibernateRequested" $PreflightLog

# Python version (best effort; do not fail the whole run if it cannot be invoked yet)
try {
    $pyVersion = & $PythonExe --version 2>&1 | Out-String
    Write-Log ("Python       : " + $pyVersion.Trim()) $PreflightLog
} catch {
    Write-Log ("Python       : [FAILED to invoke '" + $PythonExe + "'] " + $_.Exception.Message) $PreflightLog
}

# GPU info (nvidia-smi is optional; ignore failure on non-CUDA boxes)
try {
    $gpuInfo = & nvidia-smi -L 2>&1 | Out-String
    Write-Log ("GPU(s)       : " + $gpuInfo.Trim()) $PreflightLog
} catch {
    Write-Log "GPU(s)       : [nvidia-smi not available]" $PreflightLog
}

# Git HEAD short SHA (so we can correlate results with code state)
try {
    $gitSha = & git rev-parse --short HEAD 2>&1 | Out-String
    Write-Log ("Git HEAD     : " + $gitSha.Trim()) $PreflightLog
} catch {
    Write-Log "Git HEAD     : [git not available]" $PreflightLog
}

Write-Log ("Planned runs (" + $Runs.Count + "):") $PreflightLog
foreach ($run in $Runs) {
    Write-Log ("  - " + $run.Name + "  (model: " + $run.ModelName + ")") $PreflightLog
}

# Validate config files exist before starting (catch typos early)
$missingConfigs = @()
foreach ($run in $Runs) {
    $cfg = Join-Path $ProjectRoot ("configs\weekend\" + $run.Name + ".yaml")
    if (-not (Test-Path $cfg)) {
        $missingConfigs += $cfg
    }
}
if ($missingConfigs.Count -gt 0) {
    Write-Log "ERROR: missing config files:" $PreflightLog
    foreach ($m in $missingConfigs) { Write-Log ("  - " + $m) $PreflightLog }
    throw ("Aborting: " + $missingConfigs.Count + " config(s) missing.")
}

if ($DryRun) {
    Write-Banner "DRY RUN -- printing commands and exiting." $PreflightLog
    foreach ($run in $Runs) {
        $cfg = "configs\weekend\" + $run.Name + ".yaml"
        $ckptSrc = "outputs\best_" + $run.ModelName + ".pt"
        $ckptDst = "outputs\runs\" + $RunId + "\" + $run.Name + ".pt"
        Write-Log ("TRAIN  : " + $PythonExe + " scripts\train.py --config " + $cfg) $PreflightLog
        Write-Log ("MOVE   : " + $ckptSrc + " -> " + $ckptDst) $PreflightLog
        Write-Log ("EVAL   : " + $PythonExe + " scripts\evaluate.py --checkpoint " + $ckptDst + " --config " + $cfg) $PreflightLog
    }
    Write-Banner "Dry run finished. Inspect the preflight log for details." $PreflightLog
    Write-Log ("Preflight log: " + $PreflightLog) $PreflightLog
    return
}

# -----------------------------------------------------------------------------
# Helper: run a python command, capture stdout+stderr to a log,
# return exit code + duration
# -----------------------------------------------------------------------------
function Invoke-PythonStep {
    param(
        [string[]]$Args,
        [string]$LogPath,
        [string]$StepName,
        [string]$RunLogPath
    )

    Write-Log (">>> " + $StepName + " : " + $PythonExe + " " + ($Args -join ' ')) $RunLogPath

    $start = Get-Date

    # Start-Process with stdout+stderr both pointed to files so we don't lose
    # anything and we get the real exit code back (PS 5.1's 2>&1 on native
    # exes wraps lines in NativeCommandError which corrupts $? on success).
    $errPath = $LogPath + ".err"
    $proc = Start-Process -FilePath $PythonExe `
                          -ArgumentList $Args `
                          -NoNewWindow `
                          -PassThru `
                          -RedirectStandardOutput $LogPath `
                          -RedirectStandardError $errPath `
                          -Wait

    # Merge .err into the main log so summary readers only need one file.
    if (Test-Path $errPath) {
        Add-Content -Path $LogPath -Value "`n----- STDERR -----"
        Get-Content -Path $errPath | Add-Content -Path $LogPath
        Remove-Item -Path $errPath -Force
    }

    $duration = (Get-Date) - $start
    $exitCode = $proc.ExitCode
    $mins = [math]::Round($duration.TotalMinutes, 1)

    if ($exitCode -eq 0) {
        Write-Log ("<<< " + $StepName + " OK (exit=0, " + $mins + " min)") $RunLogPath
    } else {
        Write-Log ("<<< " + $StepName + " FAIL (exit=" + $exitCode + ", " + $mins + " min)") $RunLogPath
    }

    return @{ ExitCode = $exitCode; DurationSec = [int]$duration.TotalSeconds }
}

# -----------------------------------------------------------------------------
# Per-run loop
# -----------------------------------------------------------------------------
$Summary = @()
$RunLogPath = Join-Path $LogDir "_runner.log"

Write-Banner ("Starting " + $Runs.Count + " runs sequentially") $RunLogPath

$queueStart = Get-Date

foreach ($run in $Runs) {
    $name      = $run.Name
    $modelName = $run.ModelName
    $configRel = "configs\weekend\" + $name + ".yaml"
    $trainLog  = Join-Path $LogDir ($name + "_train.log")
    $evalLog   = Join-Path $LogDir ($name + "_eval.log")
    $ckptSrc   = Join-Path $OutputsDir ("best_" + $modelName + ".pt")
    $ckptDst   = Join-Path $RunDir ($name + ".pt")

    $entry = [ordered]@{
        name           = $name
        model_name     = $modelName
        config         = $configRel
        status         = "PENDING"
        train_log      = $trainLog
        eval_log       = $evalLog
        checkpoint     = $ckptDst
        train_seconds  = 0
        eval_seconds   = 0
        error          = $null
        started_at     = (Get-Date -Format "o")
        finished_at    = $null
    }

    Write-Banner ("[" + $name + "] starting") $RunLogPath

    try {
        # --- Train ---
        $trainResult = Invoke-PythonStep `
            -Args @("scripts\train.py", "--config", $configRel) `
            -LogPath $trainLog `
            -StepName "train" `
            -RunLogPath $RunLogPath
        $entry.train_seconds = $trainResult.DurationSec

        if ($trainResult.ExitCode -ne 0) {
            $entry.status = "FAIL_TRAIN"
            $entry.error  = "train.py exit=" + $trainResult.ExitCode + "; see " + $trainLog
            Write-Log ("[" + $name + "] train failed; skipping eval, continuing queue.") $RunLogPath
            $Summary += $entry
            continue
        }

        # --- Move checkpoint into the run dir (prevents sibling clobber) ---
        if (-not (Test-Path $ckptSrc)) {
            $entry.status = "FAIL_NO_CKPT"
            $entry.error  = "Expected checkpoint not found: " + $ckptSrc
            Write-Log ("[" + $name + "] checkpoint missing at " + $ckptSrc + "; skipping eval.") $RunLogPath
            $Summary += $entry
            continue
        }
        Move-Item -Path $ckptSrc -Destination $ckptDst -Force
        Write-Log ("[" + $name + "] checkpoint moved -> " + $ckptDst) $RunLogPath

        # --- Evaluate ---
        $evalResult = Invoke-PythonStep `
            -Args @("scripts\evaluate.py", "--checkpoint", $ckptDst, "--config", $configRel) `
            -LogPath $evalLog `
            -StepName "evaluate" `
            -RunLogPath $RunLogPath
        $entry.eval_seconds = $evalResult.DurationSec

        if ($evalResult.ExitCode -ne 0) {
            $entry.status = "FAIL_EVAL"
            $entry.error  = "evaluate.py exit=" + $evalResult.ExitCode + "; see " + $evalLog
            Write-Log ("[" + $name + "] evaluate failed; continuing queue.") $RunLogPath
        } else {
            $entry.status = "PASS"
            Write-Log ("[" + $name + "] PASS") $RunLogPath
        }
    }
    catch {
        $entry.status = "FAIL_EXCEPTION"
        $entry.error  = $_.Exception.Message
        Write-Log ("[" + $name + "] EXCEPTION: " + $_.Exception.Message) $RunLogPath
    }
    finally {
        $entry.finished_at = (Get-Date -Format "o")
        $Summary += $entry
    }
}

$queueDuration = (Get-Date) - $queueStart

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
$SummaryTxt  = Join-Path $RunDir "summary.txt"
$SummaryJson = Join-Path $RunDir "summary.json"

$passed = ($Summary | Where-Object { $_.status -eq 'PASS' }).Count

$header = @(
    "Weekend training queue -- RUN_ID = $RunId"
    ("Total wall time : " + [math]::Round($queueDuration.TotalHours, 2) + " h")
    ("Total runs      : " + $Summary.Count)
    ("Passed          : " + $passed)
    ""
    "Per-run results:"
) -join "`n"

$rows = $Summary | ForEach-Object {
    $trainMin = [math]::Round($_.train_seconds / 60.0, 1)
    $evalMin  = [math]::Round($_.eval_seconds  / 60.0, 1)
    "  {0,-30}  {1,-14}  train={2,6} min  eval={3,5} min  log={4}" -f `
        $_.name, $_.status, $trainMin, $evalMin, $_.train_log
}

Set-Content -Path $SummaryTxt -Value ($header + "`n" + ($rows -join "`n")) -Encoding utf8

# JSON: pandas-friendly for notebooks/03_results.ipynb
$jsonPayload = [ordered]@{
    run_id          = $RunId
    started_at      = $queueStart.ToString("o")
    finished_at     = (Get-Date).ToString("o")
    total_seconds   = [int]$queueDuration.TotalSeconds
    project_root    = $ProjectRoot
    runs            = $Summary
}
$jsonPayload | ConvertTo-Json -Depth 6 | Set-Content -Path $SummaryJson -Encoding utf8

Write-Banner "Queue finished. Summary written to:" $RunLogPath
Write-Log ("  " + $SummaryTxt)  $RunLogPath
Write-Log ("  " + $SummaryJson) $RunLogPath
Write-Host ""
Get-Content -Path $SummaryTxt | Write-Host

# -----------------------------------------------------------------------------
# Hibernate (unless suppressed)
# -----------------------------------------------------------------------------
if ($HibernateRequested) {
    Write-Banner "Hibernating PC via 'shutdown /h'" $RunLogPath
    # shutdown /h is immediate; no countdown to abort. The runner only
    # reaches here after the summary is on disk, so a Ctrl-C now is moot.
    & shutdown.exe /h
} else {
    $reason = "DryRun=" + $DryRun + ", NoHibernate=" + $NoHibernate + ", NO_HIBERNATE=" + $env:NO_HIBERNATE
    Write-Banner ("Hibernate skipped (" + $reason + ")") $RunLogPath
}
