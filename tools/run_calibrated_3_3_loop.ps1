param(
    [double]$TargetTokS = 3.30,
    [int]$MaxAttempts = 5,
    [string]$OutDir = "results\throughput_loop",
    [string]$MonitorStopFile = ""
)

$ErrorActionPreference = "Continue"

if (-not (Test-Path -LiteralPath $OutDir)) {
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
}

$activate = Join-Path (Get-Location) "verification_env\Scripts\Activate.ps1"
if (Test-Path -LiteralPath $activate) {
    . $activate
}

$env:STREAMING_GPU_LM_HEAD = "1"
$env:STREAMING_BACKGROUND_PREFETCH = "1"
$env:STREAMING_WINDOWS_BATCH_PRELOAD = "1"
$env:STREAMING_SHOW_PROGRESS = "0"
$env:STREAMING_ALLOW_TRITON_PRE_AMPERE = "1"
$env:SCA_TRITON_ALLOW_WINDOWS_PRE_AMPERE = "1"
$env:SCA_TRITON_MAX_FUSED_TOPK = "64"

$summaryPath = Join-Path $OutDir "summary.csv"
$runLogPath = Join-Path $OutDir "loop_controller.log"

function Write-ControllerLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "o"), $Message
    $line | Tee-Object -FilePath $runLogPath -Append
}

function Add-SummaryRow {
    param([pscustomobject]$Row)
    if (Test-Path -LiteralPath $summaryPath) {
        $Row | Export-Csv -Path $summaryPath -Append -NoTypeInformation
    } else {
        $Row | Export-Csv -Path $summaryPath -NoTypeInformation
    }
}

$candidates = @(
    [ordered]@{ HotCacheGB = 4.00; CalibrationTokens = 8; SparseTopK = 40; AttnHeads = 9; MaxNewTokens = 3; HotTargetBlocks = 40 },
    [ordered]@{ HotCacheGB = 3.50; CalibrationTokens = 8; SparseTopK = 32; AttnHeads = 9; MaxNewTokens = 3; HotTargetBlocks = 32 },
    [ordered]@{ HotCacheGB = 3.00; CalibrationTokens = 8; SparseTopK = 24; AttnHeads = 9; MaxNewTokens = 3; HotTargetBlocks = 24 },
    [ordered]@{ HotCacheGB = 4.00; CalibrationTokens = 8; SparseTopK = 51; AttnHeads = 9; MaxNewTokens = 3; HotTargetBlocks = 0 },
    [ordered]@{ HotCacheGB = 4.25; CalibrationTokens = 8; SparseTopK = 51; AttnHeads = 9; MaxNewTokens = 3; HotTargetBlocks = 0 }
)

$attempt = 0
$targetReached = $false
$skipAboveGb = $null

try {
    Write-ControllerLog "Starting calibrated 3.3 tok/s loop. target=$TargetTokS max_attempts=$MaxAttempts out_dir=$OutDir"

    while ($attempt -lt $MaxAttempts -and -not $targetReached) {
        foreach ($candidate in $candidates) {
            if ($attempt -ge $MaxAttempts -or $targetReached) {
                break
            }

            $hotCacheGb = [double]$candidate.HotCacheGB
            if ($skipAboveGb -ne $null -and $hotCacheGb -gt [double]$skipAboveGb) {
                Write-ControllerLog ("Skipping hot_cache_gb={0:N2}; previous VRAM pressure at {1:N2} GB." -f $hotCacheGb, [double]$skipAboveGb)
                continue
            }

            $attempt += 1
            $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
            $hotLabel = ("{0:N2}" -f $hotCacheGb).Replace(".", "_")
            $jsonPath = Join-Path $OutDir ("attempt_{0:000}_{1}_hc{2}_k{3}_heads{4}.json" -f $attempt, $stamp, $hotLabel, $candidate.SparseTopK, $candidate.AttnHeads)
            $logPath = Join-Path $OutDir ("attempt_{0:000}_{1}_hc{2}_k{3}_heads{4}.log" -f $attempt, $stamp, $hotLabel, $candidate.SparseTopK, $candidate.AttnHeads)
            $startedAt = Get-Date -Format "o"

            $env:STREAMING_VRAM_HOT_CACHE_GB = ("{0:N2}" -f $hotCacheGb)
            if ([int]$candidate.HotTargetBlocks -gt 0) {
                $env:STREAMING_HOT_CACHE_TARGET_BLOCKS = [string]$candidate.HotTargetBlocks
            } else {
                Remove-Item Env:\\STREAMING_HOT_CACHE_TARGET_BLOCKS -ErrorAction SilentlyContinue
            }
            Write-ControllerLog ("Attempt {0}: hot_cache_gb={1:N2} calib_tokens={2} sparse_top_k={3} attn_heads={4} hot_target_blocks={5}" -f $attempt, $hotCacheGb, $candidate.CalibrationTokens, $candidate.SparseTopK, $candidate.AttnHeads, $candidate.HotTargetBlocks)

            $pythonArgs = @(
                "-m", "llama3_neuroplastic.experiments.run_streaming_inference",
                "--model-name", "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit",
                "--local-files-only",
                "--taylor-layers", "none",
                "--sparse-basis-path", "results/mlp_basis_intermediate_full126.pt",
                "--sparse-mlp-execution", "exact_intermediate_sparse",
                "--sparse-top-k", ([string]$candidate.SparseTopK),
                "--sparse-basis-top-k", "64",
                "--sparse-mlp-prefill-mode", "hot_cache",
                "--vram-hot-cache-gb", ([string]$hotCacheGb),
                "--pre-warm",
                "--calibrate-hot-cache",
                "--hot-cache-calibration-tokens", ([string]$candidate.CalibrationTokens),
                "--hot-cache-calibration-prompt", "What is the capital of France? Answer with one word.",
                "--attn-head-importance-path", "results/attn_head_importance_405b.pt",
                "--attn-active-heads", ([string]$candidate.AttnHeads),
                "--attn-min-active-heads", ([string]$candidate.AttnHeads),
                "--attn-max-active-heads", ([string]$candidate.AttnHeads),
                "--sparse-attn-prefill-mode", "sparse",
                "--kv-basis-path", "results/kv_basis_r32.pt",
                "--sparse-kv-prefill-mode", "sparse",
                "--prompt-format", "chat",
                "--max-new-tokens", ([string]$candidate.MaxNewTokens),
                "--no-stream-output",
                "--dump-json", $jsonPath,
                "--prompt", "What is the capital of France? Answer with one word."
            )

            & python @pythonArgs 2>&1 | Tee-Object -FilePath $logPath
            $exitCode = $LASTEXITCODE
            $finishedAt = Get-Date -Format "o"

            $decodeTokS = 0.0
            $totalTokS = 0.0
            $prefillLatencyS = 0.0
            $decodeLatencyS = 0.0
            $decodeMbPerLayer = 0.0
            $overallMbPerLayer = 0.0
            $completionText = ""
            $status = "json_missing"

            if (Test-Path -LiteralPath $jsonPath) {
                try {
                    $payload = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json
                    $row = $payload.rows[0]
                    $decodeTokS = [double]$row.decode_tok_s
                    $totalTokS = [double]$row.total_tok_s
                    $prefillLatencyS = [double]$row.prefill_latency_s
                    $decodeLatencyS = [double]$row.decode_latency_s
                    $completionText = [string]$row.completion_text
                    if ($row.traffic -and $row.traffic.decode) {
                        $decodeMbPerLayer = [double]$row.traffic.decode.avg_mb_per_layer
                    }
                    if ($row.traffic -and $row.traffic.overall) {
                        $overallMbPerLayer = [double]$row.traffic.overall.avg_mb_per_layer
                    }
                    $status = "below_target"
                    if ($decodeTokS -ge $TargetTokS) {
                        $status = "target_reached"
                        $targetReached = $true
                    }
                } catch {
                    $status = "json_parse_error"
                }
            }

            $logText = ""
            if (Test-Path -LiteralPath $logPath) {
                $logText = Get-Content -LiteralPath $logPath -Raw
            }
            if ($exitCode -ne 0 -and $status -ne "target_reached") {
                $status = "process_failed"
            }
            if ($logText -match "CUDA out of memory|VRAM hot-cache disabled|cuda_oom|auto-clamp") {
                if ($status -ne "target_reached") {
                    $status = "vram_pressure"
                }
                if ($skipAboveGb -eq $null -or $hotCacheGb -lt [double]$skipAboveGb) {
                    $skipAboveGb = $hotCacheGb
                }
            }

            Add-SummaryRow ([pscustomobject]@{
                attempt = $attempt
                started_at = $startedAt
                finished_at = $finishedAt
                exit_code = $exitCode
                status = $status
                hot_cache_gb = $hotCacheGb
                calibration_tokens = [int]$candidate.CalibrationTokens
                sparse_top_k = [int]$candidate.SparseTopK
                attn_heads = [int]$candidate.AttnHeads
                hot_target_blocks = [int]$candidate.HotTargetBlocks
                max_new_tokens = [int]$candidate.MaxNewTokens
                decode_tok_s = $decodeTokS
                total_tok_s = $totalTokS
                prefill_latency_s = $prefillLatencyS
                decode_latency_s = $decodeLatencyS
                decode_mb_per_layer = $decodeMbPerLayer
                overall_mb_per_layer = $overallMbPerLayer
                completion_text = $completionText
                json_path = $jsonPath
                log_path = $logPath
            })

            Write-ControllerLog ("Attempt {0} finished: status={1} decode_tok_s={2:N3} decode_mb_layer={3:N2} completion='{4}'" -f $attempt, $status, $decodeTokS, $decodeMbPerLayer, $completionText)
        }
    }

    if ($targetReached) {
        Write-ControllerLog "Target reached. Loop complete."
        exit 0
    }

    Write-ControllerLog "Target not reached within max attempts. Loop complete."
    exit 2
} finally {
    if ($MonitorStopFile) {
        New-Item -ItemType File -Force -Path $MonitorStopFile | Out-Null
    }
}
