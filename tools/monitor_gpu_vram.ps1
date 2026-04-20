param(
    [string]$OutCsv = "results\throughput_loop\gpu_vram_monitor.csv",
    [int]$IntervalSeconds = 2,
    [string]$StopFile = ""
)

$ErrorActionPreference = "Continue"

$outDir = Split-Path -Parent $OutCsv
if ($outDir -and -not (Test-Path -LiteralPath $outDir)) {
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
}

if (-not (Test-Path -LiteralPath $OutCsv)) {
    "sample_time,gpu_name,gpu_util_pct,mem_util_pct,mem_used_mib,mem_free_mib,mem_total_mib,pstate,power_w,compute_processes" |
        Out-File -FilePath $OutCsv -Encoding utf8
}

while ($true) {
    if ($StopFile -and (Test-Path -LiteralPath $StopFile)) {
        break
    }

    try {
        $gpuLine = & nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.free,memory.total,pstate,power.draw --format=csv,noheader,nounits 2>$null |
            Select-Object -First 1
        if ($gpuLine) {
            $apps = & nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>$null
            $processText = ""
            if ($apps) {
                $processText = (($apps | ForEach-Object { $_.Trim() }) -join " | ").Replace('"', '""')
            }
            "$gpuLine,""$processText""" | Out-File -FilePath $OutCsv -Append -Encoding utf8
        }
    } catch {
        $now = Get-Date -Format "yyyy/MM/dd HH:mm:ss.fff"
        $message = ($_.Exception.Message -replace '"', '""')
        "$now,,,,,,,,,""monitor_error: $message""" | Out-File -FilePath $OutCsv -Append -Encoding utf8
    }

    Start-Sleep -Seconds ([Math]::Max(1, $IntervalSeconds))
}
