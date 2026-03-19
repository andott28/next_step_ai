[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ModelName,

    [Parameter(Mandatory = $true)]
    [string]$HybridCheckpoint,

    [string]$PythonExe = ".\verification_env\Scripts\python.exe",
    [string]$OutputRoot = ".\results",
    [string]$LayerSelection = "",
    [switch]$LimitBasisInitToSelectedLayers,

    [int]$MaxNewTokens = 16,
    [int]$ValidationPrefixCount = 128,
    [int]$ValidationPrefixLength = 64,

    [int]$BasisRank = 96,
    [int]$BasisTopK = 12,
    [int]$TopK = 6,
    [int]$ScaBottomBufferLayers = 2,
    [int]$ScaDecodeGuardLayers = 12,
    [int]$DenseRolloutTokens = 8,

    [int]$MaxSamples = 256,
    [int]$MaxSeqLength = 128,
    [int]$BatchSize = 1,
    [int]$Steps = 128,
    [double]$Lr = 1e-5,

    [switch]$SkipTests
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-AbsPath {
    param([Parameter(Mandatory = $true)][string]$PathLike)
    return [System.IO.Path]::GetFullPath((Resolve-Path -LiteralPath $PathLike).Path)
}

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][scriptblock]$Action
    )
    Write-Host ""
    Write-Host "=== $Name ===" -ForegroundColor Cyan
    & $Action
    Write-Host "=== $Name complete ===" -ForegroundColor Green
}

function Resolve-NeuroplasticLibParent {
    param([Parameter(Mandatory = $true)][string]$RootDir)
    $candidate = Get-ChildItem -Path $RootDir -Recurse -Directory -Filter "neuroplastic_lib" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -ne $candidate) {
        return $candidate.Parent.FullName
    }

    $parent = Split-Path -Parent $RootDir
    if ([string]::IsNullOrWhiteSpace($parent)) {
        return $null
    }

    $candidate = Get-ChildItem -Path $parent -Recurse -Directory -Filter "neuroplastic_lib" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -ne $candidate) {
        return $candidate.Parent.FullName
    }
    return $null
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $scriptDir
try {
    if (-not (Test-Path -LiteralPath $PythonExe)) {
        throw "Python executable not found: $PythonExe"
    }
    if (-not (Test-Path -LiteralPath $HybridCheckpoint)) {
        throw "Hybrid checkpoint not found: $HybridCheckpoint"
    }

    $pythonAbs = Resolve-AbsPath -PathLike $PythonExe
    $hybridCkptAbs = Resolve-AbsPath -PathLike $HybridCheckpoint
    $outputRootAbs = [System.IO.Path]::GetFullPath((Join-Path $scriptDir $OutputRoot))
    New-Item -ItemType Directory -Force -Path $outputRootAbs | Out-Null

    $recalOutDir = Join-Path $outputRootAbs "recal_decode_manifold"
    New-Item -ItemType Directory -Force -Path $recalOutDir | Out-Null

    $profileJson = Join-Path $outputRootAbs "sca_layer_profile.json"
    $basisInitPt = Join-Path $outputRootAbs "learned_basis_init_profiled.pt"
    $diagJson = Join-Path $recalOutDir "strict_decode_diagnostic.json"
    $recalStatePt = Join-Path $recalOutDir "sca_recalibrated_state.pt"

    $neuroplasticLibParent = Resolve-NeuroplasticLibParent -RootDir $scriptDir
    $pythonPathEntries = @()
    if (-not [string]::IsNullOrWhiteSpace($neuroplasticLibParent)) {
        $pythonPathEntries += $neuroplasticLibParent
        Write-Host "Using local neuroplastic_lib parent: $neuroplasticLibParent" -ForegroundColor Yellow
    } else {
        Write-Host "Warning: neuroplastic_lib folder not found under repo/parent; continuing with repo-local PYTHONPATH." -ForegroundColor Yellow
    }
    $pythonPathEntries += $scriptDir
    $pythonPathEntries += (Join-Path $scriptDir "llama3_neuroplastic")
    if ($env:PYTHONPATH) {
        $pythonPathEntries += $env:PYTHONPATH
    }
    $env:PYTHONPATH = ($pythonPathEntries -join ";")

    Invoke-Step -Name "Environment check" -Action {
        & $pythonAbs -c "import importlib.util; print('neuroplastic_lib spec:', importlib.util.find_spec('neuroplastic_lib'))"
    }

    if (-not $SkipTests) {
        Invoke-Step -Name "Targeted tests" -Action {
            & $pythonAbs -m pytest `
                tests/test_sca_sparse_mlp.py `
                tests/test_hybrid_gqa_mamba.py `
                tests/test_strict_decode_metrics.py `
                tests/test_sca_recalibration_decode_manifold.py `
                -q
        }
    }

    Invoke-Step -Name "Layer decode profile" -Action {
        $args = @(
            ".\llama3_neuroplastic\run_sca_layer_decode_profile.py",
            "--checkpoint", $hybridCkptAbs,
            "--output-path", $profileJson,
            "--sca-basis-rank", $BasisRank,
            "--sca-bottom-buffer-layers", $ScaBottomBufferLayers,
            "--sca-decode-guard-layers", $ScaDecodeGuardLayers,
            "--max-new-tokens", $MaxNewTokens,
            "--validation-prefix-count", $ValidationPrefixCount,
            "--validation-prefix-length", $ValidationPrefixLength
        )
        if (-not [string]::IsNullOrWhiteSpace($LayerSelection)) {
            $args += @("--layers", $LayerSelection)
        }
        & $pythonAbs @args
    }

    Invoke-Step -Name "Dense-informed learned basis init" -Action {
        $args = @(
            ".\llama3_neuroplastic\init_learned_basis_from_dense_mlp.py",
            "--model-name", $ModelName,
            "--hybrid-checkpoint", $hybridCkptAbs,
            "--output-path", $basisInitPt,
            "--basis-rank", $BasisRank,
            "--sca-bottom-buffer-layers", $ScaBottomBufferLayers,
            "--sca-decode-guard-layers", $ScaDecodeGuardLayers,
            "--profile-path", $profileJson,
            "--dense-rollout-tokens", $DenseRolloutTokens,
            "--max-samples", $MaxSamples,
            "--max-seq-length", $MaxSeqLength,
            "--batch-size", $BatchSize,
            "--sca-routing-mode", "semantic_latent"
        )
        if ($LimitBasisInitToSelectedLayers -and (-not [string]::IsNullOrWhiteSpace($LayerSelection))) {
            $args += @("--layers", $LayerSelection)
        }
        & $pythonAbs @args
    }

    Invoke-Step -Name "Decode-manifold recalibration" -Action {
        $args = @(
            ".\llama3_neuroplastic\run_sca_recalibration_from_hybrid_baseline.py",
            "--model-name", $ModelName,
            "--hybrid-checkpoint", $hybridCkptAbs,
            "--learned-basis-init-checkpoint", $basisInitPt,
            "--output-dir", $recalOutDir,
            "--recalibration-mode", "decode_manifold_alignment",
            "--sca-routing-mode", "semantic_latent",
            "--sca-bottom-buffer-layers", $ScaBottomBufferLayers,
            "--sca-decode-guard-layers", $ScaDecodeGuardLayers,
            "--basis-rank", $BasisRank,
            "--basis-top-k", $BasisTopK,
            "--top-k", $TopK,
            "--steps", $Steps,
            "--lr", $Lr,
            "--batch-size", $BatchSize,
            "--max-samples", $MaxSamples,
            "--max-seq-length", $MaxSeqLength,
            "--validation-prefix-count", $ValidationPrefixCount,
            "--validation-prefix-length", $ValidationPrefixLength,
            "--no-include-spatial-proj",
            "--no-strict-decode-upper-layer-cap-enabled",
            "--verbose"
        )
        if (-not [string]::IsNullOrWhiteSpace($LayerSelection)) {
            $args += @("--layers", $LayerSelection)
        }
        & $pythonAbs @args
    }

    if (-not (Test-Path -LiteralPath $recalStatePt)) {
        throw "Expected recalibrated state not found: $recalStatePt"
    }

    Invoke-Step -Name "Strict diagnostic wipe" -Action {
        $args = @(
            ".\llama3_neuroplastic\run_sca_diagnostic_wipe.py",
            "--checkpoint", $hybridCkptAbs,
            "--sca-recalibrated-checkpoint", $recalStatePt,
            "--output-json", $diagJson,
            "--sca-routing-mode", "semantic_latent",
            "--sca-basis-rank", $BasisRank,
            "--sca-bottom-buffer-layers", $ScaBottomBufferLayers,
            "--sca-decode-guard-layers", $ScaDecodeGuardLayers,
            "--max-new-tokens", $MaxNewTokens,
            "--rollout-steps", $MaxNewTokens,
            "--validation-prefix-count", $ValidationPrefixCount,
            "--validation-prefix-length", $ValidationPrefixLength
        )
        & $pythonAbs @args
    }

    Write-Host ""
    Write-Host "Pipeline complete." -ForegroundColor Green
    Write-Host "Profile:            $profileJson"
    Write-Host "Basis init:         $basisInitPt"
    Write-Host "Recalibrated state: $recalStatePt"
    Write-Host "Diagnostic report:  $diagJson"
}
finally {
    Pop-Location
}
