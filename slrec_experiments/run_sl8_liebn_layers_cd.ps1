<#
.SYNOPSIS
Run the Amazon-CD SL8 row-mean + LieBN layer screen on one Windows GPU.

.DESCRIPTION
This native PowerShell runner launches one Python process at a time, exposes
only the selected physical GPU, and always passes --validation-only.  The
formal screen covers L=2,4,6,8.  A result is reused only when its JSON is
complete, contains validation metrics, contains no test metrics, and reports
the requested Karcher/LieBN layer structure.

Use -Smoke first for an L2, one-epoch integration check.  The default formal
screen trains for 50 epochs and evaluates once, at epoch 50.
#>

[CmdletBinding()]
param(
    [switch]$Smoke,
    [ValidateRange(1, 100000)]
    [int]$Epochs = 50,
    [ValidateRange(0, 100000)]
    [int]$EvalStep = 0,
    [ValidateRange(1, 2147483647)]
    [int]$BatchSize = 16384,
    [ValidateRange(1, 1000000)]
    [int]$EvalUsers = 64,
    [ValidateRange(1, 10000000)]
    [int]$EvalItems = 1024,
    [string]$DataPath = "dataset",
    [ValidateRange(0, 127)]
    [int]$Gpu = 0,
    [string]$Python = "",
    [string]$OutputDirectory = "experiment_runs/sl8_liebn_rowmean_cd"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$BaseConfig = "baseline_config_fixed/SL8LHGCN_cd.yaml"
$MethodOverlay = "baseline_config_fixed/SL8LHGCN_liebn_rowmean_4070ti.yaml"

if ([string]::IsNullOrWhiteSpace($Python)) {
    $VenvCandidates = @(
        (Join-Path $RepoRoot ".venv-hgformer/Scripts/python.exe"),
        (Join-Path (Split-Path $RepoRoot -Parent) ".venv-hgformer/Scripts/python.exe")
    )
    $Python = "python"
    foreach ($VenvPython in $VenvCandidates) {
        if (Test-Path -LiteralPath $VenvPython -PathType Leaf) {
            $Python = $VenvPython
            break
        }
    }
}

if ([System.IO.Path]::IsPathRooted($DataPath)) {
    $ResolvedDataPath = [System.IO.Path]::GetFullPath($DataPath)
}
else {
    $ResolvedDataPath = [System.IO.Path]::GetFullPath(
        (Join-Path $RepoRoot $DataPath)
    )
}
$DatasetFile = Join-Path $ResolvedDataPath "Amazon_cd/Amazon_cd.inter"
if (-not (Test-Path -LiteralPath $DatasetFile -PathType Leaf)) {
    throw ((
            "Amazon-CD is missing at '{0}'. From the repository root run: " +
            "{1} slrec_experiments/prepare_amazon2014.py --domain Amazon_cd " +
            "--output-root `"{2}`""
        ) -f $DatasetFile, $Python, $ResolvedDataPath)
}

if ([System.IO.Path]::IsPathRooted($OutputDirectory)) {
    $ResolvedOutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
}
else {
    $ResolvedOutputDirectory = [System.IO.Path]::GetFullPath(
        (Join-Path $RepoRoot $OutputDirectory)
    )
}
New-Item -ItemType Directory -Force -Path $ResolvedOutputDirectory | Out-Null

if ($Smoke) {
    $Layers = @(2)
    $RunEpochs = 1
    $RunEvalStep = 1
    $RunKind = "smoke"
}
else {
    $Layers = @(2, 4, 6, 8)
    $RunEpochs = $Epochs
    $RunEvalStep = if ($EvalStep -eq 0) { $Epochs } else { $EvalStep }
    $RunKind = "screen"
}

function Test-CompletedValidationResult {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [int]$Layer,
        [Parameter(Mandatory = $true)]
        [int]$ExpectedEpochs
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $false
    }
    try {
        $Payload = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    }
    catch {
        return $false
    }
    if ($null -eq $Payload) {
        return $false
    }

    $Names = @($Payload.PSObject.Properties.Name)
    foreach ($RequiredName in @(
            "model",
            "dataset",
            "epochs",
            "best_valid_result",
            "test_result",
            "model_diagnostics"
        )) {
        if (-not ($Names -contains $RequiredName)) {
            return $false
        }
    }
    if (
        $Payload.model -ne "SL8LHGCN" -or
        $Payload.dataset -ne "Amazon_cd" -or
        [int]$Payload.epochs -ne $ExpectedEpochs -or
        $null -eq $Payload.best_valid_result -or
        $null -ne $Payload.test_result -or
        $null -eq $Payload.model_diagnostics
    ) {
        return $false
    }

    $DiagnosticNames = @($Payload.model_diagnostics.PSObject.Properties.Name)
    foreach ($RequiredName in @("mode", "layers", "layer_membership")) {
        if (-not ($DiagnosticNames -contains $RequiredName)) {
            return $false
        }
    }
    if (
        $Payload.model_diagnostics.mode -ne "karcher1" -or
        [int]$Payload.model_diagnostics.layers -ne $Layer
    ) {
        return $false
    }

    $LayerDiagnostics = @($Payload.model_diagnostics.layer_membership)
    if ($LayerDiagnostics.Count -ne $Layer) {
        return $false
    }
    foreach ($LayerDiagnostic in $LayerDiagnostics) {
        if (
            $null -eq $LayerDiagnostic -or
            -not (@($LayerDiagnostic.PSObject.Properties.Name) -contains "layer_norm")
        ) {
            return $false
        }
    }
    return $true
}

$PreviousCudaVisibleDevices = $env:CUDA_VISIBLE_DEVICES
try {
    # Restrict every child to one physical card. Vendored Config later
    # reasserts the same physical id from --gpu_id; PyTorch addresses that
    # sole visible card as cuda:0.
    $env:CUDA_VISIBLE_DEVICES = [string]$Gpu
    Push-Location $RepoRoot
    try {
        & $Python -c (
            "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; " +
            "assert torch.cuda.device_count() == 1, torch.cuda.device_count(); " +
            "print('CUDA_OK', torch.__version__, torch.cuda.get_device_name(0))"
        )
        if ($LASTEXITCODE -ne 0) {
            throw "CUDA preflight failed with exit code $LASTEXITCODE"
        }

        foreach ($Layer in $Layers) {
            $ResultName = (
                "{0}_L{1}_B{2}_E{3}_V{4}_U{5}_I{6}.json" -f
                $RunKind, $Layer, $BatchSize, $RunEpochs, $RunEvalStep,
                $EvalUsers, $EvalItems
            )
            $ResultFile = Join-Path $ResolvedOutputDirectory $ResultName
            if (Test-CompletedValidationResult `
                    -Path $ResultFile `
                    -Layer $Layer `
                    -ExpectedEpochs $RunEpochs) {
                Write-Host "SKIP completed validation-only result: $ResultFile"
                continue
            }

            Write-Host ((
                    "START {0}: L={1}, batch={2}, epochs={3}, eval_step={4}, " +
                    "eval={5}x{6}, physical GPU={7}"
                ) -f $RunKind, $Layer, $BatchSize, $RunEpochs, $RunEvalStep,
                $EvalUsers, $EvalItems, $Gpu)
            $Arguments = @(
                "run_recbole_gnn.py",
                "--model", "SL8LHGCN",
                "--dataset", "Amazon_cd",
                "--config-files", "$BaseConfig $MethodOverlay",
                "--validation-only",
                "--result-file", $ResultFile,
                "--gcn_layers=$Layer",
                "--n_layers=$Layer",
                "--epochs=$RunEpochs",
                "--eval_step=$RunEvalStep",
                "--stopping_step=1000",
                "--train_batch_size=$BatchSize",
                # Vendored Config reassigns CUDA_VISIBLE_DEVICES from gpu_id.
                # Pass the physical id again; inside PyTorch the sole visible
                # card is still addressed as cuda:0.
                "--gpu_id=$Gpu",
                "--data_path=$ResolvedDataPath",
                "--full_sort_user_batch_size=$EvalUsers",
                "--eval_user_chunk_size=$EvalUsers",
                "--eval_item_chunk_size=$EvalItems",
                "--sl_score_mode=group_log",
                "--eval_prefilter=none"
            )
            & $Python @Arguments
            if ($LASTEXITCODE -ne 0) {
                throw "L=$Layer failed with exit code $LASTEXITCODE"
            }
            if (-not (Test-CompletedValidationResult `
                    -Path $ResultFile `
                    -Layer $Layer `
                    -ExpectedEpochs $RunEpochs)) {
                throw "L=$Layer finished without a valid validation-only result"
            }
            Write-Host "DONE $ResultFile"
        }
    }
    finally {
        Pop-Location
    }
}
finally {
    if ($null -eq $PreviousCudaVisibleDevices) {
        Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
    }
    else {
        $env:CUDA_VISIBLE_DEVICES = $PreviousCudaVisibleDevices
    }
}
