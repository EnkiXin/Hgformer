<#
.SYNOPSIS
Run the Amazon-CD SL8 row-mean + LieBN layer screen on one Windows GPU.

.DESCRIPTION
This native PowerShell runner launches one Python process at a time, exposes
only the selected physical GPU, and always passes --validation-only.  The
formal screen covers L=2,4,6,8.  A result is reused only when its JSON is
complete, contains validation metrics, contains no test metrics, and reports
the requested Karcher/LieBN layer structure.

Use -Smoke first for an L2, one-epoch integration check.  A formal run must
explicitly request 500 epochs, validation every 10 epochs, and two validation
checks without improvement for early stopping.
#>

[CmdletBinding()]
param(
    [switch]$Smoke,
    [switch]$AcceleratedPrefilter,
    [ValidateRange(1, 10000000)]
    [int]$PrefilterCandidates = 4096,
    [ValidateRange(1, 100000)]
    [int]$Epochs = 50,
    [ValidateRange(0, 100000)]
    [int]$EvalStep = 0,
    [ValidateRange(0, 100000)]
    [int]$StoppingStep = 1000,
    [ValidateRange(1, 2147483647)]
    [int]$BatchSize = 16384,
    [int[]]$LayerGrid = @(2, 4, 6, 8),
    [double]$LearningRate = 0.005,
    [double]$LossMargin = 0.1,
    [double]$CoordClip = 0.75,
    [string]$ResultTag = "",
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
$ExpectedLogDomainSqrtSteps = 1
$ExpectedLogDomainSqrtIterations = 12
$ExpectedLogDomainSqrtResidualTolerance = 0.001
$ExpectedLogDomainTailTolerance = 0.001
$ExpectedLogDomainGuardRevision = "db_residual_spectral_tail_v1"

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
    if ($Epochs -ne 500 -or $EvalStep -ne 10 -or $StoppingStep -ne 2) {
        throw "Formal screen requires epochs=500, eval_step=10, stopping_step=2"
    }
    $Layers = @($LayerGrid)
    $RunEpochs = 500
    $RunEvalStep = 10
    $RunKind = "screen"
}

$PrefilterMode = if ($AcceleratedPrefilter) { "frobenius" } else { "none" }
$PrefilterSuffix = if ($AcceleratedPrefilter) {
    "_PFfrobeniusC$PrefilterCandidates"
} else { "" }
$GeometrySuffix = "_SQ1I12R0p001T0p001GV1"

function Test-CompletedValidationResult {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [int]$Layer,
        [Parameter(Mandatory = $true)]
        [int]$ExpectedEpochs,
        [Parameter(Mandatory = $true)]
        [int]$ExpectedEvalStep,
        [Parameter(Mandatory = $true)]
        [int]$ExpectedStoppingStep,
        [Parameter(Mandatory = $true)] [int]$ExpectedBatchSize,
        [Parameter(Mandatory = $true)] [double]$ExpectedLearningRate,
        [Parameter(Mandatory = $true)] [double]$ExpectedLossMargin,
        [Parameter(Mandatory = $true)] [double]$ExpectedCoordClip
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
            "eval_step",
            "stopping_step",
            "gcn_layers", "n_layers", "train_batch_size",
            "learning_rate", "loss_margin", "coord_clip",
            "log_domain_sqrt_steps", "log_domain_sqrt_iterations",
            "log_domain_sqrt_residual_tolerance",
            "log_domain_tail_tolerance",
            "log_domain_guard_revision",
            "eval_prefilter", "eval_prefilter_candidates",
            "best_valid_result",
            "test_result",
            "checkpoint_file",
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
        [int]$Payload.eval_step -ne $ExpectedEvalStep -or
        [int]$Payload.stopping_step -ne $ExpectedStoppingStep -or
        [int]$Payload.gcn_layers -ne $Layer -or
        [int]$Payload.n_layers -ne $Layer -or
        [int]$Payload.train_batch_size -ne $ExpectedBatchSize -or
        [math]::Abs([double]$Payload.learning_rate - $ExpectedLearningRate) -gt 1e-12 -or
        [math]::Abs([double]$Payload.loss_margin - $ExpectedLossMargin) -gt 1e-12 -or
        [math]::Abs([double]$Payload.coord_clip - $ExpectedCoordClip) -gt 1e-12 -or
        [int]$Payload.log_domain_sqrt_steps -ne $ExpectedLogDomainSqrtSteps -or
        [int]$Payload.log_domain_sqrt_iterations -ne $ExpectedLogDomainSqrtIterations -or
        [math]::Abs([double]$Payload.log_domain_sqrt_residual_tolerance - $ExpectedLogDomainSqrtResidualTolerance) -gt 1e-12 -or
        [math]::Abs([double]$Payload.log_domain_tail_tolerance - $ExpectedLogDomainTailTolerance) -gt 1e-12 -or
        $Payload.log_domain_guard_revision -ne $ExpectedLogDomainGuardRevision -or
        $Payload.eval_prefilter -ne $PrefilterMode -or
        [int]$Payload.eval_prefilter_candidates -ne $PrefilterCandidates -or
        $null -eq $Payload.best_valid_result -or
        $null -ne $Payload.test_result -or
        $null -eq $Payload.model_diagnostics
    ) {
        return $false
    }
    if (
        [string]::IsNullOrWhiteSpace([string]$Payload.checkpoint_file) -or
        -not (Test-Path -LiteralPath $Payload.checkpoint_file -PathType Leaf)
    ) {
        return $false
    }
    & $Python -c "import sys,torch; d=torch.load(sys.argv[1],map_location='cpu'); c=d['config']; exp=(int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]),int(sys.argv[6]),float(sys.argv[7]),float(sys.argv[8]),float(sys.argv[9]),1,12,1e-3,1e-3,'db_residual_spectral_tail_v1'); got=(int(c['epochs']),int(c['eval_step']),int(c['stopping_step']),int(c['gcn_layers']),int(c['train_batch_size']),float(c['learning_rate']),float(c['loss_margin']),float(c['coord_clip']),int(c['log_domain_sqrt_steps']),int(c['log_domain_sqrt_iterations']),float(c['log_domain_sqrt_residual_tolerance']),float(c['log_domain_tail_tolerance']),str(c['log_domain_guard_revision'])); assert got==exp,(got,exp)" $Payload.checkpoint_file $ExpectedEpochs $ExpectedEvalStep $ExpectedStoppingStep $Layer $ExpectedBatchSize $ExpectedLearningRate $ExpectedLossMargin $ExpectedCoordClip | Out-Null
    if ($LASTEXITCODE -ne 0) {
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
            $ResultName = if (-not [string]::IsNullOrWhiteSpace($ResultTag)) {
                "$ResultTag.json"
            } else { (
                "{0}_L{1}_B{2}_E{3}_V{4}_S{5}_U{6}_I{7}{8}.json" -f
                $RunKind, $Layer, $BatchSize, $RunEpochs, $RunEvalStep,
                $StoppingStep, $EvalUsers, $EvalItems,
                "$PrefilterSuffix$GeometrySuffix"
            ) }
            $ResultFile = Join-Path $ResolvedOutputDirectory $ResultName
            if (Test-CompletedValidationResult `
                    -Path $ResultFile `
                    -Layer $Layer `
                    -ExpectedEpochs $RunEpochs `
                    -ExpectedEvalStep $RunEvalStep `
                    -ExpectedStoppingStep $StoppingStep `
                    -ExpectedBatchSize $BatchSize `
                    -ExpectedLearningRate $LearningRate `
                    -ExpectedLossMargin $LossMargin `
                    -ExpectedCoordClip $CoordClip) {
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
                "--stopping_step=$StoppingStep",
                "--train_batch_size=$BatchSize",
                "--learning_rate=$LearningRate",
                "--loss_margin=$LossMargin",
                "--coord_clip=$CoordClip",
                "--log_domain_sqrt_steps=$ExpectedLogDomainSqrtSteps",
                "--log_domain_sqrt_iterations=$ExpectedLogDomainSqrtIterations",
                "--log_domain_sqrt_residual_tolerance=$ExpectedLogDomainSqrtResidualTolerance",
                "--log_domain_tail_tolerance=$ExpectedLogDomainTailTolerance",
                "--log_domain_guard_revision=$ExpectedLogDomainGuardRevision",
                # Vendored Config reassigns CUDA_VISIBLE_DEVICES from gpu_id.
                # Pass the physical id again; inside PyTorch the sole visible
                # card is still addressed as cuda:0.
                "--gpu_id=$Gpu",
                "--data_path=$ResolvedDataPath",
                "--full_sort_user_batch_size=$EvalUsers",
                "--eval_user_chunk_size=$EvalUsers",
                "--eval_item_chunk_size=$EvalItems",
                "--sl_score_mode=group_log",
                "--eval_prefilter=$PrefilterMode",
                "--eval_prefilter_candidates=$PrefilterCandidates"
            )
            & $Python @Arguments
            if ($LASTEXITCODE -ne 0) {
                throw "L=$Layer failed with exit code $LASTEXITCODE"
            }
            if (-not (Test-CompletedValidationResult `
                    -Path $ResultFile `
                    -Layer $Layer `
                    -ExpectedEpochs $RunEpochs `
                    -ExpectedEvalStep $RunEvalStep `
                    -ExpectedStoppingStep $StoppingStep `
                    -ExpectedBatchSize $BatchSize `
                    -ExpectedLearningRate $LearningRate `
                    -ExpectedLossMargin $LossMargin `
                    -ExpectedCoordClip $CoordClip)) {
                throw "L=$Layer finished without a valid validation-only result"
            }
            $Validated = Get-Content -LiteralPath $ResultFile -Raw | ConvertFrom-Json
            if (-not (Test-Path -LiteralPath $Validated.checkpoint_file -PathType Leaf)) {
                throw "L=$Layer result has no checkpoint"
            }
            & $Python -c "import sys,torch; d=torch.load(sys.argv[1],map_location='cpu'); c=d['config']; exp=(int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),1,12,1e-3,1e-3,'db_residual_spectral_tail_v1'); got=(int(c['epochs']),int(c['eval_step']),int(c['stopping_step']),int(c['log_domain_sqrt_steps']),int(c['log_domain_sqrt_iterations']),float(c['log_domain_sqrt_residual_tolerance']),float(c['log_domain_tail_tolerance']),str(c['log_domain_guard_revision'])); assert got==exp,(got,exp); print('CHECKPOINT_CONFIG_OK',*got)" $Validated.checkpoint_file $RunEpochs $RunEvalStep $StoppingStep
            if ($LASTEXITCODE -ne 0) { throw "L=$Layer checkpoint config validation failed" }
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
