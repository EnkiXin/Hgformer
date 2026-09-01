<# Run deterministic Stage A serially on one GPU; never evaluate test. #>
[CmdletBinding()]
param(
    [ValidateRange(0,127)] [int]$Gpu = 0,
    [string]$Python = "C:\Users\xin57\Documents\Codex\.venv-hgformer\Scripts\python.exe",
    [string]$DataPath = "dataset",
    [string]$LogPath = ""
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ChildRunner = Join-Path $PSScriptRoot "run_sl8_liebn_layers_cd.ps1"
$OutputDir = Join-Path $RepoRoot "experiment_runs/sl8_liebn_stage_a_e500_61_sq1i12r0p001t0p001gv1"
$ManifestPath = Join-Path $OutputDir "manifest.json"
$SummaryPath = Join-Path $OutputDir "summary.json"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
& $Python (Join-Path $PSScriptRoot "build_sl8_stage_a_manifest.py") $ManifestPath
if ($LASTEXITCODE -ne 0) { throw "manifest generation failed" }
$Manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
if (
    [int]$Manifest.cell_count -ne 61 -or
    -not $Manifest.checks.unique -or
    -not $Manifest.checks.expected_cells -or
    [int]$Manifest.protocol.log_domain_sqrt_steps -ne 1 -or
    [int]$Manifest.protocol.log_domain_sqrt_iterations -ne 12 -or
    [math]::Abs([double]$Manifest.protocol.log_domain_sqrt_residual_tolerance - 0.001) -gt 1e-12 -or
    [math]::Abs([double]$Manifest.protocol.log_domain_tail_tolerance - 0.001) -gt 1e-12 -or
    $Manifest.protocol.log_domain_guard_revision -ne "db_residual_spectral_tail_v1"
) { throw "Stage A manifest contract failed" }
if ([string]::IsNullOrWhiteSpace($LogPath)) { $LogPath = Join-Path $OutputDir ("stage_a_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss")) }
$Failures = @()
Start-Transcript -LiteralPath $LogPath -Append | Out-Null
try {
    Write-Host "STAGE_A_START cells=61 GPU=$Gpu epochs=500 eval_step=10 stopping_step=2 PF=4096 SQ=1/12/R0.001/T0.001 validation_only=true layers=0,2,4,6,8"
    foreach ($Cell in $Manifest.cells) {
        Write-Host "CELL_START id=$($Cell.id) source=$($Cell.source) L=$($Cell.layer) batch=$($Cell.batch) lr=$($Cell.learning_rate) margin=$($Cell.loss_margin) clip=$($Cell.coord_clip_label)"
        try {
            & $ChildRunner -Gpu $Gpu -Python $Python -DataPath $DataPath -Epochs 500 -EvalStep 10 -StoppingStep 2 -BatchSize ([int]$Cell.batch) -LayerGrid @([int]$Cell.layer) -LearningRate ([double]$Cell.learning_rate) -LossMargin ([double]$Cell.loss_margin) -CoordClip ([double]$Cell.coord_clip) -ResultTag ([string]$Cell.id) -OutputDirectory $OutputDir -AcceleratedPrefilter -PrefilterCandidates 4096
            if ($LASTEXITCODE -ne 0) { throw "child exit=$LASTEXITCODE" }
            Write-Host "CELL_DONE id=$($Cell.id)"
        } catch {
            $Failures += [pscustomobject]@{ id=$Cell.id; error=$_.Exception.Message }
            Write-Error -ErrorAction Continue "CELL_FAILED id=$($Cell.id) error=$($_.Exception.Message)"
        }
    }
    $Rows = @()
    foreach ($Cell in $Manifest.cells) {
        $Path = Join-Path $OutputDir "$($Cell.id).json"
        if (Test-Path -LiteralPath $Path) {
            try {
                $Result = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
                $Names = @($Result.PSObject.Properties.Name)
                $Required = @(
                    "model", "dataset", "gcn_layers", "train_batch_size",
                    "learning_rate", "loss_margin", "coord_clip",
                    "log_domain_sqrt_steps", "log_domain_sqrt_iterations",
                    "log_domain_sqrt_residual_tolerance",
                    "log_domain_tail_tolerance", "log_domain_guard_revision",
                    "epochs", "eval_step", "stopping_step",
                    "eval_prefilter", "eval_prefilter_candidates",
                    "best_valid_result",
                    "test_result", "checkpoint_file"
                )
                $Complete = @($Required | Where-Object { -not ($Names -contains $_) }).Count -eq 0
                if (
                    $Complete -and
                    $Result.model -eq "SL8LHGCN" -and
                    $Result.dataset -eq "Amazon_cd" -and
                    [int]$Result.gcn_layers -eq [int]$Cell.layer -and
                    [int]$Result.train_batch_size -eq [int]$Cell.batch -and
                    [math]::Abs([double]$Result.learning_rate - [double]$Cell.learning_rate) -le 1e-12 -and
                    [math]::Abs([double]$Result.loss_margin - [double]$Cell.loss_margin) -le 1e-12 -and
                    [math]::Abs([double]$Result.coord_clip - [double]$Cell.coord_clip) -le 1e-12 -and
                    [int]$Result.log_domain_sqrt_steps -eq 1 -and
                    [int]$Result.log_domain_sqrt_iterations -eq 12 -and
                    [math]::Abs([double]$Result.log_domain_sqrt_residual_tolerance - 0.001) -le 1e-12 -and
                    [math]::Abs([double]$Result.log_domain_tail_tolerance - 0.001) -le 1e-12 -and
                    $Result.log_domain_guard_revision -eq "db_residual_spectral_tail_v1" -and
                    [int]$Result.epochs -eq 500 -and
                    [int]$Result.eval_step -eq 10 -and
                    [int]$Result.stopping_step -eq 2 -and
                    $Result.eval_prefilter -eq "frobenius" -and
                    [int]$Result.eval_prefilter_candidates -eq 4096 -and
                    $null -eq $Result.test_result -and
                    $null -ne $Result.best_valid_result -and
                    (Test-Path -LiteralPath $Result.checkpoint_file -PathType Leaf)
                ) {
                    $Rows += [pscustomobject]@{id=$Cell.id;source=$Cell.source;layer=$Cell.layer;batch=$Cell.batch;learning_rate=$Cell.learning_rate;loss_margin=$Cell.loss_margin;coord_clip_label=$Cell.coord_clip_label;recall10=$Result.best_valid_result.'recall@10';ndcg10=$Result.best_valid_result.'ndcg@10';result=$Path}
                }
            } catch { }
        }
    }
    $Summary = [pscustomobject]@{generated_at=(Get-Date -Format o);completed=$Rows.Count;failures=$Failures;ranking=@($Rows | Sort-Object @{Expression='recall10';Descending=$true},@{Expression='ndcg10';Descending=$true})}
    $Summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $SummaryPath -Encoding utf8
    Write-Host "STAGE_A_DONE completed=$($Rows.Count) failures=$($Failures.Count) summary=$SummaryPath"
} finally { Stop-Transcript | Out-Null }
