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
$OutputDir = Join-Path $RepoRoot "experiment_runs/sl8_liebn_stage_a_e500_61"
$ManifestPath = Join-Path $OutputDir "manifest.json"
$SummaryPath = Join-Path $OutputDir "summary.json"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
& $Python (Join-Path $PSScriptRoot "build_sl8_stage_a_manifest.py") $ManifestPath
if ($LASTEXITCODE -ne 0) { throw "manifest generation failed" }
$Manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
if ([int]$Manifest.cell_count -ne 61 -or -not $Manifest.checks.unique -or -not $Manifest.checks.expected_cells) { throw "Stage A manifest contract failed" }
if ([string]::IsNullOrWhiteSpace($LogPath)) { $LogPath = Join-Path $OutputDir ("stage_a_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss")) }
$Failures = @()
Start-Transcript -LiteralPath $LogPath -Append | Out-Null
try {
    Write-Host "STAGE_A_START cells=61 GPU=$Gpu epochs=500 eval_step=10 stopping_step=2 PF=4096 validation_only=true layers=0,2,4,6,8"
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
                if ($null -eq $Result.test_result) { $Rows += [pscustomobject]@{id=$Cell.id;source=$Cell.source;layer=$Cell.layer;batch=$Cell.batch;learning_rate=$Cell.learning_rate;loss_margin=$Cell.loss_margin;coord_clip_label=$Cell.coord_clip_label;recall10=$Result.best_valid_result.'recall@10';ndcg10=$Result.best_valid_result.'ndcg@10';result=$Path} }
            } catch { }
        }
    }
    $Summary = [pscustomobject]@{generated_at=(Get-Date -Format o);completed=$Rows.Count;failures=$Failures;ranking=@($Rows | Sort-Object @{Expression='recall10';Descending=$true},@{Expression='ndcg10';Descending=$true})}
    $Summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $SummaryPath -Encoding utf8
    Write-Host "STAGE_A_DONE completed=$($Rows.Count) failures=$($Failures.Count) summary=$SummaryPath"
} finally { Stop-Transcript | Out-Null }
