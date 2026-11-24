[CmdletBinding()]
param(
    [ValidateSet('Debug', 'Release')]
    [string]$Configuration = 'Debug',
    [switch]$SkipDirectML
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$projectRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
& dotnet test (Join-Path $projectRoot 'PowerTorch.slnx') `
    --configuration $Configuration --no-restore
if ($LASTEXITCODE -ne 0) {
    throw "PowerTorch .NET tests failed with exit code $LASTEXITCODE."
}

$modulePath = Join-Path $projectRoot 'artifacts\module\PowerTorch.psd1'
if (-not (Test-Path -LiteralPath $modulePath -PathType Leaf)) {
    throw "PowerTorch module is missing. Run scripts\Build-PowerTorch.ps1 first."
}
Import-Module $modulePath -Force
$info = Get-PowerTorchInfo
if ($info.PythonRuntimeRequired) {
    throw 'PowerTorch unexpectedly reports a Python runtime dependency.'
}
if ($info.Version -eq '0.0.0') {
    throw 'PowerTorch module version was not resolved.'
}
$health = if ($SkipDirectML) {
    Test-PowerTorch
} else {
    Test-PowerTorch -DirectML
}
if (-not $health.Healthy) {
    throw "PowerTorch health failed: $($health.Diagnostic)"
}
$checkpoint = Get-PowerTorchCheckpoint `
    -Path (Join-Path (Split-Path -Parent $projectRoot) 'OmniParser\weights\florence31_icon_caption\model.safetensors')
if ($checkpoint.TensorCount -lt 600) {
    throw "Florence checkpoint tensor count is unexpectedly low: $($checkpoint.TensorCount)"
}

$pester = Get-Module -ListAvailable -Name Pester |
    Where-Object Version -ge ([version]'5.0.0') |
    Sort-Object Version -Descending |
    Select-Object -First 1
if ($pester) {
    Import-Module $pester.Path -Force
} else {
    $vendored = Join-Path (Split-Path -Parent (Split-Path -Parent $projectRoot)) `
        'installers\modules\Pester\5.7.1\Pester.psd1'
    if (-not (Test-Path -LiteralPath $vendored -PathType Leaf)) {
        throw "Pester 5 was not found. Run the .NET tests or provide $vendored."
    }
    Import-Module $vendored -Force
}
$pesterResult = Invoke-Pester `
    -Path (Join-Path $projectRoot 'tests\PowerShell') -PassThru
if ($pesterResult.FailedCount -gt 0 -or $pesterResult.TotalCount -eq 0) {
    throw "PowerTorch Pester tests failed: $($pesterResult.FailedCount)."
}

[pscustomobject]@{
    Version = $info.Version
    Healthy = $health.Healthy
    Cpu = $health.Cpu.Backend
    DirectML = $health.DirectML?.Driver
    TensorCount = $checkpoint.TensorCount
    CheckpointBytes = $checkpoint.DataBytes
    PesterPassed = $pesterResult.PassedCount
}
