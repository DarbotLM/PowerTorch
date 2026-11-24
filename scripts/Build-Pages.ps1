[CmdletBinding()]
param(
    [string]$Commit = '',
    [string]$GeneratedUtc = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$projectRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$siteRoot = Join-Path $projectRoot 'site'
$compatibilityPath = Join-Path $projectRoot 'docs\compatibility.json'
$verificationPath = Join-Path $projectRoot 'docs\verification.json'

foreach ($required in @(
    (Join-Path $siteRoot 'index.html'),
    (Join-Path $siteRoot '404.html'),
    (Join-Path $siteRoot 'assets\styles.css'),
    (Join-Path $siteRoot 'assets\app.js'),
    (Join-Path $siteRoot 'assets\powertorch-mark.svg'),
    $compatibilityPath,
    $verificationPath
)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required Pages input is missing: $required"
    }
}

$compatibility = Get-Content -LiteralPath $compatibilityPath -Raw -Encoding UTF8 |
    ConvertFrom-Json -ErrorAction Stop
$verification = Get-Content -LiteralPath $verificationPath -Raw -Encoding UTF8 |
    ConvertFrom-Json -ErrorAction Stop
if ($compatibility.runtime_python_required) {
    throw 'The public compatibility document cannot claim a Python-free runtime boundary with runtime_python_required=true.'
}
if ($compatibility.families.Count -eq 0) {
    throw 'The compatibility matrix is empty.'
}

Copy-Item -LiteralPath $compatibilityPath `
    -Destination (Join-Path $siteRoot 'compatibility.json') -Force

if ([string]::IsNullOrWhiteSpace($Commit)) {
    $Commit = (& git -C $projectRoot rev-parse HEAD 2>$null).Trim()
    if ($LASTEXITCODE -ne 0 -or $Commit -notmatch '^[0-9a-f]{40}$') {
        $Commit = 'unknown'
    }
}
if ([string]::IsNullOrWhiteSpace($GeneratedUtc)) {
    $GeneratedUtc = [datetime]::UtcNow.ToString('o')
}

$status = [ordered]@{
    commitShort = if ($Commit -eq 'unknown') { 'unknown' } else { $Commit.Substring(0, 8) }
    generatedUtc = $GeneratedUtc
    xunitTests = [int]$verification.xunitTests
    pesterTests = [int]$verification.pesterTests
    buildWarnings = [int]$verification.buildWarnings
    buildErrors = [int]$verification.buildErrors
    directMLAdapter = [string]$verification.directMLAdapter
    safetensorsTensors = [int]$verification.safetensorsTensors
    safetensorsDataBytes = [long]$verification.safetensorsDataBytes
}
$status | ConvertTo-Json -Depth 4 |
    Set-Content -LiteralPath (Join-Path $siteRoot 'status.json') -Encoding UTF8
Set-Content -LiteralPath (Join-Path $siteRoot '.nojekyll') -Value '' -NoNewline

$forbidden = Select-String -Path (Join-Path $siteRoot 'index.html'),
    (Join-Path $siteRoot '404.html'),
    (Join-Path $siteRoot 'assets\*.js'),
    (Join-Path $siteRoot 'assets\*.css') `
    -Pattern 'D:\\|C:\\Users\\|ghp_|github_pat_|BEGIN PRIVATE KEY' -CaseSensitive:$false
if ($forbidden) {
    throw "Pages output contains a machine-local path or secret-like literal: $($forbidden.Path)"
}

$html = Get-Content -LiteralPath (Join-Path $siteRoot 'index.html') -Raw
foreach ($requiredText in @(
    'PowerTorch',
    'id="architecture"',
    'id="capabilities"',
    'id="compatibility"',
    'id="quickstart"',
    'id="roadmap"'
)) {
    if (-not $html.Contains($requiredText)) {
        throw "Pages index is missing required content '$requiredText'."
    }
}
$app = Get-Content -LiteralPath (Join-Path $siteRoot 'assets\app.js') -Raw
if (-not $app.Contains('./compatibility.json')) {
    throw 'Pages app.js does not load ./compatibility.json.'
}

[pscustomobject]@{
    SiteRoot = $siteRoot
    Commit = $status.commitShort
    CompatibilityFamilies = $compatibility.families.Count
    Tests = $status.xunitTests + $status.pesterTests
    GeneratedUtc = $status.generatedUtc
} | Format-List
