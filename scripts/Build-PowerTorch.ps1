[CmdletBinding()]
param(
    [ValidateSet('Debug', 'Release')]
    [string]$Configuration = 'Debug',
    [string]$OutputPath,
    [switch]$ManagedOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$projectRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
if (-not $OutputPath) {
    $OutputPath = Join-Path $projectRoot 'artifacts\module'
}

if (-not $ManagedOnly) {
    $nativeBuild = Join-Path $projectRoot 'artifacts\native'
    & cmake -S (Join-Path $projectRoot 'src\PowerTorch.DirectML.Native') `
        -B $nativeBuild -A x64
    if ($LASTEXITCODE -ne 0) {
        throw "Native DirectML CMake configure failed with exit code $LASTEXITCODE."
    }
    & cmake --build $nativeBuild --config $Configuration --target PowerTorch.DirectML.Native
    if ($LASTEXITCODE -ne 0) {
        throw "Native DirectML build failed with exit code $LASTEXITCODE."
    }
}

& dotnet restore (Join-Path $projectRoot 'PowerTorch.slnx')
if ($LASTEXITCODE -ne 0) {
    throw "dotnet restore failed with exit code $LASTEXITCODE."
}
& dotnet build (Join-Path $projectRoot 'PowerTorch.slnx') `
    --configuration $Configuration --no-restore
if ($LASTEXITCODE -ne 0) {
    throw "dotnet build failed with exit code $LASTEXITCODE."
}

if (Test-Path -LiteralPath $OutputPath) {
    Remove-Item -LiteralPath $OutputPath -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $OutputPath | Out-Null
$moduleSource = Join-Path $projectRoot "src\PowerTorch.PowerShell\bin\$Configuration\net10.0"
Get-ChildItem -LiteralPath $moduleSource -Force |
    Copy-Item -Destination $OutputPath -Recurse -Force
$manifest = Join-Path $projectRoot 'src\PowerTorch.PowerShell\PowerTorch.psd1'
Copy-Item -LiteralPath $manifest -Destination (Join-Path $OutputPath 'PowerTorch.psd1') -Force
if (-not $ManagedOnly) {
    $nativeDll = Join-Path $projectRoot "artifacts\native\$Configuration\PowerTorch.DirectML.Native.dll"
    if (Test-Path -LiteralPath $nativeDll -PathType Leaf) {
        Copy-Item -LiteralPath $nativeDll -Destination $OutputPath -Force
    } else {
        throw "Native DirectML DLL was not produced: $nativeDll"
    }
}

& dotnet publish (Join-Path $projectRoot 'src\PowerTorch.Cli\PowerTorch.Cli.csproj') `
    --configuration $Configuration --runtime win-x64 --self-contained false `
    --output (Join-Path $OutputPath 'bin')
if ($LASTEXITCODE -ne 0) {
    throw "PowerTorch.Cli publish failed with exit code $LASTEXITCODE."
}

Get-ChildItem -LiteralPath $OutputPath -Recurse |
    Select-Object FullName, Length
