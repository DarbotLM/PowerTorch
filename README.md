# PowerTorch

PowerTorch is the native PowerShell/.NET tensor runtime planned for Darbot.
The long-term target is broad PyTorch feature compatibility without a Python
runtime. The first implementation slice is a portable DirectML device bridge,
safe tensor/checkpoint metadata, CPU tensor primitives, and a binary PowerShell
module.

## Build

```powershell
pwsh -NoProfile -File .\scripts\Build-PowerTorch.ps1
pwsh -NoProfile -File .\scripts\Test-PowerTorch.ps1
```

## Load the module

```powershell
Import-Module .\artifacts\module\PowerTorch.psd1
Get-PowerTorchInfo
Get-PowerTorchDevice
Test-PowerTorch
```

PyTorch remains a development parity oracle only. PowerTorch runtime and
packaging do not start or require Python.
