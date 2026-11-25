# PowerTorch

PowerTorch is a PowerShell-native tensor, autograd, DirectML, ONNX, and
checkpoint runtime for Darbot. It is being built as a no-Python replacement
path for the PyTorch workloads used by Darbot; the current repository contains
the verified foundation, not full PyTorch parity.

[Website](https://darbotlm.github.io/PowerTorch/) ·
[Compatibility matrix](https://darbotlm.github.io/PowerTorch/#compatibility) ·
[Architecture](docs/architecture.md) ·
[Security model](docs/security.md)

## What works now

- .NET 11 managed tensor/storage and shape/stride contracts.
- CPU tensor add, multiply, subtract, divide, sigmoid, sum, and mean.
- Reverse-mode autograd and one-step AdamW for float32/float64.
- Native C++/DirectML device creation, synchronization, and float32
  element-wise add over a D3D12/DirectML compiled operator.
- Safe safetensors metadata parsing and bounded real tensor reads.
- ONNX Runtime DirectML model loading in an isolated native .NET process.
- Binary PowerShell module with device, tensor, checkpoint, ONNX, backward,
  optimizer, and health cmdlets.
- Real-checkpoint verification against the existing Florence safetensors file.

## Verified snapshot

The latest local verification records:

- Release/Debug build: 0 warnings, 0 errors.
- 7 xUnit tests and 7 Pester tests passed.
- DirectML health verified on NVIDIA T1000 8GB.
- Native DirectML add matched CPU on real Florence tensor data.
- Florence safetensors: 665 tensors, 926,065,664 data bytes.

Model-dependent tests intentionally use real approved local assets and are not
run in public CI.

## Build

Requirements:

- Windows 11 x64
- PowerShell 7.6+
- .NET 11 SDK
- Visual Studio 2022 C++ build tools
- CMake 3.20+

```powershell
pwsh -NoProfile -File .\scripts\Build-PowerTorch.ps1 -Configuration Release
pwsh -NoProfile -File .\scripts\Test-PowerTorch.ps1 -Configuration Release
```

## Use the binary PowerShell module

```powershell
Import-Module .\artifacts\module\PowerTorch.psd1

Get-PowerTorchInfo
Get-PowerTorchDevice -Backend DirectML
Test-PowerTorch -DirectML
Get-PowerTorchCheckpoint -Path .\path\to\model.safetensors
Get-PowerTorchOnnx -Path .\path\to\model.onnx
```

Read a real checkpoint tensor, execute a native operation, and inspect the
result:

```powershell
$tensor = Read-PowerTorchCheckpointTensor `
    -Path .\weights\model.safetensors `
    -MaximumBytes 100000

$result = Invoke-PowerTorchOperation `
    -Operation Add `
    -Left $tensor `
    -Right $tensor `
    -Backend DirectML

Get-PowerTorchTensor -Tensor $result
```

## Runtime boundary

PowerTorch does not link to Python, start Python, or require a Python virtual
environment. ONNX Runtime DirectML is hosted through a native .NET CLI
process because in-process ONNX initialization can terminate the PowerShell
host on some Windows runtime combinations. CPU tensor/autograd/optimizer
operations remain in-process.

PyTorch may be used only as an isolated development parity oracle. It is not a
PowerTorch runtime dependency.

## Public status and scope

PowerTorch targets broad compatibility over time, but currently reports gaps
explicitly:

- model construction for YOLO/Florence/causal-LM layers
- broadcasting, indexing, views, higher-order autograd, AMP
- distributed collectives and sharded checkpoints
- DirectML tensor residency/graph compiler parity beyond initial operators
- TorchScript/`.pt` safe archive reconstruction
- end-to-end OmniParser and Unsloth migration

See [compatibility-gaps.md](docs/compatibility-gaps.md) for the source-derived
roadmap and [compatibility.json](docs/compatibility.json) for the machine-readable
matrix.

## Repository layout

```text
src/
  PowerTorch.Abstractions/       dtypes, devices, tensor descriptors
  PowerTorch.Core/               tensor storage and dispatcher
  PowerTorch.Cpu/                CPU kernels
  PowerTorch.DirectML/           managed DirectML facade
  PowerTorch.DirectML.Native/    C++/DirectML C ABI
  PowerTorch.Serialization/      safetensors and ONNX interop
  PowerTorch.Autograd/           reverse-mode graph
  PowerTorch.Optim/              AdamW
  PowerTorch.PowerShell/         binary PowerShell cmdlets
  PowerTorch.Cli/                isolated native ONNX host
scripts/                         build/test/pages automation
tests/                           xUnit and Pester tests
docs/                            architecture, ABI, security, compatibility
site/                            GitHub Pages source
```

## License

MIT. See [LICENSE](LICENSE).
