# PowerTorch architecture

PowerTorch separates stable public contracts from execution backends.

## Layers

1. `PowerTorch.Abstractions`: dtypes, devices, tensor descriptors, ABI
   contracts, backend capability reports.
2. `PowerTorch.Core`: tensor/storage handles, shape/stride validation,
   dispatcher, execution context.
3. `PowerTorch.Cpu`: correctness-first CPU kernels.
4. `PowerTorch.DirectML`: DirectML device and graph execution facade.
5. `PowerTorch.Cuda`: independent native CUDA backend.
6. `PowerTorch.Autograd`, `PowerTorch.Nn`, `PowerTorch.Optim`: eager
   differentiation, modules, and training.
7. `PowerTorch.Serialization`: safetensors, ONNX, and safe TorchScript/archive
   readers.
8. `PowerTorch.Compiler`: typed graph IR, shape inference, fusion, lowering.
9. `PowerTorch.Distributed`: collectives, rank/device mapping, checkpoint
   sharding and recovery.
10. `PowerTorch.PowerShell`: binary cmdlets over the same managed APIs.

ONNX Runtime DirectML is isolated in `PowerTorch.Cli` because loading its
native compile API directly in the PowerShell process can terminate the host.
PowerShell cmdlets invoke this native .NET process and exchange structured JSON;
Python is not involved.

## Backend selection

Backend choice is explicit. No operator silently changes device, dtype, or
precision. The CPU backend is the correctness fallback. DirectML is the first
accelerated backend because it runs on the broad Windows DirectX 12 adapter
surface. CUDA is a separate backend with the same dispatcher contract.

## Runtime boundary

PowerShell imports a binary module. Managed code owns public objects and safe
handles. C++ native libraries own D3D12/DirectML/CUDA resources behind opaque
ABI handles. Python and PyTorch are not linked, loaded, or spawned.

## Compatibility model

The shipped `compatibility.json` reports every supported operation and
checkpoint form as `supported`, `partial`, or `unsupported`. A partial entry
must name its restrictions. An unsupported entry throws a categorized error.

## Initial vertical slices

1. Safe model metadata and tensor loading from Florence safetensors.
2. DirectML device enumeration and synchronized upload/readback.
3. CPU and DirectML tensor execution for the OmniParser post-processing
   operator subset.
4. YOLO TorchScript or converted graph execution.
5. Florence generation, then LoRA + AdamW for one retained BitNet training
   step.
