# PowerTorch native ABI

## Stability

The native ABI is a C ABI with opaque handles and `uint32_t` error codes.
Managed code must call `pt_abi_get_info` before loading any backend.
Major-version mismatch is fatal. Minor versions must only add fields/functions.

## Ownership

- The native library retains any passed D3D12/COM object while a native device
  owns it.
- Every native handle has an explicit release function.
- Managed owners use `SafeHandle`.
- A queue submission never transfers graph/buffer ownership.
- Completion is reported only through native fences/events; managed code must
  not infer completion from an enqueue call.

## Error model

Functions return:

| Code | Meaning |
|---:|---|
| `0` | success |
| `1` | invalid argument |
| `2` | unsupported backend/device feature |
| `3` | allocation or resource failure |
| `4` | shape/dtype/stride mismatch |
| `5` | graph compilation failure |
| `6` | driver/runtime failure |
| `7` | synchronization/timeout failure |

The caller supplies a bounded UTF-16/UTF-8 message buffer. HRESULT and native
error details are included in structured diagnostics but never cross the ABI
as unmanaged exceptions.

## Tensor descriptor

The ABI carries:

- dtype ID and byte width
- rank and dimensions
- strides in elements
- offset in elements
- storage/resource ID
- memory layout and mutability
- stream/queue generation

Negative strides are rejected in ABI v1. Non-contiguous tensors must be copied
to a supported layout or lowered with an explicit view plan.

## DirectML v1 scope

- Device/adapter enumeration
- D3D12 device and DirectML device creation
- Direct/compute queue ownership and fence synchronization
- Tensor resource allocation and upload/readback
- Compiled-operator/graph submission
- Feature-level, driver, and DirectML-version diagnostics

The native layer does not parse PowerShell objects or safetensors. Managed code
owns validation and orchestration.
