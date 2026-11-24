# PowerTorch serialization security

- PowerTorch does not execute arbitrary Python pickle payloads.
- Safetensors parsing uses an 8-byte bounded header length, JSON validation,
  dtype/shape validation, integer overflow checks, offset ordering, and a
  final-data-bound check before mapping.
- ONNX and TorchScript-like archives are treated as untrusted. Unknown custom
  operators, dynamic libraries, executable bytecode, or custom class records
  are rejected.
- Tensor memory is mapped read-only unless an explicit writable copy is
  requested.
- Native model packages contain a manifest with ABI, backend, driver family,
  graph/operator list, hashes, and source checkpoint hashes.
- Module and native DLL loading is restricted to package-owned runtime
  directories. Python-created environment shims are not used.
