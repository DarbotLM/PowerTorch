# Security

Do not report security issues through public issues when doing so could expose
credentials, model data, or a working exploit.

Report sensitive findings privately through the repository's GitHub security
advisory workflow. Include:

- affected commit and platform
- native/managed component
- minimal non-sensitive reproduction
- expected and observed behavior
- whether the issue reaches untrusted model files or PowerShell input

PowerTorch intentionally rejects arbitrary Python pickle deserialization.
Checkpoint readers are expected to use bounded formats such as safetensors,
ONNX, and an allowlisted TorchScript/archive subset.
