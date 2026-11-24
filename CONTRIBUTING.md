# Contributing

PowerTorch contributions must preserve native runtime boundaries:

1. No Python runtime dependency.
2. No fabricated/simulated model evidence.
3. No silent device, dtype, or precision fallback.
4. Native handles use explicit ownership and disposal.
5. Checkpoint formats reject executable pickle payloads.
6. Public PowerShell commands are typed, documented, and cancellable.

Before a pull request:

```powershell
pwsh -NoProfile -File .\scripts\Build-PowerTorch.ps1 -Configuration Release
pwsh -NoProfile -File .\scripts\Test-PowerTorch.ps1 -Configuration Release
pwsh -NoProfile -File .\scripts\Build-Pages.ps1
```

Model-dependent tests require approved real assets and may be skipped on CI.
Do not commit model weights, `bin/`, `obj/`, `artifacts/`, logs, checkpoints, or
local DirectML caches.
