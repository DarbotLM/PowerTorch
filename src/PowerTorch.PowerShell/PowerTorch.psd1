@{
    RootModule = 'PowerTorch.PowerShell.dll'
    ModuleVersion = '0.1.0'
    GUID = '9bbdfc79-bb7c-4760-b8ca-f8fba8d0b0f2'
    Author = 'DarbotLM'
    CompanyName = 'DarbotLM'
    Copyright = '(c) DarbotLM'
    Description = 'Native PowerShell module for the PowerTorch tensor runtime.'
    PowerShellVersion = '7.4'
    CompatiblePSEditions = @('Core')
    CmdletsToExport = @(
        'Get-PowerTorchInfo'
        'Get-PowerTorchDevice'
        'Test-PowerTorch'
        'Get-PowerTorchCheckpoint'
        'New-PowerTorchTensor'
        'Get-PowerTorchTensor'
        'Invoke-PowerTorchOperation'
        'Read-PowerTorchCheckpointTensor'
    )
    FunctionsToExport = @()
    AliasesToExport = @()
    PrivateData = @{
        PSData = @{
            Tags = @('PowerShell', 'DirectML', 'Tensor', 'MachineLearning')
            LicenseUri = 'https://github.com/DarbotLM/toolbox'
            ProjectUri = 'https://github.com/DarbotLM/toolbox'
        }
    }
}
