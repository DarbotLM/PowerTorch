using System.Management.Automation;
using System.Reflection;
using PowerTorch.Core;
using PowerTorch.Cpu;
using PowerTorch.DirectML;
using PowerTorch.Serialization;

namespace PowerTorch.PowerShell;

[Cmdlet(VerbsCommon.Get, "PowerTorchInfo")]
[OutputType(typeof(PowerTorchInfo))]
public sealed class GetPowerTorchInfoCommand : PSCmdlet
{
    protected override void ProcessRecord()
    {
        DirectMLAbi? abi = null;
        string? directMLDiagnostic = null;
        try
        {
            abi = DirectMLBackend.GetAbi();
        }
        catch (Exception exception) when (
            exception is DllNotFoundException or BadImageFormatException or EntryPointNotFoundException)
        {
            directMLDiagnostic = exception.Message;
        }

        var assembly = Assembly.GetExecutingAssembly().GetName();
        WriteObject(new PowerTorchInfo(
            "PowerTorch",
            assembly.Version?.ToString() ?? "0.0.0",
            Environment.Version.ToString(),
            "PowerShell binary module",
            abi,
            directMLDiagnostic,
            false));
    }
}

[Cmdlet(VerbsCommon.Get, "PowerTorchDevice")]
[OutputType(typeof(BackendCapability))]
public sealed class GetPowerTorchDeviceCommand : PSCmdlet
{
    [Parameter]
    [ValidateSet("Cpu", "DirectML", "Cuda")]
    public string Backend { get; set; } = "Cpu";

    [Parameter]
    [ValidateRange(0, 15)]
    public int AdapterIndex { get; set; }

    protected override void ProcessRecord()
    {
        switch (Backend)
        {
            case "Cpu":
                WriteObject(new CpuBackend().GetCapabilities());
                return;
            case "Cuda":
                WriteObject(new BackendCapability(
                    Device.Cuda(AdapterIndex),
                    "PowerTorch.Cuda",
                    "native CUDA backend not implemented",
                    false,
                    Array.Empty<string>(),
                    new Dictionary<string, bool>()));
                return;
            case "DirectML":
                try
                {
                    using var backend = DirectMLBackend.Create(AdapterIndex);
                    backend.SelfTest();
                    WriteObject(backend.GetCapabilities());
                }
                catch (PowerTorchException exception)
                {
                    ThrowTerminatingError(new ErrorRecord(
                        exception,
                        "PowerTorch.DirectMLUnavailable",
                        ErrorCategory.ResourceUnavailable,
                        AdapterIndex));
                }
                catch (DllNotFoundException exception)
                {
                    ThrowTerminatingError(new ErrorRecord(
                        exception,
                        "PowerTorch.DirectMLNativeMissing",
                        ErrorCategory.NotInstalled,
                        AdapterIndex));
                }
                return;
            default:
                throw new InvalidOperationException($"Unknown backend '{Backend}'.");
        }
    }
}

[Cmdlet(VerbsDiagnostic.Test, "PowerTorch")]
[OutputType(typeof(PowerTorchHealth))]
public sealed class TestPowerTorchCommand : PSCmdlet
{
    [Parameter]
    public SwitchParameter DirectML { get; set; }

    protected override void ProcessRecord()
    {
        var cpu = new CpuBackend().GetCapabilities();
        BackendCapability? directML = null;
        string? directMLError = null;
        if (DirectML.IsPresent)
        {
            try
            {
                using var backend = DirectMLBackend.Create();
                backend.SelfTest();
                directML = backend.GetCapabilities();
            }
            catch (Exception exception) when (
                exception is PowerTorchException or DllNotFoundException or BadImageFormatException)
            {
                directMLError = exception.Message;
            }
        }

        WriteObject(new PowerTorchHealth(
            cpu.IsAvailable && (directML is not null || !DirectML.IsPresent),
            cpu,
            directML,
            directMLError));
    }
}

[Cmdlet(VerbsCommon.Get, "PowerTorchCheckpoint")]
[OutputType(typeof(SafeTensorMetadata))]
public sealed class GetPowerTorchCheckpointCommand : PSCmdlet
{
    [Parameter(Mandatory = true, Position = 0)]
    [ValidateNotNullOrEmpty]
    public string Path { get; set; } = string.Empty;

    protected override void ProcessRecord()
    {
        string resolvedPath = GetUnresolvedProviderPathFromPSPath(Path);
        try
        {
            using var checkpoint = new SafetensorsFile(resolvedPath);
            WriteObject(new SafeTensorMetadata(
                resolvedPath,
                checkpoint.HeaderLength,
                checkpoint.DataLength,
                checkpoint.Tensors.Count,
                checkpoint.Tensors.Values
                    .Select(tensor => tensor.SourceDataType)
                    .Distinct(StringComparer.Ordinal)
                    .OrderBy(dtype => dtype, StringComparer.Ordinal)
                    .ToArray(),
                checkpoint.MetadataJson));
        }
        catch (Exception exception) when (exception is PowerTorchException or IOException or UnauthorizedAccessException)
        {
            ThrowTerminatingError(new ErrorRecord(
                exception,
                "PowerTorch.CheckpointInvalid",
                ErrorCategory.InvalidData,
                resolvedPath));
        }
    }
}

public sealed record PowerTorchInfo(
    string Name,
    string Version,
    string Runtime,
    string ModuleKind,
    DirectMLAbi? DirectMLAbi,
    string? Diagnostic,
    bool PythonRuntimeRequired);

public sealed record PowerTorchHealth(
    bool Healthy,
    BackendCapability Cpu,
    BackendCapability? DirectML,
    string? Diagnostic);

public sealed record SafeTensorMetadata(
    string Path,
    long HeaderBytes,
    long DataBytes,
    int TensorCount,
    IReadOnlyList<string> DTypes,
    string? MetadataJson);
