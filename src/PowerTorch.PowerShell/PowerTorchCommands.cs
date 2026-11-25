using System.Management.Automation;
using System.Diagnostics;
using System.Reflection;
using System.Text.Json;
using PowerTorch.Core;
using PowerTorch.Cpu;
using PowerTorch.Autograd;
using PowerTorch.DirectML;
using PowerTorch.Optim;
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
[OutputType(typeof(SafeTensorMetadata), typeof(ShardedTensorMetadata))]
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
            if (IsCheckpointIndex(resolvedPath))
            {
                using var shardedCheckpoint = new ShardedSafetensorsCheckpoint(resolvedPath);
                SafeTensorDescriptor[] tensors = shardedCheckpoint.TensorNames
                    .Select(shardedCheckpoint.GetTensor)
                    .ToArray();
                WriteObject(new ShardedTensorMetadata(
                    resolvedPath,
                    shardedCheckpoint.ShardPaths.Count,
                    tensors.Length,
                    tensors.Sum(tensor => tensor.ByteLength),
                    tensors
                        .Select(tensor => tensor.SourceDataType)
                        .Distinct(StringComparer.Ordinal)
                        .OrderBy(dtype => dtype, StringComparer.Ordinal)
                        .ToArray(),
                    shardedCheckpoint.MetadataJson));
                return;
            }
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

    internal static bool IsCheckpointIndex(string path) =>
        path.EndsWith(".index.json", StringComparison.OrdinalIgnoreCase);
}

[Cmdlet(
    VerbsCommon.New,
    "PowerTorchTensor",
    DefaultParameterSetName = Float32Set)]
[OutputType(typeof(Tensor))]
public sealed class NewPowerTorchTensorCommand : PSCmdlet
{
    private const string Float32Set = "Float32";
    private const string Float64Set = "Float64";
    private const string Int32Set = "Int32";
    private const string Int64Set = "Int64";

    [Parameter(Mandatory = true, ParameterSetName = Float32Set)]
    public float[] Float32 { get; set; } = [];

    [Parameter(Mandatory = true, ParameterSetName = Float64Set)]
    public double[] Float64 { get; set; } = [];

    [Parameter(Mandatory = true, ParameterSetName = Int32Set)]
    public int[] Int32 { get; set; } = [];

    [Parameter(Mandatory = true, ParameterSetName = Int64Set)]
    public long[] Int64 { get; set; } = [];

    [Parameter]
    [ValidateNotNull]
    public long[] Shape { get; set; } = [];

    protected override void ProcessRecord()
    {
        long[]? shape = Shape.Length == 0 ? null : Shape;
        Tensor tensor = ParameterSetName switch
        {
            Float32Set => TensorFactory.FromArray(Float32, shape),
            Float64Set => TensorFactory.FromArray(Float64, shape),
            Int32Set => TensorFactory.FromArray(Int32, shape),
            Int64Set => TensorFactory.FromArray(Int64, shape),
            _ => throw new InvalidOperationException($"Unknown parameter set '{ParameterSetName}'."),
        };
        WriteObject(tensor);
    }
}

[Cmdlet(VerbsCommon.Get, "PowerTorchTensor")]
[OutputType(typeof(PowerTorchTensorInfo))]
public sealed class GetPowerTorchTensorCommand : PSCmdlet
{
    [Parameter(Mandatory = true, ValueFromPipeline = true)]
    [ValidateNotNull]
    public Tensor Tensor { get; set; } = null!;

    protected override void ProcessRecord()
    {
        WriteObject(new PowerTorchTensorInfo(
            Tensor.DataType.ToString(),
            Tensor.Device.ToString(),
            Tensor.Shape.ToArray(),
            Tensor.Strides.ToArray(),
            Tensor.ElementCount,
            Tensor.IsContiguous));
    }
}

[Cmdlet(VerbsCommunications.Read, "PowerTorchCheckpointTensor")]
[OutputType(typeof(Tensor))]
public sealed class ReadPowerTorchCheckpointTensorCommand : PSCmdlet
{
    [Parameter(Mandatory = true, Position = 0)]
    [ValidateNotNullOrEmpty]
    public string Path { get; set; } = string.Empty;

    [Parameter]
    public string? TensorName { get; set; }

    [Parameter]
    [ValidateRange(1, 2_147_483_647)]
    public long MaximumBytes { get; set; } = 1_000_000;

    protected override void ProcessRecord()
    {
        string resolvedPath = GetUnresolvedProviderPathFromPSPath(Path);
        if (GetPowerTorchCheckpointCommand.IsCheckpointIndex(resolvedPath))
        {
            using var shardedCheckpoint = new ShardedSafetensorsCheckpoint(resolvedPath);
            SafeTensorDescriptor selectedDescriptor;
            if (TensorName is not null)
            {
                selectedDescriptor = shardedCheckpoint.GetTensor(TensorName);
            }
            else
            {
                selectedDescriptor = shardedCheckpoint.TensorNames
                    .Select(shardedCheckpoint.GetTensor)
                    .Where(candidate =>
                        candidate.DataType is DataType.Float32 or DataType.BFloat16 &&
                        candidate.ByteLength <= MaximumBytes)
                    .OrderBy(candidate => candidate.ByteLength)
                    .FirstOrDefault()
                    ?? throw new PowerTorchException(
                        PowerTorchErrorCategory.Serialization,
                        $"No float32 or bfloat16 tensor of at most {MaximumBytes} bytes exists in '{resolvedPath}'.");
            }

            WriteObject(shardedCheckpoint.LoadTensorAsync(
                    selectedDescriptor.Name,
                    MaximumBytes)
                .AsTask()
                .GetAwaiter()
                .GetResult());
            return;
        }
        using var checkpoint = new SafetensorsFile(resolvedPath);
        SafeTensorDescriptor descriptor;
        if (TensorName is not null)
        {
            descriptor = checkpoint.GetTensor(TensorName);
        }
        else
        {
            descriptor = checkpoint.Tensors.Values
                .Where(candidate =>
                    candidate.DataType is DataType.Float32 or DataType.BFloat16 &&
                    candidate.ByteLength <= MaximumBytes)
                .OrderBy(candidate => candidate.ByteLength)
                .FirstOrDefault()
                ?? throw new PowerTorchException(
                    PowerTorchErrorCategory.Serialization,
                    $"No float32 or bfloat16 tensor of at most {MaximumBytes} bytes exists in '{resolvedPath}'.");
        }

        byte[] bytes = checkpoint.ReadTensorBytesAsync(descriptor.Name).AsTask().GetAwaiter().GetResult();
        WriteObject(TensorFactory.FromRawBytes(descriptor.DataType, descriptor.Shape, bytes));
    }
}

[Cmdlet(VerbsCommon.Get, "PowerTorchOnnx")]
[OutputType(typeof(OnnxModelInfo))]
public sealed class GetPowerTorchOnnxCommand : PSCmdlet
{
    private static readonly JsonSerializerOptions JsonOptions =
        new(JsonSerializerDefaults.Web);

    [Parameter(Mandatory = true, Position = 0)]
    [ValidateNotNullOrEmpty]
    public string Path { get; set; } = string.Empty;

    [Parameter]
    public int AdapterIndex { get; set; }

    [Parameter]
    public SwitchParameter Cpu { get; set; }

    protected override void ProcessRecord()
    {
        string resolvedPath = GetUnresolvedProviderPathFromPSPath(Path);
        string? cliPath = ResolvePowerTorchCli();
        if (cliPath is null)
        {
            ThrowTerminatingError(new ErrorRecord(
                new FileNotFoundException("PowerTorch.Cli.exe was not found in the module package."),
                "PowerTorch.CliMissing",
                ErrorCategory.NotInstalled,
                resolvedPath));
            return;
        }

        try
        {
            var startInfo = new ProcessStartInfo
            {
                FileName = cliPath,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            startInfo.ArgumentList.Add("onnx-inspect");
            startInfo.ArgumentList.Add(resolvedPath);
            if (Cpu.IsPresent)
            {
                startInfo.ArgumentList.Add("--cpu");
            }
            else
            {
                startInfo.ArgumentList.Add("--adapter");
                startInfo.ArgumentList.Add(
                    AdapterIndex.ToString(System.Globalization.CultureInfo.InvariantCulture));
            }
            using var process = Process.Start(startInfo) ??
                throw new InvalidOperationException("PowerTorch.Cli process did not start.");
            string stdout = process.StandardOutput.ReadToEnd();
            string stderr = process.StandardError.ReadToEnd();
            process.WaitForExit();
            if (process.ExitCode != 0)
            {
                throw new InvalidOperationException(
                    $"PowerTorch.Cli exited {process.ExitCode}: {stderr.Trim()}");
            }
            var metadata = JsonSerializer.Deserialize<OnnxModelInfo>(
                stdout,
                JsonOptions) ??
                throw new InvalidDataException("PowerTorch.Cli returned no ONNX metadata.");
            WriteObject(metadata);
        }
        catch (Exception exception)
        {
            ThrowTerminatingError(new ErrorRecord(
                exception,
                "PowerTorch.OnnxInvalid",
                ErrorCategory.InvalidData,
                resolvedPath));
        }
    }

    private static string? ResolvePowerTorchCli()
    {
        string moduleDirectory = System.IO.Path.GetDirectoryName(
            Assembly.GetExecutingAssembly().Location) ?? string.Empty;
        string packaged = System.IO.Path.Combine(moduleDirectory, "bin", "PowerTorch.Cli.exe");
        if (File.Exists(packaged))
        {
            return packaged;
        }
        string? configured = Environment.GetEnvironmentVariable("POWERTORCH_CLI");
        return configured is not null && File.Exists(configured) ? configured : null;
    }
}

[Cmdlet(VerbsLifecycle.Invoke, "PowerTorchOperation")]
[OutputType(typeof(Tensor))]
public sealed class InvokePowerTorchOperationCommand : PSCmdlet
{
    [Parameter(Mandatory = true, Position = 0)]
    [ValidateSet(
        "Add",
        "Multiply",
        "Subtract",
        "Divide",
        "Sigmoid",
        "Sum",
        "Mean",
        "Reshape",
        "Permute",
        "Transpose",
        "Slice",
        "Matmul",
        "Softmax",
        "LayerNorm",
        "RmsNorm",
        "Embedding",
        "Conv1D",
        "Rotary",
        "CausalAttention")]
    public string Operation { get; set; } = string.Empty;

    [Parameter(Mandatory = true)]
    [ValidateNotNull]
    public Tensor Left { get; set; } = null!;

    [Parameter]
    public Tensor? Right { get; set; }

    [Parameter]
    public long[] Shape { get; set; } = [];

    [Parameter]
    public int[] Permutation { get; set; } = [];

    [Parameter]
    public int Dimension { get; set; } = -1;

    [Parameter]
    public int SecondDimension { get; set; } = -2;

    [Parameter]
    public long Start { get; set; }

    [Parameter]
    public long? End { get; set; }

    [Parameter]
    [ValidateRange(1, long.MaxValue)]
    public long Step { get; set; } = 1;

    [Parameter]
    public Tensor? Weight { get; set; }

    [Parameter]
    public Tensor? Bias { get; set; }

    [Parameter]
    public double Epsilon { get; set; } = 1e-5;

    [Parameter]
    public Tensor? Positions { get; set; }

    [Parameter]
    public Tensor? Value { get; set; }

    [Parameter]
    [ValidateRange(1, int.MaxValue)]
    public int Stride { get; set; } = 1;

    [Parameter]
    [ValidateRange(0, int.MaxValue)]
    public int Padding { get; set; }

    [Parameter]
    [ValidateRange(1, int.MaxValue)]
    public int Dilation { get; set; } = 1;

    [Parameter]
    [ValidateRange(1, int.MaxValue)]
    public int Groups { get; set; } = 1;

    [Parameter]
    public SwitchParameter Causal { get; set; }

    [Parameter]
    public int SequenceDimension { get; set; } = -2;

    [Parameter]
    [ValidateRange(0, int.MaxValue)]
    public int RotaryDimension { get; set; }

    [Parameter]
    public double BaseFrequency { get; set; } = 10_000;

    [Parameter]
    public SwitchParameter Interleaved { get; set; }

    [Parameter]
    public double? Scale { get; set; }

    [Parameter]
    [ValidateSet("Cpu", "DirectML")]
    public string Backend { get; set; } = "Cpu";

    protected override void ProcessRecord()
    {
        if (Backend == "DirectML")
        {
            if (Operation != "Add")
            {
                throw new PowerTorchException(
                    PowerTorchErrorCategory.Unsupported,
                    "The DirectML host bridge currently supports only Add.");
            }
            var right = RequireRight();
            if (Left.DataType != DataType.Float32 ||
                right.DataType != DataType.Float32 ||
                !Left.Shape.SequenceEqual(right.Shape))
            {
                throw new PowerTorchException(
                    PowerTorchErrorCategory.ShapeMismatch,
                    "DirectML Add currently requires equal-shape Float32 CPU tensors.");
            }
            using var directML = DirectMLBackend.Create();
            float[] values = directML.AddF32(
                Left.AsSpan<float>(),
                right.AsSpan<float>());
            WriteObject(TensorFactory.FromArray(values, Left.Shape));
            return;
        }

        var runtime = new PowerTorchRuntime();
        runtime.Register(new CpuBackend());
        Tensor result = Operation switch
        {
            "Add" => runtime.Add(Left, RequireRight()),
            "Multiply" => runtime.Multiply(Left, RequireRight()),
            "Subtract" => runtime.Subtract(Left, RequireRight()),
            "Divide" => runtime.Divide(Left, RequireRight()),
            "Sigmoid" => runtime.Sigmoid(Left),
            "Sum" => runtime.Sum(Left),
            "Mean" => runtime.Mean(Left),
            "Reshape" => runtime.Reshape(Left, RequireShape()),
            "Permute" => runtime.Permute(Left, RequirePermutation()),
            "Transpose" => runtime.Transpose(Left, Dimension, SecondDimension),
            "Slice" => runtime.Slice(
                Left,
                Dimension,
                Start,
                End ?? Left.Shape[NormalizeDimension(Dimension, Left.Shape.Count)],
                Step),
            "Matmul" => runtime.Matmul(Left, RequireRight()),
            "Softmax" => runtime.Softmax(Left, Dimension),
            "LayerNorm" => runtime.LayerNorm(Left, RequireShape(), Weight, Bias, Epsilon),
            "RmsNorm" => runtime.RmsNorm(Left, RequireShape(), Weight, Epsilon),
            "Embedding" => runtime.Embedding(Left, RequireRight()),
            "Conv1D" => runtime.Conv1D(
                Left,
                RequireRight(),
                Bias,
                Stride,
                Padding,
                Dilation,
                Groups,
                Causal.IsPresent),
            "Rotary" => runtime.Rotary(
                Left,
                RequirePositions(),
                SequenceDimension,
                RotaryDimension,
                BaseFrequency,
                Interleaved.IsPresent),
            "CausalAttention" => runtime.CausalAttention(
                Left,
                RequireRight(),
                RequireValue(),
                Scale),
            _ => throw new InvalidOperationException($"Unknown operation '{Operation}'."),
        };
        WriteObject(result);
    }

    private Tensor RequireRight()
    {
        if (Right is null)
        {
            throw new ArgumentException($"Operation '{Operation}' requires -Right.");
        }
        return Right;
    }

    private long[] RequireShape()
    {
        if (Shape.Length == 0)
        {
            throw new ArgumentException($"Operation '{Operation}' requires -Shape.");
        }
        return Shape;
    }

    private int[] RequirePermutation()
    {
        if (Permutation.Length == 0)
        {
            throw new ArgumentException($"Operation '{Operation}' requires -Permutation.");
        }
        return Permutation;
    }

    private Tensor RequirePositions() =>
        Positions ?? throw new ArgumentException("Operation 'Rotary' requires -Positions.");

    private Tensor RequireValue() =>
        Value ?? throw new ArgumentException("Operation 'CausalAttention' requires -Value.");

    private static int NormalizeDimension(int dimension, int rank)
    {
        int normalized = dimension < 0 ? rank + dimension : dimension;
        return normalized >= 0 && normalized < rank
            ? normalized
            : throw new ArgumentOutOfRangeException(nameof(dimension));
    }
}

[Cmdlet(VerbsCommon.New, "PowerTorchVariable")]
[OutputType(typeof(Variable))]
public sealed class NewPowerTorchVariableCommand : PSCmdlet
{
    [Parameter(Mandatory = true, Position = 0, ValueFromPipeline = true)]
    [ValidateNotNull]
    public Tensor Tensor { get; set; } = null!;

    [Parameter]
    public SwitchParameter RequiresGradient { get; set; }

    protected override void ProcessRecord()
    {
        WriteObject(Variable.Leaf(Tensor, RequiresGradient.IsPresent));
    }
}

[Cmdlet(VerbsLifecycle.Invoke, "PowerTorchVariableOperation")]
[OutputType(typeof(Variable))]
public sealed class InvokePowerTorchVariableOperationCommand : PSCmdlet
{
    [Parameter(Mandatory = true, Position = 0)]
    [ValidateSet("Add", "Multiply", "Subtract", "Divide", "Sigmoid", "Sum", "Mean")]
    public string Operation { get; set; } = string.Empty;

    [Parameter(Mandatory = true)]
    [ValidateNotNull]
    public Variable Left { get; set; } = null!;

    [Parameter]
    public Variable? Right { get; set; }

    protected override void ProcessRecord()
    {
        WriteObject(Operation switch
        {
            "Add" => Left.Add(RequireRight()),
            "Multiply" => Left.Multiply(RequireRight()),
            "Subtract" => Left.Subtract(RequireRight()),
            "Divide" => Left.Divide(RequireRight()),
            "Sigmoid" => Left.Sigmoid(),
            "Sum" => Left.Sum(),
            "Mean" => Left.Mean(),
            _ => throw new InvalidOperationException($"Unknown operation '{Operation}'."),
        });
    }

    private Variable RequireRight()
    {
        if (Right is null)
        {
            throw new ArgumentException($"Operation '{Operation}' requires -Right.");
        }
        return Right;
    }
}

[Cmdlet(VerbsLifecycle.Invoke, "PowerTorchBackward")]
public sealed class InvokePowerTorchBackwardCommand : PSCmdlet
{
    [Parameter(Mandatory = true, Position = 0, ValueFromPipeline = true)]
    [ValidateNotNull]
    public Variable Variable { get; set; } = null!;

    protected override void ProcessRecord()
    {
        Variable.Backward();
        WriteObject(Variable);
    }
}

[Cmdlet(VerbsCommon.New, "PowerTorchAdamW")]
[OutputType(typeof(AdamW))]
public sealed class NewPowerTorchAdamWCommand : PSCmdlet
{
    [Parameter]
    [ValidateRange(0.000000001, 1.0)]
    public double LearningRate { get; set; } = 1e-4;

    [Parameter]
    [ValidateRange(0.0, 1.0)]
    public double WeightDecay { get; set; } = 0.01;

    protected override void ProcessRecord()
    {
        WriteObject(new AdamW(
            learningRate: LearningRate,
            weightDecay: WeightDecay));
    }
}

[Cmdlet(VerbsLifecycle.Invoke, "PowerTorchOptimizerStep")]
public sealed class InvokePowerTorchOptimizerStepCommand : PSCmdlet
{
    [Parameter(Mandatory = true, Position = 0)]
    [ValidateNotNull]
    public AdamW Optimizer { get; set; } = null!;

    [Parameter(Mandatory = true, Position = 1, ValueFromPipeline = true)]
    [ValidateNotNull]
    public Variable[] Parameter { get; set; } = [];

    protected override void ProcessRecord()
    {
        WriteObject(Optimizer.Step(Parameter));
    }
}

[Cmdlet(VerbsCommon.Clear, "PowerTorchGradient")]
public sealed class ClearPowerTorchGradientCommand : PSCmdlet
{
    [Parameter(Mandatory = true, Position = 0, ValueFromPipeline = true)]
    [ValidateNotNull]
    public Variable[] Parameter { get; set; } = [];

    protected override void ProcessRecord()
    {
        AdamW.ZeroGrad(Parameter);
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

public sealed record ShardedTensorMetadata(
    string Path,
    int ShardCount,
    int TensorCount,
    long DataBytes,
    IReadOnlyList<string> DTypes,
    string? MetadataJson);

public sealed record PowerTorchTensorInfo(
    string DataType,
    string Device,
    IReadOnlyList<long> Shape,
    IReadOnlyList<long> Strides,
    long ElementCount,
    bool IsContiguous);
