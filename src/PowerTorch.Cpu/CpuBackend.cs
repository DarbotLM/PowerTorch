using PowerTorch.Core;

namespace PowerTorch.Cpu;

public sealed class CpuBackend : ITensorBackend
{
    public Device Device => Device.Cpu;

    public BackendCapability GetCapabilities() => new(
        Device,
        "PowerTorch.Cpu",
        Environment.OSVersion.VersionString,
        true,
        [
            nameof(DataType.Int32),
            nameof(DataType.Int64),
            nameof(DataType.Float32),
            nameof(DataType.Float64),
        ],
        new Dictionary<string, bool>
        {
            ["tensor.add"] = true,
            ["tensor.multiply"] = true,
            ["autograd"] = false,
            ["graph_capture"] = false,
        });

    public Tensor Add(Tensor left, Tensor right)
    {
        var output = TensorFactory.Empty(left.DataType, left.Shape.ToArray());
        switch (left.DataType)
        {
            case DataType.Int32:
                AddSpans(left.AsSpan<int>(), right.AsSpan<int>(), output.AsSpan<int>());
                break;
            case DataType.Int64:
                AddSpans(left.AsSpan<long>(), right.AsSpan<long>(), output.AsSpan<long>());
                break;
            case DataType.Float32:
                AddSpans(left.AsSpan<float>(), right.AsSpan<float>(), output.AsSpan<float>());
                break;
            case DataType.Float64:
                AddSpans(left.AsSpan<double>(), right.AsSpan<double>(), output.AsSpan<double>());
                break;
            default:
                throw Unsupported(left.DataType);
        }
        return output;
    }

    public Tensor Multiply(Tensor left, Tensor right)
    {
        var output = TensorFactory.Empty(left.DataType, left.Shape.ToArray());
        switch (left.DataType)
        {
            case DataType.Int32:
                MultiplySpans(left.AsSpan<int>(), right.AsSpan<int>(), output.AsSpan<int>());
                break;
            case DataType.Int64:
                MultiplySpans(left.AsSpan<long>(), right.AsSpan<long>(), output.AsSpan<long>());
                break;
            case DataType.Float32:
                MultiplySpans(left.AsSpan<float>(), right.AsSpan<float>(), output.AsSpan<float>());
                break;
            case DataType.Float64:
                MultiplySpans(left.AsSpan<double>(), right.AsSpan<double>(), output.AsSpan<double>());
                break;
            default:
                throw Unsupported(left.DataType);
        }
        return output;
    }

    private static PowerTorchException Unsupported(DataType dataType) => new(
        PowerTorchErrorCategory.Unsupported,
        $"CPU v0 backend does not implement dtype '{dataType}'.");

    private static void AddSpans<T>(ReadOnlySpan<T> left, ReadOnlySpan<T> right, Span<T> output)
        where T : struct, System.Numerics.IAdditionOperators<T, T, T>
    {
        for (int index = 0; index < left.Length; index++)
        {
            output[index] = left[index] + right[index];
        }
    }

    private static void MultiplySpans<T>(ReadOnlySpan<T> left, ReadOnlySpan<T> right, Span<T> output)
        where T : struct, System.Numerics.IMultiplyOperators<T, T, T>
    {
        for (int index = 0; index < left.Length; index++)
        {
            output[index] = left[index] * right[index];
        }
    }
}
