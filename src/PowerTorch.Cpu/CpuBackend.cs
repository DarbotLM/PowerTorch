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
            ["tensor.subtract"] = true,
            ["tensor.divide"] = true,
            ["tensor.sigmoid"] = true,
            ["tensor.sum"] = true,
            ["tensor.mean"] = true,
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

    public Tensor Subtract(Tensor left, Tensor right)
    {
        var output = TensorFactory.Empty(left.DataType, left.Shape.ToArray());
        switch (left.DataType)
        {
            case DataType.Int32:
                SubtractSpans(left.AsSpan<int>(), right.AsSpan<int>(), output.AsSpan<int>());
                break;
            case DataType.Int64:
                SubtractSpans(left.AsSpan<long>(), right.AsSpan<long>(), output.AsSpan<long>());
                break;
            case DataType.Float32:
                SubtractSpans(left.AsSpan<float>(), right.AsSpan<float>(), output.AsSpan<float>());
                break;
            case DataType.Float64:
                SubtractSpans(left.AsSpan<double>(), right.AsSpan<double>(), output.AsSpan<double>());
                break;
            default:
                throw Unsupported(left.DataType);
        }
        return output;
    }

    public Tensor Divide(Tensor left, Tensor right)
    {
        var output = TensorFactory.Empty(left.DataType, left.Shape.ToArray());
        switch (left.DataType)
        {
            case DataType.Float32:
                DivideSpans(left.AsSpan<float>(), right.AsSpan<float>(), output.AsSpan<float>());
                break;
            case DataType.Float64:
                DivideSpans(left.AsSpan<double>(), right.AsSpan<double>(), output.AsSpan<double>());
                break;
            default:
                throw new PowerTorchException(
                    PowerTorchErrorCategory.Unsupported,
                    $"CPU divide requires a floating dtype; received '{left.DataType}'.");
        }
        return output;
    }

    public Tensor Sigmoid(Tensor input)
    {
        var output = TensorFactory.Empty(input.DataType, input.Shape.ToArray());
        switch (input.DataType)
        {
            case DataType.Float32:
                var source32 = input.AsSpan<float>();
                var destination32 = output.AsSpan<float>();
                for (int index = 0; index < source32.Length; index++)
                {
                    destination32[index] = 1.0f / (1.0f + MathF.Exp(-source32[index]));
                }
                break;
            case DataType.Float64:
                var source64 = input.AsSpan<double>();
                var destination64 = output.AsSpan<double>();
                for (int index = 0; index < source64.Length; index++)
                {
                    destination64[index] = 1.0 / (1.0 + Math.Exp(-source64[index]));
                }
                break;
            default:
                throw new PowerTorchException(
                    PowerTorchErrorCategory.Unsupported,
                    $"CPU sigmoid requires a floating dtype; received '{input.DataType}'.");
        }
        return output;
    }

    public Tensor Sum(Tensor input) => Reduce(input, average: false);

    public Tensor Mean(Tensor input) => Reduce(input, average: true);

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

    private static void SubtractSpans<T>(ReadOnlySpan<T> left, ReadOnlySpan<T> right, Span<T> output)
        where T : struct, System.Numerics.ISubtractionOperators<T, T, T>
    {
        for (int index = 0; index < left.Length; index++)
        {
            output[index] = left[index] - right[index];
        }
    }

    private static void DivideSpans<T>(ReadOnlySpan<T> left, ReadOnlySpan<T> right, Span<T> output)
        where T : struct, System.Numerics.IDivisionOperators<T, T, T>
    {
        for (int index = 0; index < left.Length; index++)
        {
            output[index] = left[index] / right[index];
        }
    }

    private static Tensor Reduce(Tensor input, bool average)
    {
        Tensor output = TensorFactory.Empty(input.DataType);
        switch (input.DataType)
        {
            case DataType.Float32:
                output.AsSpan<float>()[0] = Reduce<float>(input.AsSpan<float>(), average);
                break;
            case DataType.Float64:
                output.AsSpan<double>()[0] = Reduce<double>(input.AsSpan<double>(), average);
                break;
            case DataType.Int32:
                if (average)
                {
                    throw new PowerTorchException(
                        PowerTorchErrorCategory.Unsupported,
                        "Integer mean is unsupported.");
                }
                int intSum = 0;
                foreach (int value in input.AsSpan<int>())
                {
                    intSum = checked(intSum + value);
                }
                output.AsSpan<int>()[0] = intSum;
                break;
            case DataType.Int64:
                if (average)
                {
                    throw new PowerTorchException(
                        PowerTorchErrorCategory.Unsupported,
                        "Integer mean is unsupported.");
                }
                long longSum = 0;
                foreach (long value in input.AsSpan<long>())
                {
                    longSum = checked(longSum + value);
                }
                output.AsSpan<long>()[0] = longSum;
                break;
            default:
                throw Unsupported(input.DataType);
        }
        return output;
    }

    private static T Reduce<T>(ReadOnlySpan<T> input, bool average)
        where T : struct, System.Numerics.INumber<T>
    {
        T sum = T.Zero;
        for (int index = 0; index < input.Length; index++)
        {
            sum += input[index];
        }
        if (average)
        {
            sum /= T.CreateChecked(input.Length);
        }
        return sum;
    }
}
