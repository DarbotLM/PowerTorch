namespace PowerTorch.Core;

public sealed class PowerTorchRuntime
{
    private readonly Dictionary<DeviceKind, ITensorBackend> _backends = new();

    public static PowerTorchRuntime Shared { get; } = new();

    public IReadOnlyList<BackendCapability> Capabilities =>
        _backends.Values.Select(backend => backend.GetCapabilities()).ToArray();

    public void Register(ITensorBackend backend)
    {
        ArgumentNullException.ThrowIfNull(backend);
        _backends[backend.Device.Kind] = backend;
    }

    public ITensorBackend GetBackend(DeviceKind kind)
    {
        if (!_backends.TryGetValue(kind, out var backend))
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.Backend,
                $"Backend '{kind}' is not registered.");
        }
        return backend;
    }

    public Tensor Add(Tensor left, Tensor right) => Resolve(left, right).Add(left, right);

    public Tensor Multiply(Tensor left, Tensor right) => Resolve(left, right).Multiply(left, right);

    public Tensor Subtract(Tensor left, Tensor right) => Resolve(left, right).Subtract(left, right);

    public Tensor Divide(Tensor left, Tensor right) => Resolve(left, right).Divide(left, right);

    public Tensor Sigmoid(Tensor input) => Resolve(input).Sigmoid(input);

    public Tensor Sum(Tensor input) => Resolve(input).Sum(input);

    public Tensor Mean(Tensor input) => Resolve(input).Mean(input);

    public Tensor Reshape(Tensor input, IReadOnlyList<long> shape) =>
        Resolve(input).Reshape(input, shape);

    public Tensor Permute(Tensor input, IReadOnlyList<int> dimensions) =>
        Resolve(input).Permute(input, dimensions);

    public Tensor Transpose(Tensor input, int firstDimension, int secondDimension) =>
        Resolve(input).Transpose(input, firstDimension, secondDimension);

    public Tensor Slice(
        Tensor input,
        int dimension,
        long start,
        long end,
        long step = 1) =>
        Resolve(input).Slice(input, dimension, start, end, step);

    public Tensor Matmul(Tensor left, Tensor right) =>
        ResolveCompatible(left, right).Matmul(left, right);

    public Tensor Softmax(Tensor input, int dimension = -1) =>
        Resolve(input).Softmax(input, dimension);

    public Tensor LayerNorm(
        Tensor input,
        IReadOnlyList<long> normalizedShape,
        Tensor? weight = null,
        Tensor? bias = null,
        double epsilon = 1e-5)
    {
        ResolveOptionalParameters(input, weight, bias);
        return Resolve(input).LayerNorm(input, normalizedShape, weight, bias, epsilon);
    }

    public Tensor RmsNorm(
        Tensor input,
        IReadOnlyList<long> normalizedShape,
        Tensor? weight = null,
        double epsilon = 1e-5)
    {
        ResolveOptionalParameters(input, weight);
        return Resolve(input).RmsNorm(input, normalizedShape, weight, epsilon);
    }

    private ITensorBackend Resolve(Tensor left, Tensor right)
    {
        if (left.Device != right.Device)
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.Backend,
                $"Tensor devices must match; left={left.Device}, right={right.Device}.");
        }
        if (left.DataType != right.DataType)
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.DTypeMismatch,
                $"Tensor dtypes must match; left={left.DataType}, right={right.DataType}.");
        }
        if (!left.Shape.SequenceEqual(right.Shape))
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.ShapeMismatch,
                "Broadcasting is not implemented in the current runtime slice.");
        }
        return GetBackend(left.Device.Kind);
    }

    private ITensorBackend ResolveCompatible(Tensor left, Tensor right)
    {
        if (left.Device != right.Device)
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.Backend,
                $"Tensor devices must match; left={left.Device}, right={right.Device}.");
        }
        if (left.DataType != right.DataType)
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.DTypeMismatch,
                $"Tensor dtypes must match; left={left.DataType}, right={right.DataType}.");
        }
        return GetBackend(left.Device.Kind);
    }

    private static void ResolveOptionalParameters(Tensor input, params Tensor?[] parameters)
    {
        foreach (var parameter in parameters)
        {
            if (parameter is null)
            {
                continue;
            }
            if (parameter.Device != input.Device || parameter.DataType != input.DataType)
            {
                throw new PowerTorchException(
                    PowerTorchErrorCategory.DTypeMismatch,
                    "Normalization parameters must match the input device and dtype.");
            }
        }
    }

    private ITensorBackend Resolve(Tensor input) => GetBackend(input.Device.Kind);
}
