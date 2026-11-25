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

    public Tensor Embedding(Tensor weight, Tensor indices)
    {
        ResolveEmbedding(weight, indices);
        return Resolve(weight).Embedding(weight, indices);
    }

    public Tensor Conv1D(
        Tensor input,
        Tensor weight,
        Tensor? bias = null,
        int stride = 1,
        int padding = 0,
        int dilation = 1,
        int groups = 1,
        bool causal = false)
    {
        ResolveCompatible(input, weight);
        if (bias is not null)
        {
            ResolveOptionalParameters(input, bias);
        }
        return Resolve(input).Conv1D(
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
            causal);
    }

    public Tensor Rotary(
        Tensor input,
        Tensor positions,
        int sequenceDimension = -2,
        int rotaryDimension = 0,
        double baseFrequency = 10_000,
        bool interleaved = false)
    {
        if (input.Device != positions.Device)
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.Backend,
                "Rotary input and position tensors must use the same device.");
        }
        if (positions.DataType is not (DataType.Int32 or DataType.Int64))
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.DTypeMismatch,
                "Rotary positions must use Int32 or Int64.");
        }
        return Resolve(input).Rotary(
            input,
            positions,
            sequenceDimension,
            rotaryDimension,
            baseFrequency,
            interleaved);
    }

    public Tensor CausalAttention(
        Tensor query,
        Tensor key,
        Tensor value,
        double? scale = null)
    {
        ResolveCompatible(query, key);
        ResolveCompatible(query, value);
        return Resolve(query).CausalAttention(query, key, value, scale);
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

    private static void ResolveEmbedding(Tensor weight, Tensor indices)
    {
        if (weight.Device != indices.Device)
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.Backend,
                "Embedding weights and indices must use the same device.");
        }
        if (indices.DataType is not (DataType.Int32 or DataType.Int64))
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.DTypeMismatch,
                "Embedding indices must use Int32 or Int64.");
        }
    }

    private ITensorBackend Resolve(Tensor input) => GetBackend(input.Device.Kind);
}
