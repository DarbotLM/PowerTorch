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
}
