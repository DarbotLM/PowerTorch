namespace PowerTorch;

public enum DeviceKind
{
    Cpu,
    DirectML,
    Cuda,
}

public readonly record struct Device(DeviceKind Kind, int Index = 0)
{
    public static Device Cpu => new(DeviceKind.Cpu);

    public static Device DirectML(int index = 0) => new(DeviceKind.DirectML, index);

    public static Device Cuda(int index = 0) => new(DeviceKind.Cuda, index);

    public override string ToString() => Kind switch
    {
        DeviceKind.Cpu => "cpu",
        _ => $"{Kind.ToString().ToLowerInvariant()}:{Index}",
    };
}

public sealed record BackendCapability(
    Device Device,
    string Backend,
    string Driver,
    bool IsAvailable,
    IReadOnlyList<string> DTypes,
    IReadOnlyDictionary<string, bool> Features);
