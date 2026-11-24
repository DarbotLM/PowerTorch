namespace PowerTorch;

public enum PowerTorchErrorCategory
{
    Unknown,
    InvalidArgument,
    Unsupported,
    Allocation,
    ShapeMismatch,
    DTypeMismatch,
    Backend,
    Compilation,
    Synchronization,
    Serialization,
    Security,
}

public sealed class PowerTorchException : Exception
{
    public PowerTorchException(
        PowerTorchErrorCategory category,
        string message,
        Exception? innerException = null)
        : base(message, innerException)
    {
        Category = category;
    }

    public PowerTorchErrorCategory Category { get; }
}
