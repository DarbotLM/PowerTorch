using System.Runtime.InteropServices;

namespace PowerTorch.Core;

public enum TensorResidency
{
    HostManaged,
    Device,
}

public sealed class Tensor : IDisposable
{
    private readonly TensorStorage _storage;
    private int _disposed;

    public Tensor(TensorDescriptor descriptor, Device device)
    {
        Descriptor = descriptor ?? throw new ArgumentNullException(nameof(descriptor));
        Device = device;
        _storage = new TensorStorage(descriptor.ByteLength);
    }

    internal Tensor(TensorDescriptor descriptor, Device device, byte[] bytes)
    {
        Descriptor = descriptor ?? throw new ArgumentNullException(nameof(descriptor));
        if (bytes.Length != descriptor.ByteLength)
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.ShapeMismatch,
                $"Storage byte length {bytes.Length} does not match tensor length {descriptor.ByteLength}.");
        }

        Device = device;
        _storage = new TensorStorage(descriptor.ByteLength);
        bytes.CopyTo(_storage.Bytes);
    }

    public TensorDescriptor Descriptor { get; }

    public DataType DataType => Descriptor.DataType;

    public Device Device { get; }

    public long ElementCount => Descriptor.ElementCount;

    public IReadOnlyList<long> Shape => Descriptor.Shape;

    public IReadOnlyList<long> Strides => Descriptor.Strides;

    public bool IsContiguous => Descriptor.IsContiguous;

    public bool IsHostResident => Device.Kind == DeviceKind.Cpu;

    public TensorResidency Residency =>
        IsHostResident ? TensorResidency.HostManaged : TensorResidency.Device;

    public Span<T> AsSpan<T>() where T : unmanaged
    {
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        if (Device.Kind != DeviceKind.Cpu)
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.Unsupported,
                $"Direct CPU access is unavailable for {Device} tensors.");
        }
        if (typeof(T) != DataTypeInfo.ClrType(DataType))
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.DTypeMismatch,
                $"Type {typeof(T).Name} cannot view a {DataType} tensor.");
        }
        if (!Descriptor.IsContiguous)
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.Unsupported,
                "Span access currently requires contiguous CPU storage.");
        }

        return MemoryMarshal.Cast<byte, T>(_storage.Bytes);
    }

    public Tensor Clone()
    {
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        var clone = new Tensor(Descriptor, Device);
        _storage.Bytes.CopyTo(clone._storage.Bytes);
        return clone;
    }

    public override string ToString() =>
        $"PowerTorch.Tensor(shape=[{string.Join(",", Shape)}], dtype={DataType}, device={Device})";

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _storage.Dispose();
        }
        GC.SuppressFinalize(this);
    }

    internal Span<byte> RawBytes
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed != 0, this);
            return _storage.Bytes;
        }
    }
}
