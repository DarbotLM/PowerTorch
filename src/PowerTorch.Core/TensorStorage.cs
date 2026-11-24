using System.Buffers;

namespace PowerTorch.Core;

internal sealed class TensorStorage : IDisposable
{
    private byte[]? _bytes;

    public TensorStorage(long byteLength)
    {
        if (byteLength < 0 || byteLength > int.MaxValue)
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.Allocation,
                $"Managed CPU storage currently supports 0..{int.MaxValue} bytes; requested {byteLength}.");
        }

        _bytes = ArrayPool<byte>.Shared.Rent(checked((int)byteLength));
        UsableLength = checked((int)byteLength);
    }

    public int UsableLength { get; }

    public Span<byte> Bytes =>
        _bytes is null
            ? throw new ObjectDisposedException(nameof(TensorStorage))
            : _bytes.AsSpan(0, UsableLength);

    public Memory<byte> Memory =>
        _bytes is null
            ? throw new ObjectDisposedException(nameof(TensorStorage))
            : _bytes.AsMemory(0, UsableLength);

    public void Dispose()
    {
        var bytes = Interlocked.Exchange(ref _bytes, null);
        if (bytes is not null)
        {
            ArrayPool<byte>.Shared.Return(bytes);
        }
    }
}
