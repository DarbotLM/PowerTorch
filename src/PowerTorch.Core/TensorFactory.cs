using System.Runtime.InteropServices;

namespace PowerTorch.Core;

public static class TensorFactory
{
    public static Tensor FromArray(float[] values, IEnumerable<long>? shape = null)
    {
        var resolvedShape = shape?.ToArray() ?? [values.LongLength];
        var descriptor = TensorDescriptors.Contiguous(DataType.Float32, resolvedShape);
        ValidateCount(descriptor, values.LongLength);
        return FromArray(values, descriptor);
    }

    public static Tensor FromArray(int[] values, IEnumerable<long>? shape = null)
    {
        var resolvedShape = shape?.ToArray() ?? [values.LongLength];
        var descriptor = TensorDescriptors.Contiguous(DataType.Int32, resolvedShape);
        ValidateCount(descriptor, values.LongLength);
        return FromArray(values, descriptor);
    }

    public static Tensor FromArray(double[] values, IEnumerable<long>? shape = null)
    {
        var resolvedShape = shape?.ToArray() ?? [values.LongLength];
        var descriptor = TensorDescriptors.Contiguous(DataType.Float64, resolvedShape);
        ValidateCount(descriptor, values.LongLength);
        return FromArray(values, descriptor);
    }

    public static Tensor FromArray(long[] values, IEnumerable<long>? shape = null)
    {
        var resolvedShape = shape?.ToArray() ?? [values.LongLength];
        var descriptor = TensorDescriptors.Contiguous(DataType.Int64, resolvedShape);
        ValidateCount(descriptor, values.LongLength);
        return FromArray(values, descriptor);
    }

    public static Tensor Empty(DataType dataType, params long[] shape) =>
        new(TensorDescriptors.Contiguous(dataType, shape), Device.Cpu);

    public static Tensor FromRawBytes(
        DataType dataType,
        IEnumerable<long> shape,
        ReadOnlySpan<byte> bytes)
    {
        var descriptor = TensorDescriptors.Contiguous(dataType, shape.ToArray());
        if (bytes.Length != descriptor.ByteLength)
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.ShapeMismatch,
                $"Raw data has {bytes.Length} bytes; shape/dtype require {descriptor.ByteLength}.");
        }
        return new Tensor(descriptor, Device.Cpu, bytes.ToArray());
    }

    private static Tensor FromArray<T>(T[] values, TensorDescriptor descriptor)
        where T : unmanaged
    {
        var bytes = MemoryMarshal.AsBytes(values.AsSpan()).ToArray();
        return new Tensor(descriptor, Device.Cpu, bytes);
    }

    private static void ValidateCount(TensorDescriptor descriptor, long count)
    {
        if (descriptor.ElementCount != count)
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.ShapeMismatch,
                $"Shape describes {descriptor.ElementCount} elements but the source has {count}.");
        }
    }
}
