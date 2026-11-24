using System.Collections.ObjectModel;

namespace PowerTorch;

public sealed class TensorDescriptor
{
    private readonly long[] _shape;
    private readonly long[] _strides;

    public TensorDescriptor(DataType dataType, IEnumerable<long> shape, IEnumerable<long>? strides = null)
    {
        _shape = shape?.ToArray() ?? throw new ArgumentNullException(nameof(shape));
        if (_shape.Any(dimension => dimension < 0))
        {
            throw new ArgumentOutOfRangeException(nameof(shape), "Tensor dimensions cannot be negative.");
        }

        DataType = dataType;
        _strides = strides is null ? CreateContiguousStrides(_shape) : strides.ToArray();
        if (_strides.Length != _shape.Length)
        {
            throw new ArgumentException("Stride rank must match shape rank.", nameof(strides));
        }
        if (_strides.Any(stride => stride < 0))
        {
            throw new ArgumentException("Negative strides are rejected in ABI v1.", nameof(strides));
        }
        Shape = new ReadOnlyCollection<long>(_shape);
    }

    public DataType DataType { get; }

    public IReadOnlyList<long> Shape { get; }

    public IReadOnlyList<long> Strides => _strides;

    public long ElementCount
    {
        get
        {
            if (_shape.Length == 0)
            {
                return 1;
            }

            long count = 1;
            foreach (long dimension in _shape)
            {
                count = checked(count * dimension);
            }
            return count;
        }
    }

    public long ByteLength => checked(ElementCount * DataTypeInfo.ElementSize(DataType));

    public bool IsScalar => _shape.Length == 0;

    public bool IsContiguous
    {
        get
        {
            var expected = CreateContiguousStrides(_shape);
            return expected.AsSpan().SequenceEqual(_strides);
        }
    }

    public TensorDescriptor WithShape(IEnumerable<long> shape)
    {
        var result = new TensorDescriptor(DataType, shape);
        if (ElementCount != result.ElementCount)
        {
            throw new ArgumentException("Reshape must preserve element count.", nameof(shape));
        }
        return result;
    }

    private static long[] CreateContiguousStrides(long[] shape)
    {
        var strides = new long[shape.Length];
        long stride = 1;
        for (int index = shape.Length - 1; index >= 0; index--)
        {
            strides[index] = stride;
            stride = checked(stride * shape[index]);
        }
        return strides;
    }
}
