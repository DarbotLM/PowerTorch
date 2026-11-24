namespace PowerTorch.Serialization;

public sealed record SafeTensorDescriptor(
    string Name,
    DataType DataType,
    string SourceDataType,
    IReadOnlyList<long> Shape,
    long StartOffset,
    long EndOffset)
{
    public long ByteLength => checked(EndOffset - StartOffset);

    public long ElementCount
    {
        get
        {
            if (Shape.Count == 0)
            {
                return 1;
            }

            long count = 1;
            foreach (long dimension in Shape)
            {
                count = checked(count * dimension);
            }
            return count;
        }
    }
}
