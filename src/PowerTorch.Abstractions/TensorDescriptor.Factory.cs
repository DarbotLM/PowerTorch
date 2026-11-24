namespace PowerTorch;

public static class TensorDescriptors
{
    public static TensorDescriptor Contiguous(DataType dataType, params long[] shape)
    {
        return new TensorDescriptor(dataType, shape);
    }
}
