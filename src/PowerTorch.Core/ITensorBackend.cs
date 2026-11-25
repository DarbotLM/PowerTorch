namespace PowerTorch.Core;

public interface ITensorBackend
{
    Device Device { get; }

    BackendCapability GetCapabilities();

    Tensor Add(Tensor left, Tensor right);

    Tensor Multiply(Tensor left, Tensor right);

    Tensor Subtract(Tensor left, Tensor right);

    Tensor Divide(Tensor left, Tensor right);

    Tensor Sigmoid(Tensor input);

    Tensor Sum(Tensor input);

    Tensor Mean(Tensor input);

    Tensor Reshape(Tensor input, IReadOnlyList<long> shape);

    Tensor Permute(Tensor input, IReadOnlyList<int> dimensions);

    Tensor Transpose(Tensor input, int firstDimension, int secondDimension);

    Tensor Slice(Tensor input, int dimension, long start, long endExclusive, long stride);

    Tensor Matmul(Tensor left, Tensor right);

    Tensor Softmax(Tensor input, int dimension);

    Tensor LayerNorm(
        Tensor input,
        IReadOnlyList<long> normalizedShape,
        Tensor? weight,
        Tensor? bias,
        double epsilon);

    Tensor RmsNorm(
        Tensor input,
        IReadOnlyList<long> normalizedShape,
        Tensor? weight,
        double epsilon);
}
