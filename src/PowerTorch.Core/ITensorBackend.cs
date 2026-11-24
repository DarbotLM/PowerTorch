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
}
