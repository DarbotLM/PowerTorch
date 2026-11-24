namespace PowerTorch.Core;

public interface ITensorBackend
{
    Device Device { get; }

    BackendCapability GetCapabilities();

    Tensor Add(Tensor left, Tensor right);

    Tensor Multiply(Tensor left, Tensor right);
}
