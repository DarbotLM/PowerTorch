using PowerTorch.DirectML;

namespace PowerTorch.Tests;

public sealed class DirectMLTests
{
    [Fact]
    public void DirectMLNativeAbiAndDeviceHealthAreValid()
    {
        if (!OperatingSystem.IsWindows())
        {
            return;
        }

        var abi = DirectMLBackend.GetAbi();
        Assert.Equal((ushort)1, abi.Major);

        var adapters = DirectMLBackend.GetAdapterNames();
        Assert.NotEmpty(adapters);
        using var backend = DirectMLBackend.Create();
        backend.SelfTest();
        Assert.True(backend.GetCapabilities().IsAvailable);
    }
}
