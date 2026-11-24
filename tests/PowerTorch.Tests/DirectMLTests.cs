using PowerTorch.DirectML;
using PowerTorch.Core;
using PowerTorch.Serialization;

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

    [Fact]
    public async Task DirectMLAddMatchesCpuOnARealFlorenceTensor()
    {
        if (!OperatingSystem.IsWindows())
        {
            return;
        }

        using var checkpoint = new SafetensorsFile(RealAsset.FlorenceSafetensors());
        var source = checkpoint.Tensors.Values
            .First(tensor => tensor.DataType == DataType.Float32 && tensor.ByteLength <= 100_000);
        byte[] bytes = await checkpoint.ReadTensorBytesAsync(source.Name);
        using var tensor = TensorFactory.FromRawBytes(source.DataType, source.Shape, bytes);
        var values = tensor.AsSpan<float>();
        using var backend = DirectMLBackend.Create();
        float[] actual = backend.AddF32(values, values);
        Assert.Equal(values.Length, actual.Length);
        for (int index = 0; index < values.Length; index++)
        {
            float expected = values[index] * 2.0f;
            Assert.Equal(expected, actual[index], precision: 5);
        }
    }
}
