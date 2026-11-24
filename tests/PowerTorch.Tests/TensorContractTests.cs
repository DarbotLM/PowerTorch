using PowerTorch.Core;
using PowerTorch.Cpu;
using PowerTorch.Serialization;
using PowerTorch.Autograd;
using PowerTorch.Optim;

namespace PowerTorch.Tests;

public sealed class TensorContractTests
{
    [Fact]
    public void RealSafetensorsCheckpointHasValidBoundedMetadata()
    {
        string path = RealAsset.FlorenceSafetensors();
        using var checkpoint = new SafetensorsFile(path);

        Assert.True(checkpoint.Tensors.Count > 0);
        Assert.True(checkpoint.HeaderLength > 0);
        Assert.True(checkpoint.HeaderLength <= checkpoint.DataLength);
        Assert.All(
            checkpoint.Tensors.Values,
            tensor =>
            {
                Assert.True(tensor.EndOffset >= tensor.StartOffset);
                Assert.True(tensor.EndOffset <= checkpoint.DataLength);
                Assert.True(tensor.ByteLength > 0);
                Assert.All(tensor.Shape, dimension => Assert.True(dimension >= 0));
            });
    }

    [Fact]
    public async Task CpuAddOperatesOnRealFlorenceTensorSlice()
    {
        string path = RealAsset.FlorenceSafetensors();
        using var checkpoint = new SafetensorsFile(path);
        var source = checkpoint.Tensors.Values
            .First(tensor => tensor.DataType == DataType.Float32 && tensor.ByteLength <= 1_000_000);
        byte[] bytes = await checkpoint.ReadTensorBytesAsync(source.Name);
        using var left = TensorFactory.FromRawBytes(source.DataType, source.Shape, bytes);
        using var right = TensorFactory.FromRawBytes(source.DataType, source.Shape, bytes);

        var runtime = new PowerTorchRuntime();
        runtime.Register(new CpuBackend());
        using var result = runtime.Add(left, right);

        var expected = left.AsSpan<float>();
        var actual = result.AsSpan<float>();
        Assert.Equal(expected.Length, actual.Length);
        for (int index = 0; index < actual.Length; index++)
        {
            float doubled = expected[index] * 2;
            Assert.Equal(doubled, actual[index], precision: 6);
        }
    }

    [Fact]
    public void DescriptorRejectsNegativeStrides()
    {
        Assert.Throws<ArgumentException>(
            () => new TensorDescriptor(
                DataType.Float32,
                [32, 16],
                [16, -1]));
    }

    [Fact]
    public async Task AutogradAndAdamWUpdateARealFlorenceTensor()
    {
        string path = RealAsset.FlorenceSafetensors();
        using var checkpoint = new SafetensorsFile(path);
        var source = checkpoint.Tensors.Values
            .First(tensor => tensor.DataType == DataType.Float32 && tensor.ByteLength <= 100_000);
        byte[] bytes = await checkpoint.ReadTensorBytesAsync(source.Name);
        using var input = TensorFactory.FromRawBytes(source.DataType, source.Shape, bytes);
        float[] before = input.AsSpan<float>().ToArray();
        var parameter = Variable.Leaf(input, requiresGradient: true);
        var sigmoid = parameter.Sigmoid();
        var loss = sigmoid.Sum();

        loss.Backward();

        Assert.NotNull(parameter.Gradient);
        Assert.Equal(parameter.Data.ElementCount, parameter.Gradient.ElementCount);
        using var optimizer = new AdamW(learningRate: 1e-4, weightDecay: 0.0);
        optimizer.Step([parameter]);

        bool changed = false;
        var after = parameter.Data.AsSpan<float>();
        for (int index = 0; index < after.Length; index++)
        {
            changed |= Math.Abs(after[index] - before[index]) > 1e-7;
        }
        Assert.True(changed);
        AdamW.ZeroGrad([parameter]);
        Assert.Null(parameter.Gradient);

        loss.Dispose();
        sigmoid.Dispose();
        parameter.Dispose();
    }
}

internal static class RealAsset
{
    public static string FlorenceSafetensors()
    {
        string? directory = AppContext.BaseDirectory;
        while (directory is not null)
        {
            string candidate = Path.Combine(
                directory,
                "OmniParser",
                "weights",
                "florence31_icon_caption",
                "model.safetensors");
            if (File.Exists(candidate))
            {
                return candidate;
            }

            directory = Directory.GetParent(directory)?.FullName;
        }

        throw new FileNotFoundException(
            "The approved Florence safetensors corpus was not found beneath the toolbox tree.");
    }

    public static string OcrDetectorOnnx()
    {
        string? directory = AppContext.BaseDirectory;
        while (directory is not null)
        {
            string candidate = Path.Combine(
                directory,
                "darbot-env",
                "Lib",
                "site-packages",
                "rapidocr",
                "models",
                "PP-OCRv6_det_small.onnx");
            if (File.Exists(candidate))
            {
                return candidate;
            }
            directory = Directory.GetParent(directory)?.FullName;
        }

        throw new FileNotFoundException(
            "The approved RapidOCR ONNX detector was not found in the Darbot environment.");
    }
}
