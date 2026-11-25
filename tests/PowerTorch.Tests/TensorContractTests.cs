using System.Buffers.Binary;
using System.Text;
using PowerTorch.Core;
using PowerTorch.Cpu;
using PowerTorch.Serialization;
using PowerTorch.Autograd;
using PowerTorch.Optim;

namespace PowerTorch.Tests;

public sealed class TensorContractTests
{
    [Fact]
    public void CpuReportsVibeVoiceTensorCapabilities()
    {
        var capabilities = new CpuBackend().GetCapabilities();

        Assert.Contains(nameof(DataType.BFloat16), capabilities.DTypes);
        Assert.True(capabilities.Features["tensor.reshape"]);
        Assert.True(capabilities.Features["tensor.permute"]);
        Assert.True(capabilities.Features["tensor.slice"]);
        Assert.True(capabilities.Features["tensor.matmul"]);
        Assert.True(capabilities.Features["tensor.softmax"]);
        Assert.True(capabilities.Features["tensor.layer_norm"]);
        Assert.True(capabilities.Features["tensor.rms_norm"]);
    }

    [Fact]
    public void BFloat16CheckpointBytesExposeTypedTensorValues()
    {
        byte[] bytes = [0x80, 0x3F, 0x00, 0x40, 0xC0, 0xBF];
        using var tensor = TensorFactory.FromRawBytes(DataType.BFloat16, [3], bytes);

        Assert.Equal(1.0f, (float)tensor.AsSpan<BFloat16>()[0], precision: 3);
        Assert.Equal(2.0f, (float)tensor.AsSpan<BFloat16>()[1], precision: 3);
        Assert.Equal(-1.5f, (float)tensor.AsSpan<BFloat16>()[2], precision: 3);
        Assert.Equal(2, DataTypeInfo.ElementSize(DataType.BFloat16));
    }

    [Fact]
    public async Task SafetensorsBFloat16CheckpointTensorCanBeRead()
    {
        string path = Path.Combine(
            AppContext.BaseDirectory,
            $"powertorch-bf16-{Guid.NewGuid():N}.safetensors");
        const string Header =
            """{"weight":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}""";
        try
        {
            byte[] headerBytes = Encoding.UTF8.GetBytes(Header);
            byte[] lengthBytes = new byte[sizeof(ulong)];
            BinaryPrimitives.WriteUInt64LittleEndian(lengthBytes, (ulong)headerBytes.Length);
            await File.WriteAllBytesAsync(
                path,
                [.. lengthBytes, .. headerBytes, 0x80, 0x3F, 0x00, 0x40]);

            using var checkpoint = new SafetensorsFile(path);
            byte[] bytes = await checkpoint.ReadTensorBytesAsync("weight");
            using var tensor = TensorFactory.FromRawBytes(
                checkpoint.GetTensor("weight").DataType,
                checkpoint.GetTensor("weight").Shape,
                bytes);

            Assert.Equal(DataType.BFloat16, checkpoint.GetTensor("weight").DataType);
            Assert.Equal(
                [1.0f, 2.0f],
                tensor.AsSpan<BFloat16>().ToArray().Select(value => (float)value).ToArray());
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void CpuShapeOperationsProduceContiguousCorrectValues()
    {
        using var input = TensorFactory.FromArray(
            Enumerable.Range(0, 24).Select(value => (float)value).ToArray(),
            [2, 3, 4]);
        var runtime = CreateRuntime();

        using var reshaped = runtime.Reshape(input, [4, -1]);
        Assert.Equal(new long[] { 4, 6 }, reshaped.Shape);

        using var transposed = runtime.Transpose(input, 0, 2);
        Assert.Equal(new long[] { 4, 3, 2 }, transposed.Shape);
        Assert.Equal(0, transposed.AsSpan<float>()[0], precision: 0);
        Assert.Equal(12, transposed.AsSpan<float>()[1], precision: 0);
        Assert.Equal(4, transposed.AsSpan<float>()[2], precision: 0);

        using var permuted = runtime.Permute(input, [1, 0, 2]);
        Assert.Equal(new long[] { 3, 2, 4 }, permuted.Shape);
        Assert.Equal(12, permuted.AsSpan<float>()[4], precision: 0);

        using var sliced = runtime.Slice(reshaped, 1, 1, 6, 2);
        Assert.Equal(new long[] { 4, 3 }, sliced.Shape);
        Assert.Equal(
            [1.0f, 3.0f, 5.0f, 7.0f, 9.0f, 11.0f],
            sliced.AsSpan<float>().ToArray()[..6]);
    }

    [Fact]
    public void CpuMatmulSupportsBFloat16AndBatchedFloatMatrices()
    {
        var runtime = CreateRuntime();
        long[] matrixShape = [2, 2];
        BFloat16[] leftValues =
            [(BFloat16)1.0f, (BFloat16)2.0f, (BFloat16)3.0f, (BFloat16)4.0f];
        BFloat16[] rightValues =
            [(BFloat16)5.0f, (BFloat16)6.0f, (BFloat16)7.0f, (BFloat16)8.0f];
        using var leftBf16 = TensorFactory.FromArray(leftValues, matrixShape);
        using var rightBf16 = TensorFactory.FromArray(rightValues, matrixShape);

        using var bf16Result = runtime.Matmul(leftBf16, rightBf16);
        Assert.Equal(
            [19.0f, 22.0f, 43.0f, 50.0f],
            bf16Result.AsSpan<BFloat16>().ToArray().Select(value => (float)value).ToArray());

        using var batched = TensorFactory.FromArray(
            Enumerable.Range(1, 8).Select(value => (float)value).ToArray(),
            [2, 2, 2]);
        using var identity = TensorFactory.FromArray([1.0f, 0.0f, 0.0f, 1.0f], [2, 2]);
        using var result = runtime.Matmul(batched, identity);
        Assert.Equal(batched.Shape, result.Shape);
        Assert.Equal(batched.AsSpan<float>().ToArray(), result.AsSpan<float>().ToArray());
    }

    [Fact]
    public void CpuSoftmaxAndNormalizationAreNumericallyCorrect()
    {
        var runtime = CreateRuntime();
        using var input = TensorFactory.FromArray([1.0f, 2.0f, 3.0f, 4.0f], [2, 2]);
        using var softmax = runtime.Softmax(input, -1);
        Assert.Equal(0.26894143f, softmax.AsSpan<float>()[0], precision: 6);
        Assert.Equal(0.7310586f, softmax.AsSpan<float>()[1], precision: 6);

        using var layerNorm = runtime.LayerNorm(input, [2], epsilon: 1e-5);
        float layerScale = 0.5f / MathF.Sqrt(0.25f + 1e-5f);
        Assert.Equal(-layerScale, layerNorm.AsSpan<float>()[0], precision: 6);
        Assert.Equal(layerScale, layerNorm.AsSpan<float>()[1], precision: 6);

        using var rmsInput = TensorFactory.FromArray([3.0f, 4.0f], [2]);
        using var rmsNorm = runtime.RmsNorm(rmsInput, [2], epsilon: 1e-5);
        Assert.Equal(3.0f / MathF.Sqrt(12.50001f), rmsNorm.AsSpan<float>()[0], precision: 5);
        Assert.Equal(4.0f / MathF.Sqrt(12.50001f), rmsNorm.AsSpan<float>()[1], precision: 5);
    }

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

    private static PowerTorchRuntime CreateRuntime()
    {
        var runtime = new PowerTorchRuntime();
        runtime.Register(new CpuBackend());
        return runtime;
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
