using PowerTorch.Serialization;

namespace PowerTorch.Tests;

public sealed class OnnxTests
{
    [Fact]
    public void DirectMLOnnxModelLoadsRealOcrDetector()
    {
        using var model = OnnxModel.Open(RealAsset.OcrDetectorOnnx(), useDirectML: true);
        var metadata = model.Metadata;

        Assert.True(metadata.DirectMLRequested);
        Assert.Equal("DmlExecutionProvider", metadata.RequestedProvider);
        Assert.NotEmpty(metadata.Inputs);
        Assert.NotEmpty(metadata.Outputs);
    }
}
