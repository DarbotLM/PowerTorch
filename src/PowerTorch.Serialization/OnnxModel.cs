using Microsoft.ML.OnnxRuntime;

namespace PowerTorch.Serialization;

public sealed class OnnxModel : IDisposable
{
    private readonly SessionOptions _options;
    private readonly InferenceSession _session;
    private bool _disposed;

    private OnnxModel(
        string path,
        SessionOptions options,
        InferenceSession session,
        bool directMLRequested,
        string requestedProvider)
    {
        Path = System.IO.Path.GetFullPath(path);
        _options = options;
        _session = session;
        DirectMLRequested = directMLRequested;
        RequestedProvider = requestedProvider;
    }

    public string Path { get; }

    public bool DirectMLRequested { get; }

    public string RequestedProvider { get; }

    public OnnxModelInfo Metadata
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            var model = _session.ModelMetadata;
            return new OnnxModelInfo(
                Path,
                model.ProducerName,
                model.GraphName,
                model.Domain,
                model.Version,
                DirectMLRequested,
                RequestedProvider,
                CreateNodes(_session.InputMetadata),
                CreateNodes(_session.OutputMetadata),
                new Dictionary<string, string>(
                    model.CustomMetadataMap,
                    StringComparer.Ordinal));
        }
    }

    public static OnnxModel Open(
        string path,
        bool useDirectML = true,
        int adapterIndex = 0)
    {
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
        {
            throw new FileNotFoundException("The ONNX model was not found.", path);
        }
        ArgumentOutOfRangeException.ThrowIfNegative(adapterIndex);

        var options = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR,
        };
        string provider;
        bool requestedDirectML = false;
        try
        {
            string[] availableProviders = OrtEnv.Instance().GetAvailableProviders();
            if (useDirectML && availableProviders.Contains("DmlExecutionProvider", StringComparer.Ordinal))
            {
                options.AppendExecutionProvider_DML(adapterIndex);
                requestedDirectML = true;
                provider = "DmlExecutionProvider";
            }
            else
            {
                options.AppendExecutionProvider_CPU();
                provider = "CPUExecutionProvider";
            }

            var session = new InferenceSession(path, options);
            return new OnnxModel(path, options, session, requestedDirectML, provider);
        }
        catch
        {
            options.Dispose();
            throw;
        }
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }
        _disposed = true;
        _session.Dispose();
        _options.Dispose();
    }

    private static OnnxTensorDescriptor[] CreateNodes(
        IReadOnlyDictionary<string, NodeMetadata> nodes) =>
        nodes.Select(pair => new OnnxTensorDescriptor(
            pair.Key,
            pair.Value.IsTensor,
            pair.Value.ElementType.FullName ?? pair.Value.ElementType.Name,
            pair.Value.Dimensions.Select(dimension => (long)dimension).ToArray(),
            pair.Value.SymbolicDimensions))
        .ToArray();
}

public sealed record OnnxTensorDescriptor(
    string Name,
    bool IsTensor,
    string ElementType,
    IReadOnlyList<long> Dimensions,
    IReadOnlyList<string> SymbolicDimensions);

public sealed record OnnxModelInfo(
    string Path,
    string Producer,
    string GraphName,
    string Domain,
    long Version,
    bool DirectMLRequested,
    string RequestedProvider,
    IReadOnlyList<OnnxTensorDescriptor> Inputs,
    IReadOnlyList<OnnxTensorDescriptor> Outputs,
    IReadOnlyDictionary<string, string> CustomMetadata);
