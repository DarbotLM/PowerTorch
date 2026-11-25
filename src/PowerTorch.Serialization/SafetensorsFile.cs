using System.Buffers.Binary;
using System.Text;
using System.Text.Json;
using PowerTorch.Core;

namespace PowerTorch.Serialization;

public sealed class SafetensorsFile : IDisposable
{
    private const long MaximumHeaderLength = 256L * 1024L * 1024L;
    private readonly FileStream _stream;
    private readonly Dictionary<string, SafeTensorDescriptor> _tensors;
    private readonly long _dataStart;

    public SafetensorsFile(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            throw new ArgumentException("A safetensors path is required.", nameof(path));
        }

        Path = System.IO.Path.GetFullPath(path);
        _stream = new FileStream(
            Path,
            FileMode.Open,
            FileAccess.Read,
            FileShare.Read,
            bufferSize: 64 * 1024,
            useAsync: true);

        Span<byte> lengthBytes = stackalloc byte[sizeof(ulong)];
        if (_stream.Read(lengthBytes) != lengthBytes.Length)
        {
            throw SerializationError("The file is too short to contain a safetensors header length.");
        }

        ulong rawHeaderLength = BinaryPrimitives.ReadUInt64LittleEndian(lengthBytes);
        if (rawHeaderLength == 0 || rawHeaderLength > (ulong)MaximumHeaderLength)
        {
            throw SerializationError(
                $"Safetensors header length {rawHeaderLength} is outside the allowed range 1..{MaximumHeaderLength}.");
        }
        if (checked((long)rawHeaderLength) > _stream.Length - sizeof(ulong))
        {
            throw SerializationError("Safetensors header length exceeds the file length.");
        }

        int headerLength = checked((int)rawHeaderLength);
        var headerBytes = new byte[headerLength];
        if (_stream.Read(headerBytes, 0, headerBytes.Length) != headerBytes.Length)
        {
            throw SerializationError("The safetensors header was truncated.");
        }
        _dataStart = sizeof(ulong) + headerLength;
        var availableDataBytes = _stream.Length - _dataStart;

        try
        {
            using var document = JsonDocument.Parse(headerBytes);
            if (document.RootElement.ValueKind != JsonValueKind.Object)
            {
                throw SerializationError("The safetensors header root must be a JSON object.");
            }

            string? metadata = null;
            var tensors = new Dictionary<string, SafeTensorDescriptor>(StringComparer.Ordinal);
            foreach (var property in document.RootElement.EnumerateObject())
            {
                if (property.NameEquals("__metadata__"))
                {
                    metadata = property.Value.GetRawText();
                    continue;
                }

                tensors.Add(
                    property.Name,
                    ParseTensor(property.Name, property.Value, availableDataBytes));
            }

            ValidateNoOverlaps(tensors.Values);
            _tensors = tensors;
            MetadataJson = metadata;
        }
        catch (PowerTorchException)
        {
            throw;
        }
        catch (JsonException exception)
        {
            throw SerializationError("The safetensors header is not valid JSON.", exception);
        }
    }

    public string Path { get; }

    public string? MetadataJson { get; }

    public long HeaderLength => _dataStart - sizeof(ulong);

    public long DataLength => _stream.Length - _dataStart;

    public IReadOnlyDictionary<string, SafeTensorDescriptor> Tensors => _tensors;

    public SafeTensorDescriptor GetTensor(string name)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("A tensor name is required.", nameof(name));
        }
        if (!_tensors.TryGetValue(name, out var descriptor))
        {
            throw new KeyNotFoundException($"Tensor '{name}' was not found in '{Path}'.");
        }
        return descriptor;
    }

    public async ValueTask<byte[]> ReadTensorBytesAsync(
        string name,
        CancellationToken cancellationToken = default)
    {
        var descriptor = GetTensor(name);
        if (descriptor.ByteLength > int.MaxValue)
        {
            throw SerializationError(
                $"Tensor '{name}' is {descriptor.ByteLength} bytes and cannot be copied to one managed array.");
        }

        var bytes = new byte[checked((int)descriptor.ByteLength)];
        int written = 0;
        long fileOffset = checked(_dataStart + descriptor.StartOffset);
        while (written < bytes.Length)
        {
            int count = await RandomAccess.ReadAsync(
                _stream.SafeFileHandle,
                bytes.AsMemory(written),
                fileOffset,
                cancellationToken);
            if (count == 0)
            {
                throw SerializationError(
                    $"Tensor '{name}' ended after {written} of {bytes.Length} bytes.");
            }
            written += count;
            fileOffset = checked(fileOffset + count);
        }
        return bytes;
    }

    public void Dispose()
    {
        _stream.Dispose();
    }

    private static SafeTensorDescriptor ParseTensor(
        string name,
        JsonElement value,
        long availableDataBytes)
    {
        if (value.ValueKind != JsonValueKind.Object)
        {
            throw SerializationError($"Tensor '{name}' must be represented by a JSON object.");
        }
        if (!value.TryGetProperty("dtype", out var dtypeElement) ||
            dtypeElement.ValueKind != JsonValueKind.String)
        {
            throw SerializationError($"Tensor '{name}' is missing a string dtype.");
        }
        if (!value.TryGetProperty("shape", out var shapeElement) ||
            shapeElement.ValueKind != JsonValueKind.Array)
        {
            throw SerializationError($"Tensor '{name}' is missing a shape array.");
        }
        if (!value.TryGetProperty("data_offsets", out var offsetsElement) ||
            offsetsElement.ValueKind != JsonValueKind.Array ||
            offsetsElement.GetArrayLength() != 2)
        {
            throw SerializationError($"Tensor '{name}' must contain two data offsets.");
        }

        string dtypeCode = dtypeElement.GetString()!;
        var shape = shapeElement.EnumerateArray()
            .Select(element =>
            {
                if (!element.TryGetInt64(out long dimension) || dimension < 0)
                {
                    throw SerializationError(
                        $"Tensor '{name}' has a negative or non-integer shape dimension.");
                }
                return dimension;
            })
            .ToArray();
        if (!offsetsElement[0].TryGetInt64(out long start) ||
            !offsetsElement[1].TryGetInt64(out long end) ||
            start < 0 ||
            end < start)
        {
            throw SerializationError($"Tensor '{name}' has invalid data offsets.");
        }
        if (end > availableDataBytes)
        {
            throw SerializationError(
                $"Tensor '{name}' ends at offset {end}, beyond {availableDataBytes} data bytes.");
        }

        DataType dataType;
        try
        {
            dataType = DataTypeInfo.FromSafetensorsCode(dtypeCode);
        }
        catch (NotSupportedException exception)
        {
            throw SerializationError($"Tensor '{name}' uses an unsupported dtype '{dtypeCode}'.", exception);
        }

        var descriptor = new SafeTensorDescriptor(name, dataType, dtypeCode, shape, start, end);
        long expectedBytes = checked(descriptor.ElementCount * DataTypeInfo.ElementSize(dataType));
        if (descriptor.ByteLength != expectedBytes)
        {
            throw SerializationError(
                $"Tensor '{name}' declares {descriptor.ByteLength} bytes but shape/dtype require {expectedBytes}.");
        }
        return descriptor;
    }

    private static void ValidateNoOverlaps(IEnumerable<SafeTensorDescriptor> descriptors)
    {
        SafeTensorDescriptor? previous = null;
        foreach (var descriptor in descriptors.OrderBy(item => item.StartOffset))
        {
            if (previous is not null && descriptor.StartOffset < previous.EndOffset)
            {
                throw SerializationError(
                    $"Tensor data overlaps for '{previous.Name}' and '{descriptor.Name}'.");
            }
            previous = descriptor;
        }
    }

    private static PowerTorchException SerializationError(
        string message,
        Exception? innerException = null) => new(
        PowerTorchErrorCategory.Serialization,
        message,
        innerException);
}

public sealed class ShardedSafetensorsCheckpoint : IDisposable
{
    private const int MaximumIndexBytes = 16 * 1024 * 1024;
    private readonly Dictionary<string, string> _shardsByTensor;
    private readonly Dictionary<string, SafetensorsFile> _shards =
        new(StringComparer.Ordinal);
    private bool _disposed;

    public ShardedSafetensorsCheckpoint(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            throw new ArgumentException("A checkpoint index path is required.", nameof(path));
        }

        IndexPath = Path.GetFullPath(path);
        var indexInfo = new FileInfo(IndexPath);
        if (!indexInfo.Exists ||
            indexInfo.Length > MaximumIndexBytes ||
            (indexInfo.Attributes & FileAttributes.ReparsePoint) != 0)
        {
            throw CheckpointError(
                $"Checkpoint index must exist and be at most {MaximumIndexBytes} bytes.");
        }

        try
        {
            byte[] bytes = File.ReadAllBytes(IndexPath);
            using var document = JsonDocument.Parse(bytes);
            if (document.RootElement.ValueKind != JsonValueKind.Object ||
                !document.RootElement.TryGetProperty("weight_map", out var mapElement) ||
                mapElement.ValueKind != JsonValueKind.Object)
            {
                throw CheckpointError(
                    "Checkpoint index must be an object containing a weight_map.");
            }

            string root = Path.GetDirectoryName(IndexPath) ??
                throw CheckpointError("The checkpoint index has no containing directory.");
            var assignments = new Dictionary<string, string>(StringComparer.Ordinal);
            var shardPaths = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (var property in mapElement.EnumerateObject())
            {
                string? shardName = property.Value.ValueKind == JsonValueKind.String
                    ? property.Value.GetString()
                    : null;
                if (string.IsNullOrWhiteSpace(property.Name) ||
                    string.IsNullOrWhiteSpace(shardName) ||
                    Path.IsPathRooted(shardName))
                {
                    throw CheckpointError(
                        "Every weight_map entry must have a tensor name and relative shard path.");
                }

                string shardPath = Path.GetFullPath(Path.Combine(root, shardName));
                string relative = Path.GetRelativePath(root, shardPath);
                if (relative == ".." ||
                    relative.StartsWith($"..{Path.DirectorySeparatorChar}", StringComparison.Ordinal) ||
                    relative.StartsWith($"..{Path.AltDirectorySeparatorChar}", StringComparison.Ordinal))
                {
                    throw CheckpointError(
                        $"Shard path '{shardName}' escapes the checkpoint directory.");
                }
                assignments.Add(property.Name, shardPath);
                shardPaths.Add(shardPath);
            }
            if (assignments.Count == 0)
            {
                throw CheckpointError("Checkpoint weight_map is empty.");
            }

            _shardsByTensor = assignments;
            foreach (string shardPath in shardPaths)
            {
                var shardInfo = new FileInfo(shardPath);
                if (!shardInfo.Exists ||
                    (shardInfo.Attributes & FileAttributes.ReparsePoint) != 0)
                {
                    throw CheckpointError(
                        $"Checkpoint shard '{shardPath}' is missing or is a reparse point.");
                }
                _shards.Add(shardPath, new SafetensorsFile(shardPath));
            }

            foreach ((string tensorName, string shardPath) in assignments)
            {
                if (!_shards[shardPath].Tensors.ContainsKey(tensorName))
                {
                    throw CheckpointError(
                        $"Indexed tensor '{tensorName}' was not found in shard '{shardPath}'.");
                }
            }
            MetadataJson = document.RootElement.TryGetProperty("metadata", out var metadata)
                ? metadata.GetRawText()
                : null;
        }
        catch (PowerTorchException)
        {
            Dispose();
            throw;
        }
        catch (JsonException exception)
        {
            Dispose();
            throw CheckpointError("The checkpoint index is not valid JSON.", exception);
        }
        catch (IOException exception)
        {
            Dispose();
            throw CheckpointError("The checkpoint index could not be read.", exception);
        }
        catch (UnauthorizedAccessException exception)
        {
            Dispose();
            throw CheckpointError("The checkpoint index could not be accessed.", exception);
        }
        catch (Exception exception) when (
            exception is ArgumentException or InvalidOperationException)
        {
            Dispose();
            throw CheckpointError("The checkpoint index contains invalid entries.", exception);
        }
    }

    public string IndexPath { get; }

    public string? MetadataJson { get; }

    public IReadOnlyList<string> TensorNames =>
        _shardsByTensor.Keys.Order(StringComparer.Ordinal).ToArray();

    public IReadOnlyList<string> ShardPaths =>
        _shards.Keys.Order(StringComparer.Ordinal).ToArray();

    public SafeTensorDescriptor GetTensor(string name)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("A tensor name is required.", nameof(name));
        }
        if (!_shardsByTensor.TryGetValue(name, out string? shardPath))
        {
            throw new KeyNotFoundException(
                $"Tensor '{name}' was not found in checkpoint index '{IndexPath}'.");
        }
        return _shards[shardPath].GetTensor(name);
    }

    public async ValueTask<Tensor> LoadTensorAsync(
        string name,
        long maximumBytes = 1_000_000_000,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("A tensor name is required.", nameof(name));
        }
        if (!_shardsByTensor.TryGetValue(name, out string? shardPath))
        {
            throw new KeyNotFoundException(
                $"Tensor '{name}' was not found in checkpoint index '{IndexPath}'.");
        }
        SafeTensorDescriptor descriptor = _shards[shardPath].GetTensor(name);
        if (descriptor.ByteLength > maximumBytes || maximumBytes <= 0)
        {
            throw CheckpointError(
                $"Tensor '{name}' is {descriptor.ByteLength} bytes, exceeding the {maximumBytes}-byte limit.");
        }
        byte[] bytes = await _shards[shardPath]
            .ReadTensorBytesAsync(name, cancellationToken);
        return TensorFactory.FromRawBytes(descriptor.DataType, descriptor.Shape, bytes);
    }

    public async ValueTask<IReadOnlyDictionary<string, Tensor>> LoadStateAsync(
        IEnumerable<string> names,
        long maximumTotalBytes = 1_000_000_000,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(names);
        ObjectDisposedException.ThrowIf(_disposed, this);
        string[] requested = names.Distinct(StringComparer.Ordinal).ToArray();
        long totalBytes = 0;
        foreach (string name in requested)
        {
            totalBytes = checked(totalBytes + GetTensor(name).ByteLength);
            if (totalBytes > maximumTotalBytes || maximumTotalBytes <= 0)
            {
                throw CheckpointError(
                    $"Requested state exceeds the {maximumTotalBytes}-byte aggregate limit.");
            }
        }

        var state = new Dictionary<string, Tensor>(requested.Length, StringComparer.Ordinal);
        try
        {
            foreach (string name in requested)
            {
                state.Add(
                    name,
                    await LoadTensorAsync(
                        name,
                        maximumTotalBytes,
                        cancellationToken));
            }
            return state;
        }
        catch
        {
            foreach (Tensor tensor in state.Values)
            {
                tensor.Dispose();
            }
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
        foreach (SafetensorsFile shard in _shards.Values)
        {
            shard.Dispose();
        }
    }

    private static PowerTorchException CheckpointError(
        string message,
        Exception? innerException = null) =>
        new(
            PowerTorchErrorCategory.Serialization,
            message,
            innerException);
}
