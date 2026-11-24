using System.Buffers.Binary;
using System.Text;
using System.Text.Json;

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
