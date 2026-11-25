using System.Text;
using PowerTorch.Core;

namespace PowerTorch.DirectML;

public sealed class DirectMLBackend : ITensorBackend, IDisposable
{
    private readonly DirectMLDeviceHandle _deviceHandle;
    private bool _disposed;

    private DirectMLBackend(Device device, DirectMLDeviceHandle deviceHandle, string adapterName)
    {
        Device = device;
        _deviceHandle = deviceHandle;
        AdapterName = adapterName;
    }

    public Device Device { get; }

    public string AdapterName { get; }

    public static DirectMLBackend Create(int adapterIndex = 0)
    {
        ValidateAbi();
        var handle = DirectMLDeviceHandle.Create(adapterIndex);
        return new DirectMLBackend(
            Device.DirectML(adapterIndex),
            handle,
            GetAdapterName(adapterIndex));
    }

    public static DirectMLAbi GetAbi()
    {
        DirectMLNative.GetAbiInfo(out var native);
        return new DirectMLAbi(native.Major, native.Minor, native.StructVersion);
    }

    public static IReadOnlyList<string> GetAdapterNames()
    {
        int count = DirectMLNative.AdapterCount();
        if (count < 0)
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.Backend,
                $"DirectML adapter enumeration failed with native status {-count}.");
        }

        var adapters = new List<string>(count);
        for (int index = 0; index < count; index++)
        {
            adapters.Add(GetAdapterName(index));
        }
        return adapters;
    }

    public BackendCapability GetCapabilities() => new(
        Device,
        "PowerTorch.DirectML",
        AdapterName,
        true,
        [
            nameof(DataType.Float16),
            nameof(DataType.Float32),
            nameof(DataType.Int32),
            nameof(DataType.Int64),
        ],
        new Dictionary<string, bool>
        {
            ["native_abi"] = true,
            ["device_enumeration"] = true,
            ["fence_synchronization"] = false,
            ["tensor.add"] = false,
            ["host_f32.add"] = true,
            ["graph_compile"] = false,
            ["device.transfer"] = false,
            ["tensor_residency"] = false,
            ["autograd"] = false,
        });

    public void SelfTest()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _deviceHandle.SelfTest();
    }

    public float[] AddF32(ReadOnlySpan<float> left, ReadOnlySpan<float> right)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return _deviceHandle.AddF32(left, right);
    }

    public Tensor Add(Tensor left, Tensor right) => throw Unsupported("tensor.add");

    public Tensor Multiply(Tensor left, Tensor right) => throw Unsupported("tensor.multiply");

    public Tensor Subtract(Tensor left, Tensor right) => throw Unsupported("tensor.subtract");

    public Tensor Divide(Tensor left, Tensor right) => throw Unsupported("tensor.divide");

    public Tensor Sigmoid(Tensor input) => throw Unsupported("tensor.sigmoid");

    public Tensor Sum(Tensor input) => throw Unsupported("tensor.sum");

    public Tensor Mean(Tensor input) => throw Unsupported("tensor.mean");

    public Tensor Reshape(Tensor input, IReadOnlyList<long> shape) =>
        throw Unsupported("tensor.reshape");

    public Tensor Permute(Tensor input, IReadOnlyList<int> dimensions) =>
        throw Unsupported("tensor.permute");

    public Tensor Transpose(Tensor input, int firstDimension, int secondDimension) =>
        throw Unsupported("tensor.transpose");

    public Tensor Slice(
        Tensor input,
        int dimension,
        long start,
        long endExclusive,
        long stride) =>
        throw Unsupported("tensor.slice");

    public Tensor Matmul(Tensor left, Tensor right) => throw Unsupported("tensor.matmul");

    public Tensor Softmax(Tensor input, int dimension) =>
        throw Unsupported("tensor.softmax");

    public Tensor LayerNorm(
        Tensor input,
        IReadOnlyList<long> normalizedShape,
        Tensor? weight,
        Tensor? bias,
        double epsilon) =>
        throw Unsupported("tensor.layer_norm");

    public Tensor RmsNorm(
        Tensor input,
        IReadOnlyList<long> normalizedShape,
        Tensor? weight,
        double epsilon) =>
        throw Unsupported("tensor.rms_norm");

    public Tensor Embedding(Tensor weight, Tensor indices) =>
        throw Unsupported("tensor.embedding");

    public Tensor Conv1D(
        Tensor input,
        Tensor weight,
        Tensor? bias,
        int stride,
        int padding,
        int dilation,
        int groups,
        bool causal) =>
        throw Unsupported("tensor.conv1d");

    public Tensor Rotary(
        Tensor input,
        Tensor positions,
        int sequenceDimension,
        int rotaryDimension,
        double baseFrequency,
        bool interleaved) =>
        throw Unsupported("tensor.rotary");

    public Tensor CausalAttention(
        Tensor query,
        Tensor key,
        Tensor value,
        double? scale) =>
        throw Unsupported("tensor.causal_attention");

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }
        _disposed = true;
        _deviceHandle.Dispose();
    }

    private static void ValidateAbi()
    {
        var abi = GetAbi();
        if (abi.Major != 1)
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.Backend,
                $"DirectML ABI major mismatch: managed=1 native={abi.Major}.");
        }
    }

    private static string GetAdapterName(int adapterIndex)
    {
        byte[] name = new byte[512];
        int status = DirectMLNative.AdapterName(adapterIndex, name, (uint)name.Length);
        if (status != 0)
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.Backend,
                $"DirectML adapter name failed with native status {status}.");
        }
        int terminator = Array.IndexOf(name, (byte)0);
        return Encoding.UTF8.GetString(name, 0, terminator >= 0 ? terminator : name.Length);
    }

    private static PowerTorchException Unsupported(string operation) => new(
        PowerTorchErrorCategory.Unsupported,
        $"DirectML operation '{operation}' is declared planned but is not implemented in ABI v1.");
}

public readonly record struct DirectMLAbi(ushort Major, ushort Minor, uint StructVersion);
