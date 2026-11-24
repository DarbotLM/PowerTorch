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
            ["graph_compile"] = false,
            ["autograd"] = false,
        });

    public void SelfTest()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _deviceHandle.SelfTest();
    }

    public Tensor Add(Tensor left, Tensor right) => throw Unsupported("tensor.add");

    public Tensor Multiply(Tensor left, Tensor right) => throw Unsupported("tensor.multiply");

    public Tensor Subtract(Tensor left, Tensor right) => throw Unsupported("tensor.subtract");

    public Tensor Divide(Tensor left, Tensor right) => throw Unsupported("tensor.divide");

    public Tensor Sigmoid(Tensor input) => throw Unsupported("tensor.sigmoid");

    public Tensor Sum(Tensor input) => throw Unsupported("tensor.sum");

    public Tensor Mean(Tensor input) => throw Unsupported("tensor.mean");

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
