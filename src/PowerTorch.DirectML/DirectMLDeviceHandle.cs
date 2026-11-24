using System.Text;
using Microsoft.Win32.SafeHandles;

namespace PowerTorch.DirectML;

internal sealed class DirectMLDeviceHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    public DirectMLDeviceHandle() : base(ownsHandle: true)
    {
    }

    private DirectMLDeviceHandle(IntPtr handle) : base(ownsHandle: true)
    {
        SetHandle(handle);
    }

    public static DirectMLDeviceHandle Create(int adapterIndex)
    {
        byte[] error = new byte[2048];
        int status = DirectMLNative.CreateDevice(
            adapterIndex,
            out IntPtr nativeHandle,
            error,
            (uint)error.Length);
        if (status != 0 || nativeHandle == IntPtr.Zero)
        {
            throw DirectMLException(status, error);
        }
        return new DirectMLDeviceHandle(nativeHandle);
    }

    public void SelfTest()
    {
        byte[] error = new byte[2048];
        int status = DirectMLNative.SelfTest(handle, error, (uint)error.Length);
        if (status != 0)
        {
            throw DirectMLException(status, error);
        }
    }

    protected override bool ReleaseHandle()
    {
        DirectMLNative.ReleaseDevice(handle);
        return true;
    }

    private static PowerTorchException DirectMLException(int status, byte[] error)
    {
        int terminator = Array.IndexOf(error, (byte)0);
        string message = Encoding.UTF8.GetString(
            error,
            0,
            terminator >= 0 ? terminator : error.Length);
        return new PowerTorchException(
            PowerTorchErrorCategory.Backend,
            $"DirectML native status {status}: {message}");
    }
}
