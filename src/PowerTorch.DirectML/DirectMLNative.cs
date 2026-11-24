using System.Runtime.InteropServices;

namespace PowerTorch.DirectML;

internal static class DirectMLNative
{
    private const string Library = "PowerTorch.DirectML.Native";

    [StructLayout(LayoutKind.Sequential)]
    internal struct AbiInfo
    {
        internal uint Size;
        internal ushort Major;
        internal ushort Minor;
        internal uint StructVersion;
    }

    [DllImport(Library, EntryPoint = "pt_dml_get_abi_info", CallingConvention = CallingConvention.Cdecl)]
    internal static extern void GetAbiInfo(out AbiInfo info);

    [DllImport(Library, EntryPoint = "pt_dml_adapter_count", CallingConvention = CallingConvention.Cdecl)]
    internal static extern int AdapterCount();

    [DllImport(
        Library,
        EntryPoint = "pt_dml_adapter_name",
        CallingConvention = CallingConvention.Cdecl,
        CharSet = CharSet.Ansi)]
    internal static extern int AdapterName(int adapterIndex, byte[] output, uint outputBytes);

    [DllImport(
        Library,
        EntryPoint = "pt_dml_create_device",
        CallingConvention = CallingConvention.Cdecl,
        CharSet = CharSet.Ansi)]
    internal static extern int CreateDevice(
        int adapterIndex,
        out IntPtr device,
        byte[] error,
        uint errorBytes);

    [DllImport(Library, EntryPoint = "pt_dml_release_device", CallingConvention = CallingConvention.Cdecl)]
    internal static extern void ReleaseDevice(IntPtr device);

    [DllImport(
        Library,
        EntryPoint = "pt_dml_self_test",
        CallingConvention = CallingConvention.Cdecl,
        CharSet = CharSet.Ansi)]
    internal static extern int SelfTest(IntPtr device, byte[] error, uint errorBytes);

    [DllImport(
        Library,
        EntryPoint = "pt_dml_add_f32",
        CallingConvention = CallingConvention.Cdecl,
        CharSet = CharSet.Ansi)]
    internal static extern int AddF32(
        IntPtr device,
        float[] left,
        float[] right,
        float[] output,
        ulong elementCount,
        byte[] error,
        uint errorBytes);
}
