using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Runtime.InteropServices;

namespace PowerTorch;

[StructLayout(LayoutKind.Sequential)]
public readonly struct BFloat16 : IEquatable<BFloat16>, IComparable<BFloat16>
{
    public BFloat16(ushort rawBits)
    {
        RawBits = rawBits;
    }

    public ushort RawBits { get; }

    public static explicit operator BFloat16(float value) => FromSingle(value);

    public static implicit operator float(BFloat16 value) => value.ToSingle();

    public static explicit operator BFloat16(double value) => FromSingle((float)value);

    public static implicit operator double(BFloat16 value) => value.ToSingle();

    public static BFloat16 FromSingle(float value)
    {
        uint bits = BitConverter.SingleToUInt32Bits(value);
        uint roundingBias = 0x7FFFu + ((bits >> 16) & 1u);
        if ((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0)
        {
            return new BFloat16((ushort)((bits >> 16) | 0x0040u));
        }

        return new BFloat16((ushort)((bits + roundingBias) >> 16));
    }

    public float ToSingle() =>
        BitConverter.UInt32BitsToSingle((uint)RawBits << 16);

    public int CompareTo(BFloat16 other) => ToSingle().CompareTo(other.ToSingle());

    public bool Equals(BFloat16 other) => RawBits == other.RawBits;

    public override bool Equals([NotNullWhen(true)] object? obj) =>
        obj is BFloat16 other && Equals(other);

    public override int GetHashCode() => RawBits.GetHashCode();

    public override string ToString() => ToSingle().ToString(CultureInfo.InvariantCulture);

    public static bool operator ==(BFloat16 left, BFloat16 right) => left.Equals(right);

    public static bool operator !=(BFloat16 left, BFloat16 right) => !left.Equals(right);

    public static bool operator <(BFloat16 left, BFloat16 right) => left.CompareTo(right) < 0;

    public static bool operator <=(BFloat16 left, BFloat16 right) => left.CompareTo(right) <= 0;

    public static bool operator >(BFloat16 left, BFloat16 right) => left.CompareTo(right) > 0;

    public static bool operator >=(BFloat16 left, BFloat16 right) => left.CompareTo(right) >= 0;
}
