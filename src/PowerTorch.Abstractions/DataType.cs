namespace PowerTorch;

public enum DataType
{
    Bool = 1,
    UInt8 = 2,
    Int8 = 3,
    Int16 = 4,
    Int32 = 5,
    Int64 = 6,
    Float16 = 16,
    BFloat16 = 17,
    Float32 = 18,
    Float64 = 19,
    Complex64 = 32,
    Complex128 = 33,
}

public static class DataTypeInfo
{
    public static int ElementSize(DataType dataType) => dataType switch
    {
        DataType.Bool or DataType.UInt8 or DataType.Int8 => 1,
        DataType.Int16 or DataType.Float16 or DataType.BFloat16 => 2,
        DataType.Int32 or DataType.Float32 => 4,
        DataType.Int64 or DataType.Float64 or DataType.Complex64 => 8,
        DataType.Complex128 => 16,
        _ => throw new ArgumentOutOfRangeException(nameof(dataType), dataType, "Unsupported dtype."),
    };

    public static DataType FromSafetensorsCode(string code) => code switch
    {
        "BOOL" => DataType.Bool,
        "U8" => DataType.UInt8,
        "I8" => DataType.Int8,
        "I16" => DataType.Int16,
        "I32" => DataType.Int32,
        "I64" => DataType.Int64,
        "F16" => DataType.Float16,
        "BF16" => DataType.BFloat16,
        "F32" => DataType.Float32,
        "F64" => DataType.Float64,
        "C64" => DataType.Complex64,
        "C128" => DataType.Complex128,
        _ => throw new NotSupportedException($"Safetensors dtype '{code}' is unsupported."),
    };

    public static Type ClrType(DataType dataType) => dataType switch
    {
        DataType.Bool => typeof(bool),
        DataType.UInt8 => typeof(byte),
        DataType.Int8 => typeof(sbyte),
        DataType.Int16 => typeof(short),
        DataType.Int32 => typeof(int),
        DataType.Int64 => typeof(long),
        DataType.Float16 => typeof(Half),
        DataType.Float32 => typeof(float),
        DataType.Float64 => typeof(double),
        _ => throw new NotSupportedException($"Data type '{dataType}' has no direct CLR primitive."),
    };
}
