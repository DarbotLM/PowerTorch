using System.Text.Json;
using PowerTorch.Serialization;

const int usageExitCode = 2;
const int runtimeExitCode = 3;

if (args.Length < 2 || !args[0].Equals("onnx-inspect", StringComparison.Ordinal))
{
    Console.Error.WriteLine("usage: PowerTorch.Cli onnx-inspect <model.onnx> [--cpu] [--adapter N]");
    return usageExitCode;
}

string path = args[1];
bool cpu = args.Contains("--cpu", StringComparer.Ordinal);
int adapterIndex = 0;
int adapterOption = Array.FindIndex(args, value => value == "--adapter");
if (adapterOption >= 0)
{
    if (adapterOption + 1 >= args.Length ||
        !int.TryParse(args[adapterOption + 1], out adapterIndex) ||
        adapterIndex < 0)
    {
        Console.Error.WriteLine("--adapter requires a non-negative integer.");
        return usageExitCode;
    }
}

try
{
    using var model = OnnxModel.Open(
        path,
        useDirectML: !cpu,
        adapterIndex: adapterIndex);
    Console.WriteLine(JsonSerializer.Serialize(
        model.Metadata,
        new JsonSerializerOptions(JsonSerializerDefaults.Web)));
    return 0;
}
catch (Exception exception)
{
    Console.Error.WriteLine(exception);
    return runtimeExitCode;
}
