using PowerTorch.Core;
using PowerTorch.Cpu;

namespace PowerTorch.Autograd;

public sealed class Variable : IDisposable
{
    private readonly List<BackwardEdge>? _edges;
    private bool _disposed;

    private Variable(
        Tensor data,
        bool requiresGradient,
        List<BackwardEdge>? edges = null)
    {
        Data = data;
        RequiresGradient = requiresGradient;
        _edges = edges;
    }

    public Tensor Data { get; }

    public bool RequiresGradient { get; }

    public Tensor? Gradient { get; private set; }

    public bool IsLeaf => _edges is null;

    public static Variable Leaf(Tensor data, bool requiresGradient = true) =>
        new(data, requiresGradient);

    public static Variable Constant(Tensor data) => new(data, requiresGradient: false);

    public Variable Add(Variable other) =>
        Binary(other, AddTensors, _ => gradient => gradient.Clone());

    public Variable Multiply(Variable other) =>
        Binary(
            other,
            MultiplyTensors,
            other => gradient => MultiplyTensors(gradient, other.Data),
            self => gradient => MultiplyTensors(gradient, self.Data));

    public Variable Subtract(Variable other) =>
        Binary(
            other,
            SubtractTensors,
            _ => gradient => gradient.Clone(),
            _ => gradient => Negate(gradient));

    public Variable Divide(Variable other)
    {
        bool needsGradient = RequiresGradient || other.RequiresGradient;
        var result = new Variable(
            DivideTensors(Data, other.Data),
            needsGradient,
            needsGradient ? new List<BackwardEdge>() : null);
        if (!needsGradient)
        {
            return result;
        }
        if (RequiresGradient)
        {
            result._edges!.Add(new BackwardEdge(
                this,
                gradient => DivideTensors(gradient, other.Data)));
        }
        if (other.RequiresGradient)
        {
            result._edges!.Add(new BackwardEdge(
                other,
                gradient =>
                {
                    var numerator = MultiplyTensors(gradient, Data);
                    var denominator = MultiplyTensors(other.Data, other.Data);
                    var quotient = DivideTensors(numerator, denominator);
                    numerator.Dispose();
                    denominator.Dispose();
                    var negative = Negate(quotient);
                    quotient.Dispose();
                    return negative;
                }));
        }
        return result;
    }

    public Variable Sigmoid()
    {
        var output = Runtime().Sigmoid(Data);
        bool needsGradient = RequiresGradient;
        var edges = needsGradient ? new List<BackwardEdge>() : null;
        var variable = new Variable(output, needsGradient, edges);
        if (needsGradient)
        {
            edges!.Add(new BackwardEdge(
                this,
                gradient => SigmoidGradient(output, gradient)));
        }
        return variable;
    }

    public Variable Sum()
    {
        var output = Runtime().Sum(Data);
        bool needsGradient = RequiresGradient;
        var edges = needsGradient ? new List<BackwardEdge>() : null;
        var variable = new Variable(output, needsGradient, edges);
        if (needsGradient)
        {
            edges!.Add(new BackwardEdge(
                this,
                gradient => ExpandScalar(gradient, Data.Descriptor)));
        }
        return variable;
    }

    public Variable Mean()
    {
        var output = Runtime().Mean(Data);
        bool needsGradient = RequiresGradient;
        var edges = needsGradient ? new List<BackwardEdge>() : null;
        var variable = new Variable(output, needsGradient, edges);
        if (needsGradient)
        {
            edges!.Add(new BackwardEdge(
                this,
                gradient =>
                {
                    var expanded = ExpandScalar(gradient, Data.Descriptor);
                    var scaled = Scale(expanded, 1.0 / Data.ElementCount);
                    expanded.Dispose();
                    return scaled;
                }));
        }
        return variable;
    }

    public void Backward(Tensor? gradient = null)
    {
        if (!RequiresGradient)
        {
            throw new InvalidOperationException(
                "Backward was requested for a value that does not require gradients.");
        }

        Tensor initialGradient = gradient ?? OnesLike(Data);
        AccumulateGradient(this, initialGradient);

        var visited = new HashSet<Variable>(ReferenceEqualityComparer.Instance);
        var order = new List<Variable>();
        VisitTopological(this, visited, order);
        for (int index = order.Count - 1; index >= 0; index--)
        {
            var node = order[index];
            if (node.Gradient is null || node._edges is null)
            {
                continue;
            }
            foreach (var edge in node._edges)
            {
                AccumulateGradient(edge.Parent, edge.Gradient(node.Gradient));
            }
        }
    }

    public void ZeroGradient()
    {
        Gradient?.Dispose();
        Gradient = null;
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }
        _disposed = true;
        Gradient?.Dispose();
        Data.Dispose();
    }

    private Variable Binary(
        Variable other,
        Func<Tensor, Tensor, Tensor> forward,
        Func<Variable, Func<Tensor, Tensor>> leftGradient,
        Func<Variable, Func<Tensor, Tensor>>? rightGradient = null)
    {
        bool needsGradient = RequiresGradient || other.RequiresGradient;
        var edges = needsGradient ? new List<BackwardEdge>() : null;
        var variable = new Variable(forward(Data, other.Data), needsGradient, edges);
        if (!needsGradient)
        {
            return variable;
        }
        if (RequiresGradient)
        {
            edges!.Add(new BackwardEdge(this, leftGradient(other)));
        }
        if (other.RequiresGradient)
        {
            var factory = rightGradient?.Invoke(this) ?? (gradient => gradient.Clone());
            edges!.Add(new BackwardEdge(other, factory));
        }
        return variable;
    }

    private static PowerTorchRuntime Runtime()
    {
        var runtime = new PowerTorchRuntime();
        runtime.Register(new CpuBackend());
        return runtime;
    }

    private static Tensor AddTensors(Tensor left, Tensor right) => Runtime().Add(left, right);

    private static Tensor MultiplyTensors(Tensor left, Tensor right) => Runtime().Multiply(left, right);

    private static Tensor SubtractTensors(Tensor left, Tensor right) => Runtime().Subtract(left, right);

    private static Tensor DivideTensors(Tensor left, Tensor right) => Runtime().Divide(left, right);

    private static void VisitTopological(
        Variable variable,
        HashSet<Variable> visited,
        List<Variable> order)
    {
        if (!visited.Add(variable))
        {
            return;
        }
        if (variable._edges is not null)
        {
            foreach (var edge in variable._edges)
            {
                VisitTopological(edge.Parent, visited, order);
            }
        }
        order.Add(variable);
    }

    private static void AccumulateGradient(Variable variable, Tensor value)
    {
        if (!variable.RequiresGradient)
        {
            value.Dispose();
            return;
        }
        if (variable.Gradient is null)
        {
            variable.Gradient = value;
            return;
        }
        var combined = Runtime().Add(variable.Gradient, value);
        variable.Gradient.Dispose();
        value.Dispose();
        variable.Gradient = combined;
    }

    private static Tensor OnesLike(Tensor tensor)
    {
        var result = TensorFactory.Empty(tensor.DataType, tensor.Shape.ToArray());
        if (tensor.DataType == DataType.Float32)
        {
            result.AsSpan<float>().Fill(1.0f);
        }
        else if (tensor.DataType == DataType.Float64)
        {
            result.AsSpan<double>().Fill(1.0);
        }
        else
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.DTypeMismatch,
                $"Autograd seed requires floating dtype; got {tensor.DataType}.");
        }
        return result;
    }

    private static Tensor ExpandScalar(Tensor scalar, TensorDescriptor descriptor)
    {
        var result = TensorFactory.Empty(descriptor.DataType, descriptor.Shape.ToArray());
        if (descriptor.DataType == DataType.Float32)
        {
            result.AsSpan<float>().Fill(scalar.AsSpan<float>()[0]);
        }
        else if (descriptor.DataType == DataType.Float64)
        {
            result.AsSpan<double>().Fill(scalar.AsSpan<double>()[0]);
        }
        else
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.DTypeMismatch,
                "Scalar expansion requires a floating dtype.");
        }
        return result;
    }

    private static Tensor Scale(Tensor tensor, double factor)
    {
        var result = TensorFactory.Empty(tensor.DataType, tensor.Shape.ToArray());
        if (tensor.DataType == DataType.Float32)
        {
            var source = tensor.AsSpan<float>();
            var destination = result.AsSpan<float>();
            for (int index = 0; index < source.Length; index++)
            {
                destination[index] = source[index] * (float)factor;
            }
        }
        else if (tensor.DataType == DataType.Float64)
        {
            var source = tensor.AsSpan<double>();
            var destination = result.AsSpan<double>();
            for (int index = 0; index < source.Length; index++)
            {
                destination[index] = source[index] * factor;
            }
        }
        else
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.DTypeMismatch,
                "Scaling requires a floating dtype.");
        }
        return result;
    }

    private static Tensor SigmoidGradient(Tensor output, Tensor gradient)
    {
        var result = TensorFactory.Empty(output.DataType, output.Shape.ToArray());
        if (output.DataType == DataType.Float32)
        {
            var values = output.AsSpan<float>();
            var incoming = gradient.AsSpan<float>();
            var destination = result.AsSpan<float>();
            for (int index = 0; index < values.Length; index++)
            {
                destination[index] = incoming[index] * values[index] * (1.0f - values[index]);
            }
        }
        else if (output.DataType == DataType.Float64)
        {
            var values = output.AsSpan<double>();
            var incoming = gradient.AsSpan<double>();
            var destination = result.AsSpan<double>();
            for (int index = 0; index < values.Length; index++)
            {
                destination[index] = incoming[index] * values[index] * (1.0 - values[index]);
            }
        }
        else
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.DTypeMismatch,
                "Sigmoid gradient requires a floating dtype.");
        }
        return result;
    }

    private static Tensor Negate(Tensor tensor) => Scale(tensor, -1.0);

    private sealed record BackwardEdge(
        Variable Parent,
        Func<Tensor, Tensor> Gradient);
}
