using PowerTorch.Autograd;
using PowerTorch.Core;

namespace PowerTorch.Optim;

public sealed class AdamW : IDisposable
{
    private readonly Dictionary<Variable, AdamState> _states =
        new(ReferenceEqualityComparer.Instance);
    private readonly double _learningRate;
    private readonly double _beta1;
    private readonly double _beta2;
    private readonly double _epsilon;
    private readonly double _weightDecay;
    private long _step;

    public AdamW(
        double learningRate = 1.0e-4,
        double beta1 = 0.9,
        double beta2 = 0.999,
        double epsilon = 1.0e-8,
        double weightDecay = 0.01)
    {
        if (learningRate <= 0 || !double.IsFinite(learningRate))
        {
            throw new ArgumentOutOfRangeException(
                nameof(learningRate), "Learning rate must be finite and positive.");
        }
        if (beta1 is < 0 or >= 1 || beta2 is < 0 or >= 1)
        {
            throw new ArgumentOutOfRangeException(
                nameof(beta1), "Beta values must be in [0, 1).");
        }
        if (epsilon <= 0 || weightDecay < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(epsilon));
        }

        _learningRate = learningRate;
        _beta1 = beta1;
        _beta2 = beta2;
        _epsilon = epsilon;
        _weightDecay = weightDecay;
    }

    public long Step(IEnumerable<Variable> parameters)
    {
        _step++;
        foreach (var parameter in parameters)
        {
            var gradient = parameter.Gradient ??
                throw new InvalidOperationException(
                    "AdamW parameter is missing a gradient. Run backward first.");
            if (!parameter.RequiresGradient)
            {
                throw new InvalidOperationException(
                    "AdamW cannot update a variable that does not require gradients.");
            }
            if (!parameter.Data.Descriptor.IsContiguous ||
                !gradient.Descriptor.IsContiguous ||
                parameter.Data.Shape.Count != gradient.Shape.Count ||
                !parameter.Data.Shape.SequenceEqual(gradient.Shape) ||
                parameter.Data.DataType != gradient.DataType)
            {
                throw new PowerTorchException(
                    PowerTorchErrorCategory.ShapeMismatch,
                    "AdamW requires contiguous parameter/gradient tensors with identical shape and dtype.");
            }

            if (!_states.TryGetValue(parameter, out var state))
            {
                state = new AdamState(
                    TensorFactory.Empty(parameter.Data.DataType, parameter.Data.Shape.ToArray()),
                    TensorFactory.Empty(parameter.Data.DataType, parameter.Data.Shape.ToArray()));
                _states.Add(parameter, state);
            }
            Update(parameter.Data, gradient, state);
        }
        return _step;
    }

    public static void ZeroGrad(IEnumerable<Variable> parameters)
    {
        foreach (var parameter in parameters)
        {
            parameter.ZeroGradient();
        }
    }

    public void Dispose()
    {
        foreach (var state in _states.Values)
        {
            state.FirstMoment.Dispose();
            state.SecondMoment.Dispose();
        }
        _states.Clear();
    }

    private void Update(Tensor parameter, Tensor gradient, AdamState state)
    {
        if (parameter.DataType == DataType.Float32)
        {
            UpdateSpan(
                parameter.AsSpan<float>(),
                gradient.AsSpan<float>(),
                state.FirstMoment.AsSpan<float>(),
                state.SecondMoment.AsSpan<float>(),
                _step,
                (float)_learningRate,
                (float)_beta1,
                (float)_beta2,
                (float)_epsilon,
                (float)_weightDecay);
            return;
        }
        if (parameter.DataType == DataType.Float64)
        {
            UpdateSpan(
                parameter.AsSpan<double>(),
                gradient.AsSpan<double>(),
                state.FirstMoment.AsSpan<double>(),
                state.SecondMoment.AsSpan<double>(),
                _step,
                _learningRate,
                _beta1,
                _beta2,
                _epsilon,
                _weightDecay);
            return;
        }
        throw new PowerTorchException(
            PowerTorchErrorCategory.DTypeMismatch,
            $"AdamW supports Float32/Float64; received {parameter.DataType}.");
    }

    private static void UpdateSpan(
        Span<float> parameter,
        ReadOnlySpan<float> gradient,
        Span<float> firstMoment,
        Span<float> secondMoment,
        long step,
        float learningRate,
        float beta1,
        float beta2,
        float epsilon,
        float weightDecay)
    {
        float one = 1.0f;
        float beta1Correction = one - MathF.Pow(beta1, step);
        float beta2Correction = one - MathF.Pow(beta2, step);
        for (int index = 0; index < parameter.Length; index++)
        {
            firstMoment[index] =
                beta1 * firstMoment[index] + (one - beta1) * gradient[index];
            secondMoment[index] =
                beta2 * secondMoment[index] +
                (one - beta2) * gradient[index] * gradient[index];
            float firstHat = firstMoment[index] / beta1Correction;
            float secondHat = secondMoment[index] / beta2Correction;
            float decay = weightDecay * parameter[index];
            parameter[index] -= learningRate *
                (firstHat / (MathF.Sqrt(secondHat) + epsilon) + decay);
        }
    }

    private static void UpdateSpan(
        Span<double> parameter,
        ReadOnlySpan<double> gradient,
        Span<double> firstMoment,
        Span<double> secondMoment,
        long step,
        double learningRate,
        double beta1,
        double beta2,
        double epsilon,
        double weightDecay)
    {
        double beta1Correction = 1.0 - Math.Pow(beta1, step);
        double beta2Correction = 1.0 - Math.Pow(beta2, step);
        for (int index = 0; index < parameter.Length; index++)
        {
            firstMoment[index] =
                beta1 * firstMoment[index] + (1.0 - beta1) * gradient[index];
            secondMoment[index] =
                beta2 * secondMoment[index] +
                (1.0 - beta2) * gradient[index] * gradient[index];
            double firstHat = firstMoment[index] / beta1Correction;
            double secondHat = secondMoment[index] / beta2Correction;
            double decay = weightDecay * parameter[index];
            parameter[index] -= learningRate *
                (firstHat / (Math.Sqrt(secondHat) + epsilon) + decay);
        }
    }

    private sealed record AdamState(Tensor FirstMoment, Tensor SecondMoment);
}
