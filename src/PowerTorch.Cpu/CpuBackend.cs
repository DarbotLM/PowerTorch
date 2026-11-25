using PowerTorch.Core;

namespace PowerTorch.Cpu;

public sealed class CpuBackend : ITensorBackend
{
    public Device Device => Device.Cpu;

    public BackendCapability GetCapabilities() => new(
        Device,
        "PowerTorch.Cpu",
        Environment.OSVersion.VersionString,
        true,
        [
            nameof(DataType.Int32),
            nameof(DataType.Int64),
            nameof(DataType.BFloat16),
            nameof(DataType.Float32),
            nameof(DataType.Float64),
        ],
        new Dictionary<string, bool>
        {
            ["tensor.add"] = true,
            ["tensor.multiply"] = true,
            ["tensor.subtract"] = true,
            ["tensor.divide"] = true,
            ["tensor.sigmoid"] = true,
            ["tensor.sum"] = true,
            ["tensor.mean"] = true,
            ["tensor.reshape"] = true,
            ["tensor.permute"] = true,
            ["tensor.transpose"] = true,
            ["tensor.slice"] = true,
            ["tensor.matmul"] = true,
            ["tensor.softmax"] = true,
            ["tensor.layer_norm"] = true,
            ["tensor.rms_norm"] = true,
            ["autograd"] = false,
            ["graph_capture"] = false,
        });

    public Tensor Add(Tensor left, Tensor right)
    {
        var output = TensorFactory.Empty(left.DataType, left.Shape.ToArray());
        switch (left.DataType)
        {
            case DataType.Int32:
                AddSpans(left.AsSpan<int>(), right.AsSpan<int>(), output.AsSpan<int>());
                break;
            case DataType.Int64:
                AddSpans(left.AsSpan<long>(), right.AsSpan<long>(), output.AsSpan<long>());
                break;
            case DataType.Float32:
                AddSpans(left.AsSpan<float>(), right.AsSpan<float>(), output.AsSpan<float>());
                break;
            case DataType.Float64:
                AddSpans(left.AsSpan<double>(), right.AsSpan<double>(), output.AsSpan<double>());
                break;
            case DataType.BFloat16:
                ApplyBFloat16(left, right, output, (x, y) => x + y);
                break;
            default:
                throw Unsupported(left.DataType);
        }
        return output;
    }

    public Tensor Multiply(Tensor left, Tensor right)
    {
        var output = TensorFactory.Empty(left.DataType, left.Shape.ToArray());
        switch (left.DataType)
        {
            case DataType.Int32:
                MultiplySpans(left.AsSpan<int>(), right.AsSpan<int>(), output.AsSpan<int>());
                break;
            case DataType.Int64:
                MultiplySpans(left.AsSpan<long>(), right.AsSpan<long>(), output.AsSpan<long>());
                break;
            case DataType.Float32:
                MultiplySpans(left.AsSpan<float>(), right.AsSpan<float>(), output.AsSpan<float>());
                break;
            case DataType.Float64:
                MultiplySpans(left.AsSpan<double>(), right.AsSpan<double>(), output.AsSpan<double>());
                break;
            case DataType.BFloat16:
                ApplyBFloat16(left, right, output, (x, y) => x * y);
                break;
            default:
                throw Unsupported(left.DataType);
        }
        return output;
    }

    public Tensor Subtract(Tensor left, Tensor right)
    {
        var output = TensorFactory.Empty(left.DataType, left.Shape.ToArray());
        switch (left.DataType)
        {
            case DataType.Int32:
                SubtractSpans(left.AsSpan<int>(), right.AsSpan<int>(), output.AsSpan<int>());
                break;
            case DataType.Int64:
                SubtractSpans(left.AsSpan<long>(), right.AsSpan<long>(), output.AsSpan<long>());
                break;
            case DataType.Float32:
                SubtractSpans(left.AsSpan<float>(), right.AsSpan<float>(), output.AsSpan<float>());
                break;
            case DataType.Float64:
                SubtractSpans(left.AsSpan<double>(), right.AsSpan<double>(), output.AsSpan<double>());
                break;
            case DataType.BFloat16:
                ApplyBFloat16(left, right, output, (x, y) => x - y);
                break;
            default:
                throw Unsupported(left.DataType);
        }
        return output;
    }

    public Tensor Divide(Tensor left, Tensor right)
    {
        var output = TensorFactory.Empty(left.DataType, left.Shape.ToArray());
        switch (left.DataType)
        {
            case DataType.Float32:
                DivideSpans(left.AsSpan<float>(), right.AsSpan<float>(), output.AsSpan<float>());
                break;
            case DataType.Float64:
                DivideSpans(left.AsSpan<double>(), right.AsSpan<double>(), output.AsSpan<double>());
                break;
            case DataType.BFloat16:
                ApplyBFloat16(left, right, output, (x, y) => x / y);
                break;
            default:
                throw new PowerTorchException(
                    PowerTorchErrorCategory.Unsupported,
                    $"CPU divide requires a floating dtype; received '{left.DataType}'.");
        }
        return output;
    }

    public Tensor Sigmoid(Tensor input)
    {
        var output = TensorFactory.Empty(input.DataType, input.Shape.ToArray());
        switch (input.DataType)
        {
            case DataType.Float32:
                var source32 = input.AsSpan<float>();
                var destination32 = output.AsSpan<float>();
                for (int index = 0; index < source32.Length; index++)
                {
                    destination32[index] = 1.0f / (1.0f + MathF.Exp(-source32[index]));
                }
                break;
            case DataType.Float64:
                var source64 = input.AsSpan<double>();
                var destination64 = output.AsSpan<double>();
                for (int index = 0; index < source64.Length; index++)
                {
                    destination64[index] = 1.0 / (1.0 + Math.Exp(-source64[index]));
                }
                break;
            case DataType.BFloat16:
                var sourceBFloat16 = input.AsSpan<BFloat16>();
                var destinationBFloat16 = output.AsSpan<BFloat16>();
                for (int index = 0; index < sourceBFloat16.Length; index++)
                {
                    float value = sourceBFloat16[index];
                    destinationBFloat16[index] =
                        (BFloat16)(1.0f / (1.0f + MathF.Exp(-value)));
                }
                break;
            default:
                throw new PowerTorchException(
                    PowerTorchErrorCategory.Unsupported,
                    $"CPU sigmoid requires a floating dtype; received '{input.DataType}'.");
        }
        return output;
    }

    public Tensor Sum(Tensor input) => Reduce(input, average: false);

    public Tensor Mean(Tensor input) => Reduce(input, average: true);

    public Tensor Reshape(Tensor input, IReadOnlyList<long> shape)
    {
        long[] resolved = ResolveInferredShape(input.ElementCount, shape);
        var output = new Tensor(new TensorDescriptor(input.DataType, resolved), input.Device);
        CopyRawElements(input.RawBytes, output.RawBytes);
        return output;
    }

    public Tensor Permute(Tensor input, IReadOnlyList<int> dimensions)
    {
        int rank = input.Shape.Count;
        int[] permutation = dimensions.ToArray();
        if (permutation.Length != rank ||
            permutation.Order().Distinct().Count() != rank ||
            permutation.Any(dimension => dimension < 0 || dimension >= rank))
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.ShapeMismatch,
                $"Permutation must be a permutation of dimensions 0..{rank - 1}.");
        }

        long[] outputShape = permutation.Select(dimension => input.Shape[dimension]).ToArray();
        Tensor output = TensorFactory.Empty(input.DataType, outputShape);
        long[] coordinates = new long[rank];
        for (long outputIndex = 0; outputIndex < output.ElementCount; outputIndex++)
        {
            UnravelIndex(outputIndex, output.Shape, coordinates);
            long inputIndex = 0;
            for (int outputDimension = 0; outputDimension < rank; outputDimension++)
            {
                inputIndex +=
                    coordinates[outputDimension] * input.Strides[permutation[outputDimension]];
            }
            CopyRawElement(input.RawBytes, inputIndex, output.RawBytes, outputIndex, input.DataType);
        }
        return output;
    }

    public Tensor Transpose(Tensor input, int firstDimension, int secondDimension)
    {
        int rank = input.Shape.Count;
        firstDimension = NormalizeDimension(firstDimension, rank);
        secondDimension = NormalizeDimension(secondDimension, rank);
        int[] permutation = Enumerable.Range(0, rank).ToArray();
        (permutation[firstDimension], permutation[secondDimension]) =
            (permutation[secondDimension], permutation[firstDimension]);
        return Permute(input, permutation);
    }

    public Tensor Slice(Tensor input, int dimension, long start, long endExclusive, long stride)
    {
        int rank = input.Shape.Count;
        dimension = NormalizeDimension(dimension, rank);
        if (stride <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(stride), "Slice step must be positive.");
        }

        long dimensionLength = input.Shape[dimension];
        start = start < 0 ? Math.Max(0, dimensionLength + start) : Math.Min(start, dimensionLength);
        endExclusive = endExclusive < 0
            ? Math.Max(0, dimensionLength + endExclusive)
            : Math.Min(endExclusive, dimensionLength);
        if (endExclusive < start)
        {
            throw new ArgumentOutOfRangeException(
                nameof(endExclusive),
                "Slice end must not precede start.");
        }

        long slicedLength = (endExclusive - start + stride - 1) / stride;
        long[] outputShape = input.Shape.ToArray();
        outputShape[dimension] = slicedLength;
        Tensor output = TensorFactory.Empty(input.DataType, outputShape);
        long[] coordinates = new long[rank];
        for (long outputIndex = 0; outputIndex < output.ElementCount; outputIndex++)
        {
            UnravelIndex(outputIndex, output.Shape, coordinates);
            long inputIndex = 0;
            for (int currentDimension = 0; currentDimension < rank; currentDimension++)
            {
                long coordinate = currentDimension == dimension
                    ? start + (coordinates[currentDimension] * stride)
                    : coordinates[currentDimension];
                inputIndex += coordinate * input.Strides[currentDimension];
            }
            CopyRawElement(input.RawBytes, inputIndex, output.RawBytes, outputIndex, input.DataType);
        }
        return output;
    }

    public Tensor Matmul(Tensor left, Tensor right)
    {
        return left.DataType switch
        {
            DataType.BFloat16 => MatmulFloating(
                left,
                right,
                tensor => tensor.AsSpan<BFloat16>().ToArray().Select(value => (float)value).ToArray(),
                (output, values) =>
                {
                    var destination = output.AsSpan<BFloat16>();
                    for (int index = 0; index < destination.Length; index++)
                    {
                        destination[index] = (BFloat16)values[index];
                    }
                }),
            DataType.Float32 => MatmulFloating(
                left,
                right,
                tensor => tensor.AsSpan<float>().ToArray(),
                (output, values) => values.CopyTo(output.AsSpan<float>())),
            DataType.Float64 => MatmulDouble(left, right),
            _ => throw new PowerTorchException(
                PowerTorchErrorCategory.Unsupported,
                $"CPU matmul does not support dtype '{left.DataType}'."),
        };
    }

    public Tensor Softmax(Tensor input, int dimension)
    {
        dimension = NormalizeDimension(dimension, input.Shape.Count);
        return input.DataType switch
        {
            DataType.Float32 => SoftmaxFloat(input, dimension),
            DataType.Float64 => SoftmaxDouble(input, dimension),
            DataType.BFloat16 => SoftmaxBFloat16(input, dimension),
            _ => throw new PowerTorchException(
                PowerTorchErrorCategory.Unsupported,
                $"CPU softmax requires a floating dtype; received '{input.DataType}'."),
        };
    }

    public Tensor LayerNorm(
        Tensor input,
        IReadOnlyList<long> normalizedShape,
        Tensor? weight,
        Tensor? bias,
        double epsilon)
    {
        ValidateNormalization(input, normalizedShape, weight, bias, epsilon);
        return Normalize(input, normalizedShape, weight, bias, epsilon, rmsOnly: false);
    }

    public Tensor RmsNorm(
        Tensor input,
        IReadOnlyList<long> normalizedShape,
        Tensor? weight,
        double epsilon)
    {
        ValidateNormalization(input, normalizedShape, weight, null, epsilon);
        return Normalize(input, normalizedShape, weight, null, epsilon, rmsOnly: true);
    }

    private static void ApplyBFloat16(
        Tensor left,
        Tensor right,
        Tensor output,
        Func<float, float, float> operation)
    {
        var leftValues = left.AsSpan<BFloat16>();
        var rightValues = right.AsSpan<BFloat16>();
        var outputValues = output.AsSpan<BFloat16>();
        for (int index = 0; index < outputValues.Length; index++)
        {
            outputValues[index] = (BFloat16)operation(leftValues[index], rightValues[index]);
        }
    }

    private static void CopyRawElements(ReadOnlySpan<byte> source, Span<byte> destination)
    {
        if (source.Length != destination.Length)
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.ShapeMismatch,
                "Raw tensor copies must preserve byte length.");
        }
        source.CopyTo(destination);
    }

    private static void CopyRawElement(
        ReadOnlySpan<byte> source,
        long sourceIndex,
        Span<byte> destination,
        long destinationIndex,
        DataType dataType)
    {
        int elementSize = DataTypeInfo.ElementSize(dataType);
        int sourceOffset = checked((int)(sourceIndex * elementSize));
        int destinationOffset = checked((int)(destinationIndex * elementSize));
        source.Slice(sourceOffset, elementSize)
            .CopyTo(destination.Slice(destinationOffset, elementSize));
    }

    private static long[] ResolveInferredShape(long elementCount, IReadOnlyList<long> requested)
    {
        if (requested.Count == 0 && elementCount != 1)
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.ShapeMismatch,
                "A scalar reshape requires exactly one element.");
        }

        long knownElements = 1;
        int inferredIndex = -1;
        long[] shape = requested.ToArray();
        for (int index = 0; index < shape.Length; index++)
        {
            if (shape[index] == -1)
            {
                if (inferredIndex >= 0)
                {
                    throw new ArgumentException("Only one reshape dimension may be -1.", nameof(requested));
                }
                inferredIndex = index;
            }
            else if (shape[index] < 0)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(requested),
                    "Reshape dimensions must be non-negative or -1.");
            }
            else
            {
                knownElements = checked(knownElements * shape[index]);
            }
        }

        if (inferredIndex >= 0)
        {
            if (knownElements == 0 || elementCount % knownElements != 0)
            {
                throw new PowerTorchException(
                    PowerTorchErrorCategory.ShapeMismatch,
                    "The inferred reshape dimension cannot be resolved.");
            }
            shape[inferredIndex] = elementCount / knownElements;
        }
        else if (knownElements != elementCount)
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.ShapeMismatch,
                $"Reshape requests {knownElements} elements but the tensor has {elementCount}.");
        }
        return shape;
    }

    private static int NormalizeDimension(int dimension, int rank)
    {
        int normalized = dimension < 0 ? rank + dimension : dimension;
        if (normalized < 0 || normalized >= rank)
        {
            throw new ArgumentOutOfRangeException(
                nameof(dimension),
                dimension,
                $"Dimension must be in -{rank}..{rank - 1}.");
        }
        return normalized;
    }

    private static void UnravelIndex(
        long flatIndex,
        IReadOnlyList<long> shape,
        Span<long> coordinates)
    {
        for (int dimension = shape.Count - 1; dimension >= 0; dimension--)
        {
            long size = shape[dimension];
            coordinates[dimension] = flatIndex % size;
            flatIndex /= size;
        }
    }

    private static Tensor MatmulFloating(
        Tensor left,
        Tensor right,
        Func<Tensor, float[]> read,
        Action<Tensor, float[]> write)
    {
        MatmulPlan plan = CreateMatmulPlan(left, right);
        float[] values = ComputeMatmul(
            read(left),
            read(right),
            plan,
            static (x, y) => x * y,
            static (sum, value) => sum + value);
        var output = TensorFactory.Empty(left.DataType, plan.OutputShape);
        write(output, values);
        return output;
    }

    private static Tensor MatmulDouble(Tensor left, Tensor right)
    {
        MatmulPlan plan = CreateMatmulPlan(left, right);
        double[] values = ComputeMatmul(
            left.AsSpan<double>().ToArray(),
            right.AsSpan<double>().ToArray(),
            plan,
            static (x, y) => x * y,
            static (sum, value) => sum + value);
        var output = TensorFactory.Empty(left.DataType, plan.OutputShape);
        values.CopyTo(output.AsSpan<double>());
        return output;
    }

    private static T[] ComputeMatmul<T>(
        T[] left,
        T[] right,
        MatmulPlan plan,
        Func<T, T, T> multiply,
        Func<T, T, T> add)
        where T : struct, System.Numerics.INumber<T>
    {
        int batchCount = checked((int)plan.BatchElements);
        long valuesPerOutputBatch = plan.OutputRows * plan.OutputColumns;
        T[] output = new T[checked((int)(batchCount * valuesPerOutputBatch))];
        long[] coordinates = new long[plan.OutputBatchShape.Length];
        for (int batch = 0; batch < batchCount; batch++)
        {
            UnravelIndex(batch, plan.OutputBatchShape, coordinates);
            long leftBatchOffset = BroadcastBatchOffset(coordinates, plan.LeftBatchShape);
            long rightBatchOffset = BroadcastBatchOffset(coordinates, plan.RightBatchShape);
            for (long row = 0; row < plan.OutputRows; row++)
            {
                for (long column = 0; column < plan.OutputColumns; column++)
                {
                    T sum = T.Zero;
                    for (long inner = 0; inner < plan.Inner; inner++)
                    {
                        long leftOffset = plan.LeftIsVector
                            ? inner
                            : (leftBatchOffset * plan.MatrixElementsLeft) +
                                (row * plan.Inner) + inner;
                        long rightOffset = plan.RightIsVector
                            ? (rightBatchOffset * plan.MatrixElementsRight) + inner
                            : (rightBatchOffset * plan.MatrixElementsRight) +
                                (inner * plan.OutputColumns) + column;
                        sum = add(sum, multiply(left[checked((int)leftOffset)], right[checked((int)rightOffset)]));
                    }
                    long outputOffset = plan.BothVectors
                        ? 0
                        : (batch * valuesPerOutputBatch) +
                            (row * plan.OutputColumns) + column;
                    output[checked((int)outputOffset)] = sum;
                }
            }
        }
        return output;
    }

    private static long BroadcastBatchOffset(
        ReadOnlySpan<long> outputCoordinates,
        long[] batchShape)
    {
        int alignment = outputCoordinates.Length - batchShape.Length;
        long index = 0;
        for (int dimension = 0; dimension < batchShape.Length; dimension++)
        {
            long coordinate = batchShape[dimension] == 1
                ? 0
                : outputCoordinates[alignment + dimension];
            index = checked((index * batchShape[dimension]) + coordinate);
        }
        return index;
    }

    private static MatmulPlan CreateMatmulPlan(Tensor left, Tensor right)
    {
        if (left.Shape.Count == 0 || right.Shape.Count == 0)
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.ShapeMismatch,
                "Matmul inputs must have rank one or greater.");
        }
        bool leftVector = left.Shape.Count == 1;
        bool rightVector = right.Shape.Count == 1;
        bool bothVectors = leftVector && rightVector;
        long rows = leftVector ? 1 : left.Shape[^2];
        long inner = left.Shape[^1];
        long rightRows = rightVector ? right.Shape[0] : right.Shape[^2];
        long columns = rightVector ? 1 : right.Shape[^1];
        if (inner != rightRows)
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.ShapeMismatch,
                $"Matmul inner dimensions must match; left={inner}, right={rightRows}.");
        }

        long[] leftBatch = leftVector ? [] : left.Shape.Take(left.Shape.Count - 2).ToArray();
        long[] rightBatch = rightVector ? [] : right.Shape.Take(right.Shape.Count - 2).ToArray();
        long[] outputBatch = BroadcastShape(leftBatch, rightBatch);
        long batchElements = Product(outputBatch);
        var outputShape = outputBatch.ToList();
        if (!leftVector && !rightVector)
        {
            outputShape.Add(rows);
            outputShape.Add(columns);
        }
        else if (leftVector && !rightVector)
        {
            outputShape.Add(columns);
        }
        else if (!leftVector && rightVector)
        {
            outputShape.Add(rows);
        }

        return new MatmulPlan(
            outputShape.ToArray(),
            outputBatch,
            leftBatch,
            rightBatch,
            batchElements,
            rows,
            columns,
            inner,
            leftVector,
            rightVector,
            bothVectors);
    }

    private static long[] BroadcastShape(long[] left, long[] right)
    {
        int rank = Math.Max(left.Length, right.Length);
        long[] result = new long[rank];
        for (int index = 0; index < rank; index++)
        {
            int leftIndex = index - (rank - left.Length);
            int rightIndex = index - (rank - right.Length);
            long leftDimension = leftIndex < 0 ? 1 : left[leftIndex];
            long rightDimension = rightIndex < 0 ? 1 : right[rightIndex];
            if (leftDimension != rightDimension &&
                leftDimension != 1 &&
                rightDimension != 1)
            {
                throw new PowerTorchException(
                    PowerTorchErrorCategory.ShapeMismatch,
                    "Matmul batch dimensions are not broadcastable.");
            }
            result[index] = Math.Max(leftDimension, rightDimension);
        }
        return result;
    }

    private static Tensor SoftmaxFloat(Tensor input, int dimension)
    {
        float[] values = input.AsSpan<float>().ToArray();
        ApplySoftmax(values, input.Shape, dimension);
        return TensorFactory.FromArray(values, input.Shape);
    }

    private static Tensor SoftmaxBFloat16(Tensor input, int dimension)
    {
        float[] values = input.AsSpan<BFloat16>().ToArray()
            .Select(value => (float)value)
            .ToArray();
        ApplySoftmax(values, input.Shape, dimension);
        return TensorFactory.FromArray(
            values.Select(value => (BFloat16)value).ToArray(),
            input.Shape);
    }

    private static Tensor SoftmaxDouble(Tensor input, int dimension)
    {
        double[] values = input.AsSpan<double>().ToArray();
        long size = input.Shape[dimension];
        if (size == 0)
        {
            return TensorFactory.FromArray(values, input.Shape);
        }
        long inner = Product(input.Shape.Skip(dimension + 1).ToArray());
        long outer = input.ElementCount / (input.Shape[dimension] * inner);
        for (long outerIndex = 0; outerIndex < outer; outerIndex++)
        {
            for (long innerIndex = 0; innerIndex < inner; innerIndex++)
            {
                double maximum = double.NegativeInfinity;
                for (long index = 0; index < size; index++)
                {
                    maximum = Math.Max(
                        maximum,
                        values[checked((int)((outerIndex * size * inner) + (index * inner) + innerIndex))]);
                }
                double denominator = 0;
                for (long index = 0; index < size; index++)
                {
                    int offset = checked((int)(
                        (outerIndex * size * inner) + (index * inner) + innerIndex));
                    values[offset] = Math.Exp(values[offset] - maximum);
                    denominator += values[offset];
                }
                for (long index = 0; index < size; index++)
                {
                    int offset = checked((int)(
                        (outerIndex * size * inner) + (index * inner) + innerIndex));
                    values[offset] /= denominator;
                }
            }
        }
        return TensorFactory.FromArray(values, input.Shape);
    }

    private static void ApplySoftmax(
        float[] values,
        IReadOnlyList<long> shape,
        int dimension)
    {
        long size = shape[dimension];
        if (size == 0)
        {
            return;
        }
        long inner = Product(shape.Skip(dimension + 1).ToArray());
        long outer = values.LongLength / (shape[dimension] * inner);
        for (long outerIndex = 0; outerIndex < outer; outerIndex++)
        {
            for (long innerIndex = 0; innerIndex < inner; innerIndex++)
            {
                float maximum = float.NegativeInfinity;
                for (long index = 0; index < size; index++)
                {
                    maximum = MathF.Max(
                        maximum,
                        values[checked((int)((outerIndex * size * inner) + (index * inner) + innerIndex))]);
                }
                float denominator = 0;
                for (long index = 0; index < size; index++)
                {
                    int offset = checked((int)(
                        (outerIndex * size * inner) + (index * inner) + innerIndex));
                    values[offset] = MathF.Exp(values[offset] - maximum);
                    denominator += values[offset];
                }
                for (long index = 0; index < size; index++)
                {
                    int offset = checked((int)(
                        (outerIndex * size * inner) + (index * inner) + innerIndex));
                    values[offset] /= denominator;
                }
            }
        }
    }

    private static void ValidateNormalization(
        Tensor input,
        IReadOnlyList<long> normalizedShape,
        Tensor? weight,
        Tensor? bias,
        double epsilon)
    {
        if (normalizedShape.Count == 0 || normalizedShape.Any(dimension => dimension <= 0))
        {
            throw new ArgumentException(
                "Normalized shape must contain positive dimensions.",
                nameof(normalizedShape));
        }
        if (normalizedShape.Count > input.Shape.Count ||
            !input.Shape.Skip(input.Shape.Count - normalizedShape.Count).SequenceEqual(normalizedShape))
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.ShapeMismatch,
                "Normalized shape must match the input tensor's trailing dimensions.");
        }
        ValidateNormalizationParameter(input, normalizedShape, weight, nameof(weight));
        ValidateNormalizationParameter(input, normalizedShape, bias, nameof(bias));
        if (!double.IsFinite(epsilon) || epsilon < 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(epsilon),
                "Epsilon must be finite and non-negative.");
        }
        if (input.DataType is not (DataType.BFloat16 or DataType.Float32 or DataType.Float64))
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.Unsupported,
                $"Normalization does not support dtype '{input.DataType}'.");
        }
    }

    private static void ValidateNormalizationParameter(
        Tensor input,
        IReadOnlyList<long> normalizedShape,
        Tensor? parameter,
        string name)
    {
        if (parameter is not null &&
            (parameter.DataType != input.DataType ||
                parameter.Device != input.Device ||
                !parameter.Shape.SequenceEqual(normalizedShape)))
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.ShapeMismatch,
                $"{name} must match the input dtype, device, and normalized shape.");
        }
    }

    private static Tensor Normalize(
        Tensor input,
        IReadOnlyList<long> normalizedShape,
        Tensor? weight,
        Tensor? bias,
        double epsilon,
        bool rmsOnly)
    {
        return input.DataType == DataType.Float64
            ? NormalizeDouble(input, normalizedShape, weight, bias, epsilon, rmsOnly)
            : NormalizeFloat(input, normalizedShape, weight, bias, epsilon, rmsOnly);
    }

    private static Tensor NormalizeFloat(
        Tensor input,
        IReadOnlyList<long> normalizedShape,
        Tensor? weight,
        Tensor? bias,
        double epsilon,
        bool rmsOnly)
    {
        float[] values = input.DataType == DataType.BFloat16
            ? input.AsSpan<BFloat16>().ToArray().Select(value => (float)value).ToArray()
            : input.AsSpan<float>().ToArray();
        float[]? weightValues = weight is null
            ? null
            : weight.DataType == DataType.BFloat16
                ? weight.AsSpan<BFloat16>().ToArray().Select(value => (float)value).ToArray()
                : weight.AsSpan<float>().ToArray();
        float[]? biasValues = bias is null
            ? null
            : bias.DataType == DataType.BFloat16
                ? bias.AsSpan<BFloat16>().ToArray().Select(value => (float)value).ToArray()
                : bias.AsSpan<float>().ToArray();

        ApplyNormalization(
            values,
            weightValues,
            biasValues,
            Product(normalizedShape),
            epsilon,
            rmsOnly,
            MathF.Sqrt);
        if (input.DataType == DataType.BFloat16)
        {
            return TensorFactory.FromArray(
                values.Select(value => (BFloat16)value).ToArray(),
                input.Shape);
        }
        return TensorFactory.FromArray(values, input.Shape);
    }

    private static Tensor NormalizeDouble(
        Tensor input,
        IReadOnlyList<long> normalizedShape,
        Tensor? weight,
        Tensor? bias,
        double epsilon,
        bool rmsOnly)
    {
        double[] values = input.AsSpan<double>().ToArray();
        ApplyNormalization(
            values,
            weight?.AsSpan<double>().ToArray(),
            bias?.AsSpan<double>().ToArray(),
            Product(normalizedShape),
            epsilon,
            rmsOnly,
            Math.Sqrt);
        return TensorFactory.FromArray(values, input.Shape);
    }

    private static void ApplyNormalization<T>(
        T[] values,
        T[]? weight,
        T[]? bias,
        long normalizedElements,
        double epsilon,
        bool rmsOnly,
        Func<T, T> squareRoot)
        where T : struct, System.Numerics.IFloatingPoint<T>
    {
        int width = checked((int)normalizedElements);
        T count = T.CreateChecked(width);
        T epsilonValue = T.CreateChecked(epsilon);
        for (int rowStart = 0; rowStart < values.Length; rowStart += width)
        {
            T mean = T.Zero;
            if (!rmsOnly)
            {
                for (int index = 0; index < width; index++)
                {
                    mean += values[rowStart + index];
                }
                mean /= count;
            }

            T squared = T.Zero;
            for (int index = 0; index < width; index++)
            {
                T centered = values[rowStart + index] - mean;
                squared += centered * centered;
            }
            T scale = T.One / squareRoot((squared / count) + epsilonValue);
            for (int index = 0; index < width; index++)
            {
                T value = (values[rowStart + index] - mean) * scale;
                if (weight is not null)
                {
                    value *= weight[index];
                }
                if (bias is not null)
                {
                    value += bias[index];
                }
                values[rowStart + index] = value;
            }
        }
    }

    private static long Product(IEnumerable<long> dimensions)
    {
        long product = 1;
        foreach (long dimension in dimensions)
        {
            product = checked(product * dimension);
        }
        return product;
    }

    private static PowerTorchException Unsupported(DataType dataType) => new(
        PowerTorchErrorCategory.Unsupported,
        $"CPU v0 backend does not implement dtype '{dataType}'.");

    private static void AddSpans<T>(ReadOnlySpan<T> left, ReadOnlySpan<T> right, Span<T> output)
        where T : struct, System.Numerics.IAdditionOperators<T, T, T>
    {
        for (int index = 0; index < left.Length; index++)
        {
            output[index] = left[index] + right[index];
        }
    }

    private static void MultiplySpans<T>(ReadOnlySpan<T> left, ReadOnlySpan<T> right, Span<T> output)
        where T : struct, System.Numerics.IMultiplyOperators<T, T, T>
    {
        for (int index = 0; index < left.Length; index++)
        {
            output[index] = left[index] * right[index];
        }
    }

    private static void SubtractSpans<T>(ReadOnlySpan<T> left, ReadOnlySpan<T> right, Span<T> output)
        where T : struct, System.Numerics.ISubtractionOperators<T, T, T>
    {
        for (int index = 0; index < left.Length; index++)
        {
            output[index] = left[index] - right[index];
        }
    }

    private static void DivideSpans<T>(ReadOnlySpan<T> left, ReadOnlySpan<T> right, Span<T> output)
        where T : struct, System.Numerics.IDivisionOperators<T, T, T>
    {
        for (int index = 0; index < left.Length; index++)
        {
            output[index] = left[index] / right[index];
        }
    }

    private static Tensor Reduce(Tensor input, bool average)
    {
        Tensor output = TensorFactory.Empty(input.DataType);
        switch (input.DataType)
        {
            case DataType.Float32:
                output.AsSpan<float>()[0] = Reduce<float>(input.AsSpan<float>(), average);
                break;
            case DataType.Float64:
                output.AsSpan<double>()[0] = Reduce<double>(input.AsSpan<double>(), average);
                break;
            case DataType.BFloat16:
                var sum = 0.0f;
                foreach (BFloat16 value in input.AsSpan<BFloat16>())
                {
                    sum += value;
                }
                output.AsSpan<BFloat16>()[0] =
                    (BFloat16)(average ? sum / input.ElementCount : sum);
                break;
            case DataType.Int32:
                if (average)
                {
                    throw new PowerTorchException(
                        PowerTorchErrorCategory.Unsupported,
                        "Integer mean is unsupported.");
                }
                int intSum = 0;
                foreach (int value in input.AsSpan<int>())
                {
                    intSum = checked(intSum + value);
                }
                output.AsSpan<int>()[0] = intSum;
                break;
            case DataType.Int64:
                if (average)
                {
                    throw new PowerTorchException(
                        PowerTorchErrorCategory.Unsupported,
                        "Integer mean is unsupported.");
                }
                long longSum = 0;
                foreach (long value in input.AsSpan<long>())
                {
                    longSum = checked(longSum + value);
                }
                output.AsSpan<long>()[0] = longSum;
                break;
            default:
                throw Unsupported(input.DataType);
        }
        return output;
    }

    private static T Reduce<T>(ReadOnlySpan<T> input, bool average)
        where T : struct, System.Numerics.INumber<T>
    {
        T sum = T.Zero;
        for (int index = 0; index < input.Length; index++)
        {
            sum += input[index];
        }
        if (average)
        {
            sum /= T.CreateChecked(input.Length);
        }
        return sum;
    }

    private sealed record MatmulPlan(
        long[] OutputShape,
        long[] OutputBatchShape,
        long[] LeftBatchShape,
        long[] RightBatchShape,
        long BatchElements,
        long OutputRows,
        long OutputColumns,
        long Inner,
        bool LeftIsVector,
        bool RightIsVector,
        bool BothVectors)
    {
        public long MatrixElementsLeft => OutputRows * Inner;

        public long MatrixElementsRight => RightIsVector
            ? Inner
            : Inner * OutputColumns;
    }
}
