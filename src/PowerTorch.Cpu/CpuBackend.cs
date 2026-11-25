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
            ["tensor.embedding"] = true,
            ["tensor.rotary"] = true,
            ["tensor.causal_attention"] = true,
            ["tensor.conv1d"] = true,
            ["tensor.conv_transpose1d"] = true,
            ["tensor.concat"] = true,
            ["tensor.kv_cache_append"] = true,
            ["tensor.arg_max"] = true,
            ["tensor.random_normal"] = true,
            ["tensor.trigonometric"] = true,
            ["tensor.host_residency"] = true,
            ["tensor_residency"] = true,
            ["model.safe_state_load"] = true,
            ["model.sharded_checkpoint"] = true,
            ["device.transfer"] = false,
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

    public Tensor Embedding(Tensor weight, Tensor indices)
    {
        if (weight.Shape.Count != 2)
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.ShapeMismatch,
                "Embedding weights must have [vocabulary, width] shape.");
        }
        if (weight.DataType is not (DataType.BFloat16 or DataType.Float32 or DataType.Float64))
        {
            throw Unsupported(weight.DataType);
        }

        long[] outputShape = [.. indices.Shape, weight.Shape[1]];
        long rows = weight.Shape[0];
        long width = weight.Shape[1];
        int elementSize = DataTypeInfo.ElementSize(weight.DataType);
        long[] rawIndices = ReadInt64Indices(indices);
        for (int index = 0; index < rawIndices.Length; index++)
        {
            long row = rawIndices[index];
            if (row < 0 || row >= rows)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(indices),
                    $"Embedding index {row} is outside 0..{rows - 1}.");
            }
        }

        Tensor output = TensorFactory.Empty(weight.DataType, outputShape);
        for (int index = 0; index < rawIndices.Length; index++)
        {
            long row = rawIndices[index];
            int sourceOffset = checked((int)(row * width * elementSize));
            int destinationOffset = checked((int)(index * width * elementSize));
            weight.RawBytes.Slice(sourceOffset, checked((int)(width * elementSize)))
                .CopyTo(output.RawBytes.Slice(destinationOffset, checked((int)(width * elementSize))));
        }
        return output;
    }

    public Tensor Conv1D(
        Tensor input,
        Tensor weight,
        Tensor? bias,
        int stride,
        int padding,
        int dilation,
        int groups,
        bool causal)
    {
        ValidateConv1D(input, weight, bias, stride, padding, dilation, groups, causal);
        long batch = input.Shape[0];
        long inputChannels = input.Shape[1];
        long inputLength = input.Shape[2];
        long outputChannels = weight.Shape[0];
        long kernelLength = weight.Shape[2];
        long leftPadding = causal ? (kernelLength - 1) * dilation : padding;
        long rightPadding = causal ? 0 : padding;
        long outputLength = checked(
            (inputLength + leftPadding + rightPadding - (dilation * (kernelLength - 1)) - 1) /
            stride + 1);
        long[] outputShape = [batch, outputChannels, outputLength];
        float[] inputValues = ReadFloatingValues(input);
        float[] weightValues = ReadFloatingValues(weight);
        float[]? biasValues = bias is null ? null : ReadFloatingValues(bias);
        float[] outputValues = new float[checked((int)(batch * outputChannels * outputLength))];
        long channelsPerGroup = inputChannels / groups;
        long outputsPerGroup = outputChannels / groups;

        for (long batchIndex = 0; batchIndex < batch; batchIndex++)
        {
            for (long outputChannel = 0; outputChannel < outputChannels; outputChannel++)
            {
                long group = outputChannel / outputsPerGroup;
                for (long outputPosition = 0; outputPosition < outputLength; outputPosition++)
                {
                    float sum = biasValues is null ? 0 : biasValues[checked((int)outputChannel)];
                    for (long inputChannelOffset = 0; inputChannelOffset < channelsPerGroup; inputChannelOffset++)
                    {
                        long inputChannel = (group * channelsPerGroup) + inputChannelOffset;
                        for (long kernelIndex = 0; kernelIndex < kernelLength; kernelIndex++)
                        {
                            long inputPosition = (outputPosition * stride) -
                                leftPadding + (kernelIndex * dilation);
                            if (inputPosition < 0 || inputPosition >= inputLength)
                            {
                                continue;
                            }
                            long inputOffset =
                                (batchIndex * inputChannels * inputLength) +
                                (inputChannel * inputLength) + inputPosition;
                            long weightOffset =
                                (outputChannel * channelsPerGroup * kernelLength) +
                                (inputChannelOffset * kernelLength) + kernelIndex;
                            sum += inputValues[checked((int)inputOffset)] *
                                weightValues[checked((int)weightOffset)];
                        }
                    }
                    long outputOffset =
                        (batchIndex * outputChannels * outputLength) +
                        (outputChannel * outputLength) + outputPosition;
                    outputValues[checked((int)outputOffset)] = sum;
                }
            }
        }
        return FromFloatingValues(input.DataType, outputShape, outputValues);
    }

    public Tensor ConvTranspose1D(
        Tensor input,
        Tensor weight,
        Tensor? bias,
        int stride,
        int padding,
        int outputPadding,
        int dilation,
        int groups)
    {
        ValidateConvTranspose1D(
            input,
            weight,
            bias,
            stride,
            padding,
            outputPadding,
            dilation,
            groups);
        long batch = input.Shape[0];
        long inputChannels = input.Shape[1];
        long inputLength = input.Shape[2];
        long outputChannels = weight.Shape[1] * groups;
        long kernelLength = weight.Shape[2];
        long outputLength = checked(
            ((inputLength - 1) * stride) -
            (2L * padding) +
            dilation * (kernelLength - 1) +
            outputPadding + 1);
        long[] outputShape = [batch, outputChannels, outputLength];
        float[] inputValues = ReadFloatingValues(input);
        float[] weightValues = ReadFloatingValues(weight);
        float[]? biasValues = bias is null ? null : ReadFloatingValues(bias);
        var outputValues = new float[checked((int)(batch * outputChannels * outputLength))];
        long inputsPerGroup = inputChannels / groups;
        long outputsPerGroup = outputChannels / groups;
        for (long batchIndex = 0; batchIndex < batch; batchIndex++)
        {
            for (long inputChannel = 0; inputChannel < inputChannels; inputChannel++)
            {
                long group = inputChannel / inputsPerGroup;
                for (long inputPosition = 0; inputPosition < inputLength; inputPosition++)
                {
                    float inputValue = inputValues[checked(
                        (int)(((batchIndex * inputChannels * inputLength) +
                            (inputChannel * inputLength) + inputPosition)))];
                    for (long outputChannelOffset = 0;
                         outputChannelOffset < outputsPerGroup;
                         outputChannelOffset++)
                    {
                        long outputChannel = (group * outputsPerGroup) + outputChannelOffset;
                        for (long kernelIndex = 0; kernelIndex < kernelLength; kernelIndex++)
                        {
                            long outputPosition =
                                (inputPosition * stride) -
                                padding +
                                (kernelIndex * dilation);
                            if (outputPosition < 0 || outputPosition >= outputLength)
                            {
                                continue;
                            }
                            long weightOffset =
                                (inputChannel * outputsPerGroup * kernelLength) +
                                (outputChannelOffset * kernelLength) +
                                kernelIndex;
                            long outputOffset =
                                (batchIndex * outputChannels * outputLength) +
                                (outputChannel * outputLength) +
                                outputPosition;
                            outputValues[checked((int)outputOffset)] +=
                                inputValue * weightValues[checked((int)weightOffset)];
                        }
                    }
                }
            }
            if (biasValues is not null)
            {
                for (long outputChannel = 0; outputChannel < outputChannels; outputChannel++)
                {
                    long offset =
                        (batchIndex * outputChannels * outputLength) +
                        (outputChannel * outputLength);
                    for (long position = 0; position < outputLength; position++)
                    {
                        outputValues[checked((int)(offset + position))] +=
                            biasValues[checked((int)outputChannel)];
                    }
                }
            }
        }
        return FromFloatingValues(input.DataType, outputShape, outputValues);
    }

    public Tensor Rotary(
        Tensor input,
        Tensor positions,
        int sequenceDimension,
        int rotaryDimension,
        double baseFrequency,
        bool interleaved)
    {
        if (input.DataType is not (DataType.BFloat16 or DataType.Float32))
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.Unsupported,
                $"CPU rotary does not support dtype '{input.DataType}'.");
        }
        if (input.Shape.Count < 2)
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.ShapeMismatch,
                "Rotary input must have at least two dimensions.");
        }
        sequenceDimension = NormalizeDimension(sequenceDimension, input.Shape.Count);
        if (sequenceDimension == input.Shape.Count - 1)
        {
            throw new ArgumentException(
                "The rotary sequence dimension cannot be the feature dimension.",
                nameof(sequenceDimension));
        }
        long width = input.Shape[^1];
        rotaryDimension = rotaryDimension == 0 ? checked((int)width) : rotaryDimension;
        if (rotaryDimension <= 0 || rotaryDimension > width ||
            rotaryDimension % 2 != 0 ||
            !double.IsFinite(baseFrequency) || baseFrequency <= 0)
        {
            throw new ArgumentException(
                "Rotary dimension must be positive/even and base frequency finite/positive.");
        }
        long sequenceLength = input.Shape[sequenceDimension];
        ValidatePositions(positions, sequenceLength);

        float[] values = ReadFloatingValues(input);
        long[] indices = ReadInt64Indices(positions);
        long trailingElements = Product(input.Shape.Skip(sequenceDimension + 1).ToArray());
        long groups = input.ElementCount / (sequenceLength * trailingElements);
        if (indices.Length != sequenceLength && groups % (indices.Length / sequenceLength) != 0)
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.ShapeMismatch,
                "Position rows must broadcast across rotary input groups.");
        }
        long positionRows = indices.Length / sequenceLength;
        long groupsPerPositionRow = positionRows == 1 ? groups : groups / positionRows;
        int half = rotaryDimension / 2;
        float inverseWidth = 1.0f / rotaryDimension;

        for (long group = 0; group < groups; group++)
        {
            long row = positionRows == 1 ? 0 : group / groupsPerPositionRow;
            for (long sequence = 0; sequence < sequenceLength; sequence++)
            {
                long positionIndex = checked((int)((row * sequenceLength) + sequence));
                long sequenceOffset =
                    ((group * sequenceLength) + sequence) * trailingElements;
                for (long trailing = 0; trailing < trailingElements / width; trailing++)
                {
                    long featureOffset = sequenceOffset + (trailing * width);
                    ApplyRotaryVector(
                        values,
                        checked((int)featureOffset),
                        checked((int)width),
                        half,
                        indices[checked((int)positionIndex)],
                        baseFrequency,
                        inverseWidth,
                        interleaved);
                }
            }
        }
        return FromFloatingValues(input.DataType, input.Shape.ToArray(), values);
    }

    public Tensor CausalAttention(
        Tensor query,
        Tensor key,
        Tensor value,
        double? scale)
    {
        ValidateAttention(query, key, value, scale);
        long batch = query.Shape[0];
        long queryHeads = query.Shape[1];
        long querySequence = query.Shape[2];
        long width = query.Shape[3];
        long keyHeads = key.Shape[1];
        long keySequence = key.Shape[2];
        float[] queryValues = ReadFloatingValues(query);
        float[] keyValues = ReadFloatingValues(key);
        float[] valueValues = ReadFloatingValues(value);
        float[] outputValues = new float[checked((int)query.ElementCount)];
        float scaleValue = (float)(scale ?? (1.0 / Math.Sqrt(width)));
        if (!float.IsFinite(scaleValue) || scaleValue <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(scale), "Attention scale must be finite and positive.");
        }
        long headsPerKey = queryHeads / keyHeads;

        for (long batchIndex = 0; batchIndex < batch; batchIndex++)
        {
            for (long queryHead = 0; queryHead < queryHeads; queryHead++)
            {
                long keyHead = queryHead / headsPerKey;
                for (long queryPosition = 0; queryPosition < querySequence; queryPosition++)
                {
                    long queryOffset = AttentionOffset(batchIndex, queryHead, queryPosition, queryHeads, querySequence, width);
                    float[] probabilities = new float[checked((int)keySequence)];
                    float maximum = float.NegativeInfinity;
                    long causalLimit = Math.Min(keySequence, queryPosition + keySequence - querySequence + 1);
                    for (long keyPosition = 0; keyPosition < causalLimit; keyPosition++)
                    {
                        long keyOffset = AttentionOffset(batchIndex, keyHead, keyPosition, keyHeads, keySequence, width);
                        float score = 0;
                        for (long feature = 0; feature < width; feature++)
                        {
                            score += queryValues[checked((int)(queryOffset + feature))] *
                                keyValues[checked((int)(keyOffset + feature))];
                        }
                        score *= scaleValue;
                        probabilities[checked((int)keyPosition)] = score;
                        maximum = MathF.Max(maximum, score);
                    }
                    float denominator = 0;
                    for (long keyPosition = 0; keyPosition < causalLimit; keyPosition++)
                    {
                        float probability = MathF.Exp(
                            probabilities[checked((int)keyPosition)] - maximum);
                        probabilities[checked((int)keyPosition)] = probability;
                        denominator += probability;
                    }
                    for (long feature = 0; feature < width; feature++)
                    {
                        float sum = 0;
                        for (long keyPosition = 0; keyPosition < causalLimit; keyPosition++)
                        {
                            long valueOffset = AttentionOffset(batchIndex, keyHead, keyPosition, keyHeads, keySequence, width);
                            sum += (probabilities[checked((int)keyPosition)] / denominator) *
                                valueValues[checked((int)(valueOffset + feature))];
                        }
                        outputValues[checked((int)(queryOffset + feature))] = sum;
                    }
                }
            }
        }
        return FromFloatingValues(query.DataType, query.Shape.ToArray(), outputValues);
    }

    public Tensor Concat(IReadOnlyList<Tensor> inputs, int dimension)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        if (inputs.Count == 0)
        {
            throw new ArgumentException("Concat requires at least one tensor.", nameof(inputs));
        }
        int rank = inputs[0].Shape.Count;
        dimension = NormalizeDimension(dimension, rank);
        long[] outputShape = inputs[0].Shape.ToArray();
        outputShape[dimension] = 0;
        foreach (Tensor input in inputs)
        {
            if (input.Shape.Count != rank || input.DataType != inputs[0].DataType)
            {
                throw new PowerTorchException(
                    PowerTorchErrorCategory.ShapeMismatch,
                    "Concat inputs must have matching rank and dtype.");
            }
            for (int current = 0; current < rank; current++)
            {
                if (current != dimension && input.Shape[current] != inputs[0].Shape[current])
                {
                    throw new PowerTorchException(
                        PowerTorchErrorCategory.ShapeMismatch,
                        "Concat inputs may differ only on the selected dimension.");
                }
            }
            outputShape[dimension] = checked(outputShape[dimension] + input.Shape[dimension]);
        }

        Tensor output = TensorFactory.Empty(inputs[0].DataType, outputShape);
        int elementSize = DataTypeInfo.ElementSize(inputs[0].DataType);
        long outer = Product(outputShape.Take(dimension));
        long inner = Product(outputShape.Skip(dimension + 1));
        long outputDimension = outputShape[dimension];
        long outputDimensionOffset = 0;
        foreach (Tensor input in inputs)
        {
            long inputDimension = input.Shape[dimension];
            for (long outerIndex = 0; outerIndex < outer; outerIndex++)
            {
                long sourceElementOffset =
                    (outerIndex * inputDimension) * inner;
                long destinationElementOffset =
                    ((outerIndex * outputDimension) + outputDimensionOffset) * inner;
                int byteCount = checked((int)(inputDimension * inner * elementSize));
                input.RawBytes.Slice(
                        checked((int)(sourceElementOffset * elementSize)),
                        byteCount)
                    .CopyTo(output.RawBytes.Slice(
                        checked((int)(destinationElementOffset * elementSize)),
                        byteCount));
            }
            outputDimensionOffset += inputDimension;
        }
        return output;
    }

    public Tensor KvCacheAppend(Tensor cache, Tensor update)
    {
        if (cache.Shape.Count < 2 ||
            update.Shape.Count != cache.Shape.Count ||
            cache.DataType != update.DataType)
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.ShapeMismatch,
                "KV cache and update must have equal rank and dtype, with rank at least two.");
        }
        return Concat([cache, update], cache.Shape.Count - 2);
    }

    public Tensor ArgMax(Tensor input, int dimension)
    {
        dimension = NormalizeDimension(dimension, input.Shape.Count);
        if (input.Shape[dimension] == 0)
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.ShapeMismatch,
                "ArgMax cannot reduce an empty dimension.");
        }
        long[] outputShape = input.Shape
            .Where((_, index) => index != dimension)
            .ToArray();
        Tensor output = TensorFactory.Empty(DataType.Int64, outputShape);
        var destination = output.AsSpan<long>();
        int outputIndex = 0;
        long inner = Product(input.Shape.Skip(dimension + 1));
        long outer = Product(input.Shape.Take(dimension));
        long width = input.Shape[dimension];
        for (long outerIndex = 0; outerIndex < outer; outerIndex++)
        {
            for (long innerIndex = 0; innerIndex < inner; innerIndex++)
            {
                double best = double.NegativeInfinity;
                long bestIndex = 0;
                for (long index = 0; index < width; index++)
                {
                    long flat = (outerIndex * width * inner) + (index * inner) + innerIndex;
                    double value = ReadNumericElement(input, flat);
                    if (value > best)
                    {
                        best = value;
                        bestIndex = index;
                    }
                }
                destination[outputIndex++] = bestIndex;
            }
        }
        return output;
    }

    public Tensor RandomNormal(
        DataType dataType,
        IReadOnlyList<long> shape,
        double mean,
        double standardDeviation,
        int? seed)
    {
        ArgumentNullException.ThrowIfNull(shape);
        if (!double.IsFinite(mean) ||
            !double.IsFinite(standardDeviation) ||
            standardDeviation < 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(standardDeviation),
                "Normal mean and standard deviation must be finite, with non-negative deviation.");
        }
        if (dataType is not (DataType.BFloat16 or DataType.Float32 or DataType.Float64))
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.Unsupported,
                $"RandomNormal requires a floating dtype; received '{dataType}'.");
        }
        Tensor output = TensorFactory.Empty(dataType, shape.ToArray());
        var random = seed is null ? Random.Shared : new Random(seed.Value);
        bool hasSpare = false;
        double spare = 0;
        double NextNormal()
        {
            if (hasSpare)
            {
                hasSpare = false;
                return spare;
            }
            double first;
            double second;
            double radiusSquared;
            do
            {
                first = (2.0 * random.NextDouble()) - 1.0;
                second = (2.0 * random.NextDouble()) - 1.0;
                radiusSquared = (first * first) + (second * second);
            }
            while (radiusSquared <= double.Epsilon || radiusSquared >= 1.0);
            double multiplier = Math.Sqrt(
                -2.0 * Math.Log(radiusSquared) / radiusSquared);
            spare = second * multiplier;
            hasSpare = true;
            return first * multiplier;
        }

        switch (dataType)
        {
            case DataType.BFloat16:
                Span<BFloat16> bf16 = output.AsSpan<BFloat16>();
                for (int index = 0; index < bf16.Length; index++)
                {
                    bf16[index] = (BFloat16)((mean + (NextNormal() * standardDeviation)));
                }
                break;
            case DataType.Float32:
                Span<float> float32 = output.AsSpan<float>();
                for (int index = 0; index < float32.Length; index++)
                {
                    float32[index] = (float)(mean + (NextNormal() * standardDeviation));
                }
                break;
            case DataType.Float64:
                Span<double> float64 = output.AsSpan<double>();
                for (int index = 0; index < float64.Length; index++)
                {
                    float64[index] = mean + (NextNormal() * standardDeviation);
                }
                break;
        }
        return output;
    }

    public Tensor Sin(Tensor input) => Trigonometric(input, Math.Sin, MathF.Sin);

    public Tensor Cos(Tensor input) => Trigonometric(input, Math.Cos, MathF.Cos);

    public Tensor Transfer(Tensor input, Device target)
    {
        if (target.Kind != DeviceKind.Cpu || input.Device.Kind != DeviceKind.Cpu)
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.Unsupported,
                "The CPU backend supports only managed host-to-host copies; " +
                "persistent accelerator transfers are not implemented.");
        }
        if (target != Device.Cpu)
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.Backend,
                $"Invalid CPU target '{target}'.");
        }
        return input.Clone();
    }

    private static Tensor Trigonometric(
        Tensor input,
        Func<double, double> doubleOperation,
        Func<float, float> floatOperation)
    {
        Tensor output = TensorFactory.Empty(input.DataType, input.Shape.ToArray());
        switch (input.DataType)
        {
            case DataType.BFloat16:
                Span<BFloat16> bf16Input = input.AsSpan<BFloat16>();
                Span<BFloat16> bf16Output = output.AsSpan<BFloat16>();
                for (int index = 0; index < bf16Output.Length; index++)
                {
                    bf16Output[index] = (BFloat16)floatOperation(bf16Input[index]);
                }
                break;
            case DataType.Float32:
                Span<float> floatInput = input.AsSpan<float>();
                Span<float> floatOutput = output.AsSpan<float>();
                for (int index = 0; index < floatOutput.Length; index++)
                {
                    floatOutput[index] = floatOperation(floatInput[index]);
                }
                break;
            case DataType.Float64:
                Span<double> doubleInput = input.AsSpan<double>();
                Span<double> doubleOutput = output.AsSpan<double>();
                for (int index = 0; index < doubleOutput.Length; index++)
                {
                    doubleOutput[index] = doubleOperation(doubleInput[index]);
                }
                break;
            default:
                throw new PowerTorchException(
                    PowerTorchErrorCategory.Unsupported,
                    $"Trigonometric operations require a floating dtype; received '{input.DataType}'.");
        }
        return output;
    }

    private static double ReadNumericElement(Tensor tensor, long index)
    {
        int offset = checked((int)index);
        return tensor.DataType switch
        {
            DataType.BFloat16 => tensor.AsSpan<BFloat16>()[offset],
            DataType.Float32 => tensor.AsSpan<float>()[offset],
            DataType.Float64 => tensor.AsSpan<double>()[offset],
            DataType.Int32 => tensor.AsSpan<int>()[offset],
            DataType.Int64 => tensor.AsSpan<long>()[offset],
            _ => throw Unsupported(tensor.DataType),
        };
    }

    private static void ValidateConv1D(
        Tensor input,
        Tensor weight,
        Tensor? bias,
        int stride,
        int padding,
        int dilation,
        int groups,
        bool causal)
    {
        if (input.Shape.Count != 3 || weight.Shape.Count != 3)
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.ShapeMismatch,
                "Conv1D expects rank-3 input [batch, channels, length] and weights.");
        }
        if (input.DataType != weight.DataType ||
            input.DataType is not (DataType.BFloat16 or DataType.Float32))
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.DTypeMismatch,
                "Conv1D input and weight must have the same floating dtype.");
        }
        if (stride <= 0 || padding < 0 || dilation <= 0 || groups <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(stride),
                "Conv1D stride/dilation/groups must be positive and padding non-negative.");
        }
        if (causal && padding != 0)
        {
            throw new ArgumentException(
                "Causal Conv1D supplies left padding and requires padding=0.",
                nameof(padding));
        }
        if (input.Shape[1] % groups != 0 ||
            weight.Shape[0] % groups != 0 ||
            weight.Shape[1] != input.Shape[1] / groups)
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.ShapeMismatch,
                "Conv1D channels and groups are inconsistent.");
        }
        long effectiveKernel = (dilation * (weight.Shape[2] - 1)) + 1;
        if (weight.Shape[2] == 0 ||
            (!causal && input.Shape[2] + (2L * padding) < effectiveKernel) ||
            (causal && input.Shape[2] == 0))
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.ShapeMismatch,
                "Conv1D effective kernel exceeds the padded input.");
        }
        if (bias is not null &&
            (bias.DataType != input.DataType ||
                bias.Device != input.Device ||
                bias.Shape.Count != 1 ||
                bias.Shape[0] != weight.Shape[0]))
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.ShapeMismatch,
                "Conv1D bias must match output channels, dtype, and device.");
        }
    }

    private static void ValidateConvTranspose1D(
        Tensor input,
        Tensor weight,
        Tensor? bias,
        int stride,
        int padding,
        int outputPadding,
        int dilation,
        int groups)
    {
        if (input.Shape.Count != 3 || weight.Shape.Count != 3)
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.ShapeMismatch,
                "ConvTranspose1D expects rank-3 input and weight.");
        }
        if (input.DataType != weight.DataType ||
            input.DataType is not (DataType.BFloat16 or DataType.Float32) ||
            stride <= 0 ||
            padding < 0 ||
            outputPadding < 0 ||
            dilation <= 0 ||
            groups <= 0)
        {
            throw new ArgumentException(
                "ConvTranspose1D requires matching floating tensors and valid stride/padding/dilation/groups.");
        }
        if (input.Shape[1] % groups != 0 ||
            weight.Shape[0] != input.Shape[1] ||
            outputPadding >= stride ||
            weight.Shape[2] == 0)
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.ShapeMismatch,
                "ConvTranspose1D channels or output padding are inconsistent.");
        }
        if (bias is not null &&
            (bias.DataType != input.DataType ||
                bias.Device != input.Device ||
                bias.Shape.Count != 1 ||
                bias.Shape[0] != weight.Shape[1] * groups))
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.ShapeMismatch,
                "ConvTranspose1D bias must match output channels, dtype, and device.");
        }
    }

    private static void ValidatePositions(Tensor positions, long sequenceLength)
    {
        if (positions.Shape.Count is < 1 or > 2 ||
            positions.Shape[^1] != sequenceLength ||
            positions.ElementCount == 0 ||
            positions.DataType is not (DataType.Int32 or DataType.Int64))
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.ShapeMismatch,
                "Rotary positions must be [sequence] or [batch, sequence] integer tensors.");
        }
    }

    private static void ApplyRotaryVector(
        float[] values,
        int offset,
        int width,
        int half,
        long position,
        double baseFrequency,
        float inverseWidth,
        bool interleaved)
    {
        if (position < 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(position),
                "Rotary positions cannot be negative.");
        }
        for (int pair = 0; pair < half; pair++)
        {
            int firstIndex = interleaved ? pair * 2 : pair;
            int secondIndex = interleaved ? firstIndex + 1 : pair + half;
            double exponent = 2.0 * pair * inverseWidth;
            double angle = position * Math.Pow(baseFrequency, -exponent);
            float cosine = (float)Math.Cos(angle);
            float sine = (float)Math.Sin(angle);
            float first = values[offset + firstIndex];
            float second = values[offset + secondIndex];
            values[offset + firstIndex] = (first * cosine) - (second * sine);
            values[offset + secondIndex] = (second * cosine) + (first * sine);
        }
    }

    private static long AttentionOffset(
        long batch,
        long head,
        long sequence,
        long heads,
        long sequenceLength,
        long width) =>
        (((batch * heads + head) * sequenceLength) + sequence) * width;

    private static void ValidateAttention(
        Tensor query,
        Tensor key,
        Tensor value,
        double? scale)
    {
        if (query.Shape.Count != 4 || key.Shape.Count != 4 || value.Shape.Count != 4)
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.ShapeMismatch,
                "Causal attention tensors must have [batch, heads, sequence, width] shape.");
        }
        if (query.DataType is not (DataType.BFloat16 or DataType.Float32) ||
            query.DataType != key.DataType ||
            query.DataType != value.DataType)
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.DTypeMismatch,
                "Attention Q/K/V must have the same floating dtype.");
        }
        if (query.Shape[0] != key.Shape[0] ||
            query.Shape[0] != value.Shape[0] ||
            query.Shape[3] != key.Shape[3] ||
            query.Shape[3] != value.Shape[3] ||
            key.Shape[1] != value.Shape[1] ||
            key.Shape[2] != value.Shape[2] ||
            key.Shape[1] == 0 ||
            query.Shape[1] == 0 ||
            query.Shape[1] % key.Shape[1] != 0 ||
            query.Shape[2] == 0 ||
            key.Shape[2] == 0 ||
            query.Shape[3] == 0 ||
            query.Shape[2] > key.Shape[2])
        {
            throw new PowerTorchException(
                PowerTorchErrorCategory.ShapeMismatch,
                "Attention shapes or grouped-query head counts are inconsistent.");
        }
        if (scale is { } requestedScale &&
            (!double.IsFinite(requestedScale) || requestedScale <= 0))
        {
            throw new ArgumentOutOfRangeException(
                nameof(scale),
                "Attention scale must be finite and positive.");
        }
    }

    private static long[] ReadInt64Indices(Tensor tensor)
    {
        return tensor.DataType switch
        {
            DataType.Int32 => tensor.AsSpan<int>().ToArray().Select(value => (long)value).ToArray(),
            DataType.Int64 => tensor.AsSpan<long>().ToArray(),
            _ => throw new PowerTorchException(
                PowerTorchErrorCategory.DTypeMismatch,
                "Integer indices must use Int32 or Int64."),
        };
    }

    private static float[] ReadFloatingValues(Tensor tensor)
    {
        return tensor.DataType switch
        {
            DataType.BFloat16 => tensor.AsSpan<BFloat16>().ToArray()
                .Select(value => (float)value)
                .ToArray(),
            DataType.Float32 => tensor.AsSpan<float>().ToArray(),
            DataType.Float64 => tensor.AsSpan<double>().ToArray()
                .Select(value => checked((float)value))
                .ToArray(),
            _ => throw Unsupported(tensor.DataType),
        };
    }

    private static Tensor FromFloatingValues(
        DataType dataType,
        IReadOnlyList<long> shape,
        float[] values)
    {
        return dataType switch
        {
            DataType.BFloat16 => TensorFactory.FromArray(
                values.Select(value => (BFloat16)value).ToArray(),
                shape),
            DataType.Float32 => TensorFactory.FromArray(values, shape),
            DataType.Float64 => TensorFactory.FromArray(
                values.Select(value => (double)value).ToArray(),
                shape),
            _ => throw Unsupported(dataType),
        };
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
