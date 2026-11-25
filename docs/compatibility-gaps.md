# Initial compatibility gaps

This file records codebase-derived requirements for the first migration
slices.

## OmniParser inference

- Tensor creation/conversion: `tensor`, `from_numpy`, `arange`, `ones`, `zeros`
- Shape: `unsqueeze`, `permute`, `reshape`, `chunk`, `stack`, `cat`, slicing
- Math/post-processing: arithmetic, comparisons, `max`, `sigmoid`, `softmax`,
  `topk`, clamp, torchvision box conversion and NMS
- Device/dtype: `cpu`, `cuda`, `float32`, `float16`, integer tokens
- Model: TorchScript YOLO, Florence2 processor/model, inference mode, generation

## VibeVoice 7B production prerequisites

- BF16 checkpoint tensors: managed CPU storage and safetensors reads are supported
- Core tensor movement: reshape, permute/transpose, and positive-stride slices
  are supported
- Core CPU math: batched matmul, stable softmax, layer norm, and RMS norm use
  the correctness-first implementation
- BF16/FP32 model primitives: embedding lookup, grouped/depthwise causal Conv1D,
  rotary embeddings, and causal multi-head attention with GQA are supported
- Indexed sharded safetensors: relative shard paths are confined to the index
  directory, each shard is validated, and selected tensors are placed into
  bounded CPU storage
- Remaining blockers include model vocabulary/tokenization, attention masks,
  embedding and rotary position kernels, KV cache, decoding, audio codec/model
  execution, and a full VibeVoice model graph

## Real training step

- Causal-LM forward and logits
- Cross-entropy backward
- Autograd parameter filtering and gradient norms
- LoRA parameter injection and adapter disable
- AdamW step and zero-grad
- Native HF/TorchScript/safetensors state reconstruction
- DirectML operator execution beyond ONNX Runtime interop
- Autograd broadcasting, views, higher-order gradients, AMP, and checkpointed
  graph execution

## Implemented now

- Native safetensors metadata and BF16/float tensor reads
- CPU BF16/float32/float64 arithmetic, reductions, reshape, permute/transpose,
  slice, batched matmul, softmax, layer norm, and RMS norm
- CPU BF16/float32 embedding, causal grouped/depthwise Conv1D, rotary embeddings,
  and causal GQA attention
- Safe indexed sharded-safetensors state loading with per-tensor and aggregate
  byte limits
- Reverse-mode sigmoid/sum/mean and arithmetic gradients
- AdamW float32/float64 state and one optimizer step
- DirectML ONNX metadata loading in an isolated native process
- PowerShell cmdlets for checkpoint tensors, the implemented CPU operations,
  variable graphs, backward, and AdamW on the same tensor runtime

Device transfer/residency remains intentionally absent. DirectML currently
bridges host buffers and lacks a persistent device tensor ownership/lifecycle
model, so advertising transfer would be unsafe and misleading.

## Explicitly not present in current Darbot use

- Existing Python code does not directly use `torch.distributed`, xFormers,
  Unsloth, or `trl` APIs. Distributed and Unsloth compatibility is still in
  the M3 target, but the minimum vertical slice must first replace the actual
  OmniParser and RLHF probe paths.
