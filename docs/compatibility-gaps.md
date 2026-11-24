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

- Native safetensors metadata and real float tensor reads
- CPU float32/float64/int32/int64 add/multiply/subtract/divide and selected
  reductions
- Reverse-mode sigmoid/sum/mean and arithmetic gradients
- AdamW float32/float64 state and one optimizer step
- DirectML ONNX metadata loading in an isolated native process
- PowerShell cmdlets for checkpoint tensors, variable graphs, backward, and
  AdamW on the same tensor runtime

## Explicitly not present in current Darbot use

- Existing Python code does not directly use `torch.distributed`, xFormers,
  Unsloth, or `trl` APIs. Distributed and Unsloth compatibility is still in
  the M3 target, but the minimum vertical slice must first replace the actual
  OmniParser and RLHF probe paths.
