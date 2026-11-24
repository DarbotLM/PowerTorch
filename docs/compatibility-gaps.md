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

## Explicitly not present in current Darbot use

- Existing Python code does not directly use `torch.distributed`, xFormers,
  Unsloth, or `trl` APIs. Distributed and Unsloth compatibility is still in
  the M3 target, but the minimum vertical slice must first replace the actual
  OmniParser and RLHF probe paths.
