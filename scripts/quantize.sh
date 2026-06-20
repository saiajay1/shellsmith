#!/usr/bin/env bash
# Fuse the LoRA adapters into the base weights, then produce shareable MLX artifacts:
#   1. a fused fp16 model (HF-format)   -> dist/fused
#   2. an MLX 4-bit quantized model     -> dist/mlx-4bit  (small, fast on Mac)
#
# For a GGUF build (llama.cpp / Ollama / LM Studio), run ./scripts/export_gguf.sh
# afterwards (mlx-lm cannot export GGUF for the Qwen2 architecture).
#
# Run from project root:  ./scripts/quantize.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source ../.venv/bin/activate 2>/dev/null || source .venv/bin/activate

BASE="Qwen/Qwen2.5-Coder-1.5B-Instruct"

echo "==> [1/2] Fuse adapters into base -> dist/fused (HF-format fp16)"
mlx_lm.fuse \
  --model "$BASE" \
  --adapter-path adapters \
  --save-path dist/fused

echo "==> [2/2] Quantize the fused model to 4-bit MLX -> dist/mlx-4bit"
mlx_lm.convert \
  --hf-path dist/fused \
  --mlx-path dist/mlx-4bit \
  -q --q-bits 4 --q-group-size 64

echo "==> Artifacts ready:"
echo "    dist/fused      (fused fp16, HF-format — input for GGUF export)"
echo "    dist/mlx-4bit   (MLX 4-bit quantized — publish this for Mac users)"
echo "    Next (optional): ./scripts/export_gguf.sh  for a GGUF build"
