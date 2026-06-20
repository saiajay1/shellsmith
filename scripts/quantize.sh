#!/usr/bin/env bash
# Fuse the LoRA adapters into the base weights, then produce shareable artifacts:
#   1. an MLX fp16 fused model        (for Mac users via mlx-lm)
#   2. an MLX 4-bit quantized model   (small, fast on Mac)
#   3. a GGUF export                  (for llama.cpp / Ollama / LM Studio users)
#
# Run from project root:  ./scripts/quantize.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source ../.venv/bin/activate 2>/dev/null || source .venv/bin/activate

BASE="Qwen/Qwen2.5-Coder-1.5B-Instruct"

echo "==> [1/3] Fuse adapters -> dist/fused (fp16) + GGUF export"
mlx_lm.fuse \
  --model "$BASE" \
  --adapter-path adapters \
  --save-path dist/fused \
  --hf-path "$BASE" \
  --export-gguf \
  --gguf-path nl2shell-1.5b.gguf

echo "==> [2/3] Quantize the fused model to 4-bit MLX -> dist/mlx-4bit"
mlx_lm.convert \
  --hf-path dist/fused \
  --mlx-path dist/mlx-4bit \
  -q --q-bits 4 --q-group-size 64

echo "==> [3/3] Artifacts ready:"
echo "    dist/fused                 (MLX fp16 fused)"
echo "    dist/fused/nl2shell-1.5b.gguf  (GGUF for llama.cpp / Ollama / LM Studio)"
echo "    dist/mlx-4bit              (MLX 4-bit quantized)"
