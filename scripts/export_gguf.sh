#!/usr/bin/env bash
# Export the fused model to GGUF for llama.cpp / Ollama / LM Studio users.
#
# mlx-lm can't emit GGUF for Qwen2, so we use llama.cpp's pure-Python converter
# (no C++ build required for the f16 GGUF). Prereq: run ./scripts/quantize.sh
# first so dist/fused exists.
#
# Run from project root:  ./scripts/export_gguf.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source ../.venv/bin/activate 2>/dev/null || source .venv/bin/activate

FUSED="dist/fused"
OUT="dist/shellsmith-1.5b-f16.gguf"
TOOLS=".tools"
LLAMA="$TOOLS/llama.cpp"

[ -d "$FUSED" ] || { echo "ERROR: $FUSED missing — run ./scripts/quantize.sh first"; exit 1; }

echo "==> Using the original base tokenizer for conversion (the fine-tune does"
echo "    not change the tokenizer; mlx-lm re-serializes it in a format the"
echo "    GGUF converter can't read, so we copy the upstream files verbatim)."
BASE_SNAP=$(find ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-1.5B-Instruct/snapshots \
  -maxdepth 1 -mindepth 1 -type d | head -1)
if [ -n "$BASE_SNAP" ]; then
  for tf in tokenizer.json tokenizer_config.json vocab.json merges.txt; do
    [ -f "$BASE_SNAP/$tf" ] && cp "$BASE_SNAP/$tf" "$FUSED/$tf"
  done
fi

echo "==> Fetching llama.cpp converter (shallow clone, ~once)"
mkdir -p "$TOOLS"
if [ ! -d "$LLAMA" ]; then
  git clone --depth 1 https://github.com/ggerganov/llama.cpp "$LLAMA"
fi

echo "==> Setting up an isolated venv for the converter (keeps mlx-lm env clean)"
CONV_VENV="$TOOLS/gguf-venv"
[ -d "$CONV_VENV" ] || python -m venv "$CONV_VENV"
"$CONV_VENV/bin/pip" install --quiet -r "$LLAMA/requirements/requirements-convert_hf_to_gguf.txt"

echo "==> Converting $FUSED -> $OUT (f16)"
"$CONV_VENV/bin/python" "$LLAMA/convert_hf_to_gguf.py" "$FUSED" --outfile "$OUT" --outtype f16

echo "==> Done: $OUT"
echo
echo "Optional — make a small 4-bit GGUF (needs a one-time llama.cpp build):"
echo "    cmake -S $LLAMA -B $LLAMA/build -DCMAKE_BUILD_TYPE=Release >/dev/null"
echo "    cmake --build $LLAMA/build --target llama-quantize -j"
echo "    $LLAMA/build/bin/llama-quantize $OUT dist/shellsmith-1.5b-Q4_K_M.gguf Q4_K_M"
