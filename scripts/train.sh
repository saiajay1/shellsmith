#!/usr/bin/env bash
# Fine-tune the base model with LoRA. Run from the project root:
#   ./scripts/train.sh
set -euo pipefail
cd "$(dirname "$0")/.."

source ../.venv/bin/activate 2>/dev/null || source .venv/bin/activate

echo "==> Building dataset"
python data/build_dataset.py

echo "==> Starting LoRA fine-tune (this is the long step)"
mlx_lm.lora --config config/lora_config.yaml

echo "==> Done. Adapters written to ./adapters"
echo "    Try it:  mlx_lm.generate --model Qwen/Qwen2.5-Coder-1.5B-Instruct \\"
echo "             --adapter-path adapters --prompt 'list files by size'"
