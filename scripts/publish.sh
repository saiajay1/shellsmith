#!/usr/bin/env bash
# Publish the model + dataset to the Hugging Face Hub.
#
# PREREQUISITE (one time): log in with your token (get one at
#   https://huggingface.co/settings/tokens  -> "Write" scope):
#       source .venv/bin/activate && huggingface-cli login
#
# Then set your HF username and run:  HF_USER=your-username ./scripts/publish.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source ../.venv/bin/activate 2>/dev/null || source .venv/bin/activate

: "${HF_USER:?Set HF_USER to your Hugging Face username, e.g. HF_USER=ajay ./scripts/publish.sh}"

MODEL_REPO="$HF_USER/Qwen2.5-Coder-1.5B-Shellsmith"
DATA_REPO="$HF_USER/shellsmith-commands"

echo "==> Creating repos (idempotent)"
huggingface-cli repo create "$MODEL_REPO" --type model -y || true
huggingface-cli repo create "$DATA_REPO"  --type dataset -y || true

echo "==> Copying model card into the artifact folder"
cp card/model_card.md dist/mlx-4bit/README.md

echo "==> Uploading 4-bit MLX model + GGUF + card"
huggingface-cli upload "$MODEL_REPO" dist/mlx-4bit . --repo-type model
# GGUF (if you ran ./scripts/export_gguf.sh)
if [ -f dist/shellsmith-1.5b-f16.gguf ]; then
  huggingface-cli upload "$MODEL_REPO" dist/shellsmith-1.5b-f16.gguf shellsmith-1.5b-f16.gguf --repo-type model
fi

echo "==> Uploading dataset (train/valid/test + card)"
cp card/dataset_card.md /tmp/_ds_readme.md
huggingface-cli upload "$DATA_REPO" data/train.jsonl train.jsonl --repo-type dataset
huggingface-cli upload "$DATA_REPO" data/valid.jsonl valid.jsonl --repo-type dataset
huggingface-cli upload "$DATA_REPO" data/test.jsonl  test.jsonl  --repo-type dataset
huggingface-cli upload "$DATA_REPO" /tmp/_ds_readme.md README.md --repo-type dataset

echo "==> Done."
echo "    Model:   https://huggingface.co/$MODEL_REPO"
echo "    Dataset: https://huggingface.co/datasets/$DATA_REPO"
