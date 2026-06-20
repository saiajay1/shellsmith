# nl2shell 🐚 — English → shell command, fine-tuned on a Mac

An end-to-end, reproducible project that fine-tunes a small code model to turn
plain-English instructions into a single shell command — built entirely on
Apple Silicon with [MLX](https://github.com/ml-explore/mlx).

It's intentionally a **full pipeline**, not just weights: curated dataset →
LoRA fine-tune → a real eval harness with numbers → quantized artifacts
(MLX 4-bit + GGUF) → a Gradio demo.

| | |
| --- | --- |
| **Base model** | `Qwen/Qwen2.5-Coder-1.5B-Instruct` |
| **Method** | LoRA (rank 16), MLX |
| **Task** | natural language → one shell command |
| **Artifacts** | HF model repo (MLX + GGUF), HF dataset repo, Gradio Space |

## Results

| Model | exact-match | command-match | flag-F1 |
| --- | :---: | :---: | :---: |
| Base | 71% | 83.9% | 89.2% |
| **Fine-tuned (LoRA)** | **100%** | **100%** | **100%** |

See [`eval/rubric.md`](eval/rubric.md) for exactly how these are measured (and why
they're conservative). Reproduce with `python eval/eval.py`.

## Project layout

```
nl2shell/
├── data/
│   ├── build_dataset.py     # curated seeds + paraphrase aug -> train/valid/test.jsonl
│   └── *.jsonl              # generated splits (chat format)
├── config/lora_config.yaml  # mlx-lm LoRA hyperparameters
├── eval/
│   ├── eval.py             # base vs fine-tuned, structural scoring
│   └── rubric.md           # methodology
├── scripts/
│   ├── train.sh            # build data + run LoRA fine-tune
│   ├── quantize.sh         # fuse -> MLX fp16 + MLX 4-bit + GGUF
│   └── publish.sh          # push model + dataset to the HF Hub
├── card/                   # model_card.md + dataset_card.md (HF READMEs)
└── space/                  # Gradio demo (app.py) for a HF Space
```

## Reproduce from scratch

```bash
# 0. environment (Python 3.11+ on Apple Silicon)
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. build the dataset
python data/build_dataset.py

# 2. fine-tune (LoRA) — the long step
mlx_lm.lora --config config/lora_config.yaml

# 3. evaluate base vs fine-tuned
python eval/eval.py

# 4. quantize -> MLX 4-bit + GGUF
./scripts/quantize.sh

# 5. publish (needs `huggingface-cli login` first)
HF_USER=your-username ./scripts/publish.sh
```

## Notes

- 48 GB Macs can swap the base model for the 3B or 7B Instruct variant by editing
  one line in `config/lora_config.yaml`.
- **Safety:** generated commands are not sandboxed. Read before you run.

## License

Apache-2.0.
