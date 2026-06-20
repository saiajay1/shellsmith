---
license: apache-2.0
base_model: Qwen/Qwen2.5-Coder-1.5B-Instruct
tags:
  - mlx
  - lora
  - text-generation
  - shell
  - command-line
  - code
language:
  - en
pipeline_tag: text-generation
library_name: mlx
---

# Qwen2.5-Coder-1.5B-Shellsmith

A small, fast model that turns **plain-English instructions into a single shell
command** for macOS/Linux. LoRA fine-tune of
[`Qwen/Qwen2.5-Coder-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct),
trained and quantized end-to-end on an Apple Silicon Mac with
[MLX](https://github.com/ml-explore/mlx).

> "list files by size, biggest first" → `ls -lS`
> "find files larger than 100 megabytes" → `find . -type f -size +100M`
> "create a gzip tar archive of src named src.tar.gz" → `tar -czf src.tar.gz src`

## Results

Evaluated on a held-out test split the model never saw during training.
Metrics are structural (no command execution) and conservative — see the
[eval rubric](https://github.com/saiajay1/shellsmith/blob/main/eval/rubric.md).

| Model | exact-match | command-match | flag-F1 |
| --- | :---: | :---: | :---: |
| Base (Qwen2.5-Coder-1.5B-Instruct) | 71% | 83.9% | 89.2% |
| **This model (LoRA)** | **100%** | **100%** | **100%** |

*command-match = correct program **and** option-flag F1 ≥ 0.8.*

> **What the 100% means (and doesn't):** the test split holds out unseen
> *phrasings*, but the underlying task distribution (84 canonical tasks) overlaps
> with training. So this measures reliable **in-distribution generalization across
> wording** — the model consistently emits the canonical, idiomatic command
> (`git add -A`, `ls -lS`, `git log --oneline -5`) where the base model drifts to
> looser variants (`git add .`, `ls -lh | sort -rh`, `git log -5`). It is **not**
> evidence of generalization to entirely novel tasks; broadening the task set is
> the obvious next step.

## Usage

### MLX (Apple Silicon)

```bash
pip install mlx-lm
mlx_lm.generate --model ajayk007/Qwen2.5-Coder-1.5B-Shellsmith \
  --prompt "compress the logs folder into logs.tar.gz"
```

```python
from mlx_lm import load, generate
model, tok = load("ajayk007/Qwen2.5-Coder-1.5B-Shellsmith")
messages = [
    {"role": "system", "content": "You are a shell command generator for macOS/Linux. "
     "Given a task in plain English, reply with a single safe shell command. "
     "Output only the command on one line, no explanation, no markdown."},
    {"role": "user", "content": "find all python files modified today"},
]
prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
print(generate(model, tok, prompt=prompt, max_tokens=64))
```

### GGUF (llama.cpp / Ollama / LM Studio)

A `shellsmith-1.5b-f16.gguf` file is included in this repo for use with llama.cpp-based runtimes.

## Prompt format

Uses the Qwen chat template with the system prompt shown above. Keep the system
prompt for best results.

## Training

- **Method:** LoRA (rank 16), 16 layers, 400 iterations, batch size 4, lr 1e-4
- **Hardware:** Apple M5 Pro (48 GB), MLX
- **Data:** [`ajayk007/shellsmith-commands`](https://huggingface.co/datasets/ajayk007/shellsmith-commands) —
  curated (instruction, command) pairs with paraphrase augmentation, 80/10/10 split.

## Limitations & safety

- Generates commands across common categories (files, find/grep, archives, git,
  processes, networking). Outside this scope it falls back to base behavior.
- **Always read a generated command before running it.** It can produce
  destructive commands (`rm`, `kill`, `chmod`) if you ask for them. There is no
  sandbox or confirmation step.
- Single-command only; it does not write multi-step scripts.

## Related
Part of a series of focused "English → developer DSL" fine-tunes:
- [Qwen2.5-Coder-7B-Querysmith](https://huggingface.co/ajayk007/Qwen2.5-Coder-7B-Querysmith) — schema-grounded text-to-SQL.

## License

Apache-2.0, inheriting from the base model.
