---
license: apache-2.0
task_categories:
  - text-generation
language:
  - en
tags:
  - shell
  - command-line
  - code
  - instruction-tuning
size_categories:
  - n<1K
---

# nl2shell-commands

Curated **(natural-language instruction → shell command)** pairs for macOS/Linux,
used to fine-tune [`Qwen2.5-Coder-1.5B-nl2shell`](https://huggingface.co/REPLACE_ME/Qwen2.5-Coder-1.5B-nl2shell).

## Format

JSONL in chat format (mlx-lm / OpenAI style):

```json
{"messages": [
  {"role": "system", "content": "You are a shell command generator ..."},
  {"role": "user", "content": "list files sorted by size, largest first"},
  {"role": "assistant", "content": "ls -lS"}
]}
```

## Splits

| File | Rows |
| --- | --- |
| `train.jsonl` | 206 |
| `valid.jsonl` | 26 |
| `test.jsonl` | 31 |

Built deterministically (seed `1707`) from 84 hand-written canonical seed pairs,
augmented with natural-language paraphrases of each instruction. Categories:
listing/navigation, find, grep/text-search, text processing, file ops, archives,
disk/system, networking, git, permissions, and environment/misc.

## Construction

One canonical command per task (so evaluation has a fair reference). See
[`data/build_dataset.py`](https://github.com/REPLACE_ME/nl2shell/blob/main/data/build_dataset.py)
to reproduce or extend.

## License

Apache-2.0.
