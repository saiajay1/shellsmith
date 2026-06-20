---
title: nl2shell
emoji: 🐚
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
---

# nl2shell — English → shell command

Live demo of [`Qwen2.5-Coder-1.5B-nl2shell`](https://huggingface.co/REPLACE_ME/Qwen2.5-Coder-1.5B-nl2shell).
Type a task in plain English, get a single shell command back.

Fine-tuned with LoRA on an Apple M5 Pro using MLX. Runs here on free CPU via the
GGUF build. **Always review a generated command before running it.**

To deploy: create a new Gradio Space, then upload `app.py`, `requirements.txt`,
and this `README.md`.
