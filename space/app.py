"""Gradio demo for Qwen2.5-Coder-1.5B-Shellsmith.

Runs on a free Hugging Face Space (CPU). Loads the GGUF build via llama-cpp so
it works without a GPU; on Mac you can also run it locally with the same file.
"""
import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

REPO = "ajayk007/Qwen2.5-Coder-1.5B-Shellsmith"
GGUF = "shellsmith-1.5b-f16.gguf"

SYSTEM = (
    "You are a shell command generator for macOS/Linux. Given a task in plain "
    "English, reply with a single safe shell command that accomplishes it. "
    "Output only the command on one line, with no explanation and no markdown."
)

print("Downloading model weights...")
model_path = hf_hub_download(repo_id=REPO, filename=GGUF)
llm = Llama(model_path=model_path, n_ctx=1024, n_threads=4, verbose=False)


def to_command(instruction: str) -> str:
    instruction = (instruction or "").strip()
    if not instruction:
        return ""
    out = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": instruction},
        ],
        max_tokens=96,
        temperature=0.0,
    )
    text = out["choices"][0]["message"]["content"].strip()
    return text.splitlines()[0] if text else ""


EXAMPLES = [
    "list files sorted by size, largest first",
    "find all python files modified in the last 24 hours",
    "compress the logs folder into logs.tar.gz",
    "show the process listening on port 8080",
    "search for the word TODO in all files recursively",
    "create and switch to a new git branch called feature",
]

with gr.Blocks(title="shellsmith") as demo:
    gr.Markdown(
        "# 🔨 Shellsmith\n"
        "Plain English → a single shell command. "
        "Fine-tuned `Qwen2.5-Coder-1.5B` (LoRA, trained on an Apple M5 Pro with MLX).\n\n"
        "⚠️ **Always read a command before running it.**"
    )
    inp = gr.Textbox(label="What do you want to do?", placeholder="e.g. find files larger than 100 MB")
    out = gr.Code(label="Shell command", language="shell")
    btn = gr.Button("Generate", variant="primary")
    btn.click(to_command, inputs=inp, outputs=out)
    inp.submit(to_command, inputs=inp, outputs=out)
    gr.Examples(EXAMPLES, inputs=inp)

if __name__ == "__main__":
    demo.launch()
