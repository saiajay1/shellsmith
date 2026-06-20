#!/usr/bin/env python3
"""Substitute eval numbers from results.json into README.md and the model card.

Run after `python eval/eval.py`:  python eval/fill_results.py
Replaces the BASE_* / FT_* placeholders in-place.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

results = json.loads((ROOT / "eval" / "results.json").read_text())
by_label = {r["label"]: r for r in results}
base, ft = by_label["base"], by_label["fine-tuned"]

repl = {
    "BASE_EXACT": f'{base["exact_match"]:g}',
    "BASE_CMD": f'{base["command_match"]:g}',
    "BASE_F1": f'{base["flag_f1"]:g}',
    "FT_EXACT": f'{ft["exact_match"]:g}',
    "FT_CMD": f'{ft["command_match"]:g}',
    "FT_F1": f'{ft["flag_f1"]:g}',
}

for rel in ["README.md", "card/model_card.md"]:
    p = ROOT / rel
    text = p.read_text()
    for k, v in repl.items():
        text = text.replace(f"`{k}`", v)   # in tables we wrote `BASE_EXACT`%
        text = text.replace(k, v)
    p.write_text(text)
    print(f"filled {rel}")

print(f"\nbase       : exact {base['exact_match']}%  cmd {base['command_match']}%  f1 {base['flag_f1']}%")
print(f"fine-tuned : exact {ft['exact_match']}%  cmd {ft['command_match']}%  f1 {ft['flag_f1']}%")
