#!/usr/bin/env python3
"""Evaluate base vs. LoRA-fine-tuned model on the held-out NL->shell test set.

Metrics (all deterministic, no command execution -> safe + reproducible):

  exact_match    : normalized predicted command == reference command
  command_match  : same primary executable AND option-flag F1 >= 0.8
  flag_f1 (avg)  : mean F1 over the set of -flags / --flags

Usage:
  python eval/eval.py                       # eval base + fine-tuned, print table
  python eval/eval.py --adapter adapters    # point at a specific adapter dir
  python eval/eval.py --limit 10            # quick smoke test on first 10
"""
import argparse
import json
import re
import shlex
from pathlib import Path

from mlx_lm import load, generate

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
TEST = ROOT / "data" / "test.jsonl"


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def normalize(cmd: str) -> str:
    cmd = cmd.strip().strip("`").strip()
    # drop a leading "$ " prompt if the model adds one
    cmd = re.sub(r"^\$\s+", "", cmd)
    # collapse internal whitespace
    return re.sub(r"\s+", " ", cmd)


def primary_exec(cmd: str) -> str:
    """First real program name (handles `sudo`, pipes, redirects loosely)."""
    try:
        toks = shlex.split(cmd)
    except ValueError:
        toks = cmd.split()
    for t in toks:
        if t in ("sudo", "command", "time"):
            continue
        if "=" in t and not t.startswith("-"):  # FOO=bar env prefix
            continue
        return t
    return toks[0] if toks else ""


def flags(cmd: str) -> set:
    try:
        toks = shlex.split(cmd)
    except ValueError:
        toks = cmd.split()
    return {t for t in toks if t.startswith("-") and len(t) > 1}


def f1(pred: set, ref: set) -> float:
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    tp = len(pred & ref)
    if tp == 0:
        return 0.0
    precision = tp / len(pred)
    recall = tp / len(ref)
    return 2 * precision * recall / (precision + recall)


def score(pred: str, ref: str) -> dict:
    p, r = normalize(pred), normalize(ref)
    exact = p == r
    same_exec = primary_exec(p) == primary_exec(r) and primary_exec(r) != ""
    ff1 = f1(flags(p), flags(r))
    command_match = bool(same_exec and ff1 >= 0.8)
    return {"exact": exact, "command_match": command_match, "flag_f1": ff1}


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------
def load_test():
    rows = []
    with TEST.open() as f:
        for line in f:
            msgs = json.loads(line)["messages"]
            system = next(m["content"] for m in msgs if m["role"] == "system")
            user = next(m["content"] for m in msgs if m["role"] == "user")
            ref = next(m["content"] for m in msgs if m["role"] == "assistant")
            rows.append({"system": system, "user": user, "ref": ref})
    return rows


def predict(model, tokenizer, row) -> str:
    messages = [
        {"role": "system", "content": row["system"]},
        {"role": "user", "content": row["user"]},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    out = generate(model, tokenizer, prompt=prompt, max_tokens=96, verbose=False)
    # take only the first line of the completion
    return out.strip().splitlines()[0] if out.strip() else ""


def evaluate(adapter_path, rows, label):
    print(f"\n==> Loading model [{label}]"
          + (f" with adapter {adapter_path}" if adapter_path else ""))
    model, tokenizer = load(MODEL, adapter_path=adapter_path)
    agg = {"exact": 0, "command_match": 0, "flag_f1": 0.0}
    details = []
    for i, row in enumerate(rows, 1):
        pred = predict(model, tokenizer, row)
        s = score(pred, row["ref"])
        agg["exact"] += s["exact"]
        agg["command_match"] += s["command_match"]
        agg["flag_f1"] += s["flag_f1"]
        details.append({**row, "pred": pred, **s})
        mark = "OK " if s["command_match"] else "   "
        print(f"  [{i:2d}/{len(rows)}] {mark} {row['user'][:45]:45s} -> {pred[:40]}")
    n = len(rows)
    return {
        "label": label,
        "n": n,
        "exact_match": round(100 * agg["exact"] / n, 1),
        "command_match": round(100 * agg["command_match"] / n, 1),
        "flag_f1": round(100 * agg["flag_f1"] / n, 1),
        "details": details,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="adapters", help="adapter dir for fine-tuned run")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only first N (smoke test)")
    ap.add_argument("--base-only", action="store_true")
    ap.add_argument("--ft-only", action="store_true")
    args = ap.parse_args()

    rows = load_test()
    if args.limit:
        rows = rows[: args.limit]

    results = []
    if not args.ft_only:
        results.append(evaluate(None, rows, "base"))
    if not args.base_only:
        results.append(evaluate(args.adapter, rows, "fine-tuned"))

    print("\n" + "=" * 60)
    print(f"{'model':12s} {'exact%':>8s} {'cmd-match%':>11s} {'flag-F1%':>9s}")
    print("-" * 60)
    for r in results:
        print(f"{r['label']:12s} {r['exact_match']:>8.1f} "
              f"{r['command_match']:>11.1f} {r['flag_f1']:>9.1f}")
    print("=" * 60)

    out = ROOT / "eval" / "results.json"
    with out.open("w") as f:
        json.dump([{k: v for k, v in r.items() if k != "details"} for r in results],
                  f, indent=2)
        f.write("\n")
    with (ROOT / "eval" / "results_detailed.json").open("w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out.relative_to(ROOT)} (+ results_detailed.json)")


if __name__ == "__main__":
    main()
