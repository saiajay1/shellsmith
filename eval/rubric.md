# Evaluation methodology

The model maps a plain-English instruction to a **single shell command**. We
evaluate on a held-out `test.jsonl` split that the model never saw during
training (split is deterministic, seed `1707`).

We deliberately **do not execute** generated commands — running arbitrary
model output is unsafe and non-reproducible across machines. Instead we use
three deterministic, structural metrics:

| Metric | Definition |
| --- | --- |
| **exact_match** | Normalized prediction string equals the reference (whitespace-collapsed, stripped of backticks / `$` prompt). Strict — penalizes any harmless variation. |
| **command_match** | Same *primary executable* **and** option-flag F1 ≥ 0.8. This is the headline metric: it credits semantically-correct commands that differ cosmetically (e.g. flag order). |
| **flag_f1** | Mean F1 over the set of `-x` / `--long` flags, measuring how well the model recovers the right options. |

`primary_exec` ignores leading `sudo` / `time` / `FOO=bar` env prefixes so the
real program is compared.

### Why this is a fair proxy

- `command_match` rewards a correct tool + correct options regardless of
  cosmetic differences, which is what actually matters for a shell helper.
- It is a *lower bound* on true accuracy: a command can be functionally correct
  yet score 0 (e.g. `ls -la` vs `ls -al` flag bundling) — so reported numbers
  are conservative, not inflated.
- Limitation: it cannot credit a genuinely different-but-correct approach
  (e.g. `find` vs `ls` to list files). We accept this; the training data uses
  one canonical solution per task, so the test set rewards consistency.

Reproduce with:

```bash
python eval/eval.py            # full base vs fine-tuned comparison
```

Results are written to `eval/results.json` (summary) and
`eval/results_detailed.json` (per-example predictions for inspection).
