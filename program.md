# program.md — autoresearch loop for GSM8K GRPO

An autonomous agent experiments on `gsm8k_grpo.py` to improve GRPO on GSM8K.
Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

> **Note on this repo's layout.** The protocol below refers to the target as `gsm8k_grpo.py`
> (its name during the run). In the published repo that evolving file is preserved as
> [`gsm8k_grpo_autoresearch.py`](gsm8k_grpo_autoresearch.py) — the final state of the loop —
> while [`gsm8k_grpo.py`](gsm8k_grpo.py) holds the untuned textbook baseline the loop started from.
> The full experiment log is in [`results.tsv`](results.tsv); the progression plot is `autoresearch.png`.

## Setup

Once, before starting:

1. **Agree on a run tag** with the human (e.g. `may18`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from `main`.
3. **Read the in-scope files**:
   - `README.md` — repo context
   - `gsm8k_grpo.py` — the file you modify
4. **Verify the model is cached**: `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct` exists. If not, the first run will download it (~1GB).
5. **Initialize `results.tsv`** with just the header row (see below). Baseline gets recorded by your first run.
6. **Confirm with the human, then go.**

## What you can and can't change

You modify exactly one file: **`gsm8k_grpo.py`**.

**Frozen** — do not alter behavior of these (copy-edit only):
- `_gold`, `extract_answer`, `reward_fn` — the metric
- The `TRAIN` / `TEST` split and the GSM8K dataset itself
- `system_prompt`
- `evaluate()` — the eval harness (batched greedy decoding, full 200-problem subset, see below)
- The time-budget enforcement (see "Time budget")

**Mutable** — everything else is fair game:
- Model choice within Qwen2.5 family at ≤0.8B params
- Optimizer, LR, KL coefficient (`β`), PPO clip (`ε`), grad clip, minibatch count
- `num_rollouts`, `BATCH_SIZE`, `max_new_tokens`, `max_seqlen`
- Whether to use a ref model at all (β=0, drop it entirely)
- Whether to do SFT warmup, curriculum, prompt shaping
- The shape of the training loop itself — algorithm, not just hyperparams

## Time budget

Each experiment trains for **a fixed wall-clock budget of 5 minutes** (excluding model load and post-eval), then runs eval. Total ~8 minutes per experiment on the baseline Mac (M-series, 64GB).

Implement the budget at the top of the training loop with a `time.monotonic()` check — break out and proceed to eval when the budget is exceeded. Do not change the budget; it is what makes experiments comparable across architecture/optimizer changes.

## The metric

**`val_acc`** — accuracy on `TEST[:200]` after training. One scalar. Higher is better.

Why a 200-problem subset rather than the full 1319? Per-experiment eval on full TEST takes ~50min on the baseline Mac. 200 problems takes ~5min. The autoresearch loop needs eval << train, otherwise you only get a few experiments per night. Sampling noise at N=200 is ~±2pp — improvements smaller than that are noise.

If you want a secondary signal, log peak training EMA reward too — it's free and confirms whether training did anything.

## VRAM

Soft constraint: keep peak resident memory under ~30GB. OOM = crash = discard. Apple MPS reports memory via `torch.mps.driver_allocated_memory()` after a `torch.mps.synchronize()`.

## Simplicity criterion

All else equal, simpler is better. A 0.5pp gain that adds 30 lines of hacky code is probably worse than removing 10 lines for a 0pp change. Keep simplifications; keep meaningful gains; discard the rest. Educational clarity is part of the goal — this is `nanoRL`.

## The first run

Establish the baseline. Run `gsm8k_grpo.py` as-is on the agreed `main` config (Qwen2.5-0.5B-Instruct, B=4, lr=5e-6, β=0.04, num_rollouts=4, etc.). Whatever `val_acc` you measure is the baseline you must beat.

## Logging

Append every experiment to **`results.tsv`** (tab-separated, NOT comma). Do not commit `results.tsv` — leave it untracked.

Header + example:

```
commit	val_acc	peak_gb	status	description
a1b2c3d	0.355	6.2	keep	baseline (Qwen2.5-0.5B-Instruct, B=4, β=0.04)
b2c3d4e	0.378	4.8	keep	drop ref model, β=0 → faster, more steps in budget
c3d4e5f	0.000	0.0	crash	num_rollouts=8 OOM at B=4
d4e5f6g	0.352	6.2	discard	lr=1e-5 → unstable, KL hit 1.4
```

Columns:
1. git commit hash (7 chars)
2. `val_acc` on `TEST[:200]` — 4 decimals, `0.0000` for crash
3. peak memory in GB (`.1f`), `0.0` for crash
4. `keep` | `discard` | `crash`
5. one-line description of what was tried

## The loop

```
LOOP FOREVER:
  1. Note current branch tip (commit + val_acc).
  2. Edit gsm8k_grpo.py with one experimental change.
  3. git commit -am "<short desc>".
  4. Run: uv run python gsm8k_grpo.py > run.log 2>&1
     (redirect — do NOT tee, do NOT let output flood your context)
  5. Read results:
       grep -E "^=== Accuracy after RL:|^=== best training EMA:" run.log
       grep "peak_gb" run.log    # if you choose to log it
  6. If grep is empty → crash. tail -n 50 run.log, diagnose.
       - Dumb fix (typo, missing import): fix and re-run.
       - Fundamentally broken idea: log "crash", revert, move on.
  7. Append a row to results.tsv.
  8. If val_acc improved (or equal but simpler): keep the commit, branch advances.
     Else: git reset --hard HEAD~1.
```

## Timeout

If a single run exceeds 12 minutes (5 train + 5 eval + buffer), kill it. Treat as crash, revert, move on. Long-running experiments are noise — your iteration speed matters more than any one result.

## Never stop

Once setup is done, do not ask the human "should I continue?" or "is this a good stopping point?". The human may be asleep. Run until they interrupt you.

If you run out of ideas: re-read the in-scope files, re-read recent `results.tsv` rows for near-misses worth combining, try a more radical change (different optimizer, different KL formulation, different reward shaping). There is always something to try.

---

## Speed-up suggestions for the agent

5-minute training budget is tight at the 0.5B baseline pace (~30s/step → ~10 effective steps). The experiments that buy you more steps-per-minute are often the same ones that lower wall-clock per experiment night-over-night. Worth trying early:

1. **Drop the ref model.** Set `β=0` and remove the ref-model forward entirely. Saves ~1/3 of per-step compute, ~1.6GB resident memory, and the second `get_logprob` call. PPO with just the clip term often works fine for short runs.
2. **Smaller `max_new_tokens`.** 256 → 128 roughly halves generation time. Cuts off some long CoT answers, but the reward function only cares about the final number — most GSM8K solutions fit in 128 tokens.
3. **`minibatch = 1`.** With bf16 + lr=5e-6, the second PPO inner-loop step lands close to the precision floor (you can verify with grad_norm logging). Saves one forward + backward.
4. **`num_rollouts = 2`.** Halves generation cost. Loosen the variance filter accordingly — at G=2 the std-zero filter is more aggressive.
5. **SGD over AdamW.** Reclaims ~6GB of moment buffers. Tune `lr` accordingly (often 5-10x larger for SGD).
6. **Reuse rollout KV cache for `old_log_probs`.** The rollout already ran the model over those tokens — recomputing the forward to get logits is wasted work. Implementable but fiddly; worth ~25% of step time.
7. **Pre-truncate prompts.** GSM8K prompts vary in length; left-padding to the batch max wastes compute. Sort by length per batch.

These are starting points — the agent should propose and test new ones.

## On metric noise

`val_acc` at N=200 has ~±2pp standard deviation from sampling. If a change shows +1.5pp, that may be noise. Treat improvements <2pp as "maybe" — try the change in combination with something else, or re-run to confirm. Improvements ≥3pp are likely real.

The most reliable signal is consistent direction across multiple related experiments — if dropping the ref model gives +2pp, and dropping the ref model + smaller max_new_tokens gives +3pp, the trend is real even if no single result clears the noise floor.
