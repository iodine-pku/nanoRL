"""GRPO on GSM8K — the final state of an autonomous autoresearch loop (see program.md).

Starting from the textbook setup in gsm8k_grpo.py (ref model + KL + PPO clip), 82 experiments
converged on plain REINFORCE with a group-relative baseline: no ref model, no clip (at
minibatch=1 the PPO ratio is identically 1, so the clip never fired — it was vestigial).
Trains for a fixed 20-min wall-clock budget, then evals val_acc on TEST[:200].
See README.md and autoresearch.png for the full progression.

Note: a one-token completion-mask off-by-one present during the run has since been fixed
in all files; results.tsv predates the fix and is unaffected (well within the ±7pp noise floor).
"""
import random
import re
import time
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import torch
import torch.nn.functional as F

# Fixed seed: makes runs deterministic so experimental differences are real, not run-to-run
# training variance (which empirically spanned 7pp across re-runs of the same config).
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
set_seed(SEED)

DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'

system_prompt = (
    "Solve the problem step by step. End with 'The answer is X.' "
    "where X is a number.\n\n"
)


def _gold(ex):
    # GSM8K gold is in the answer field after '####'.
    return ex['answer'].split('####')[-1].strip().replace(',', '')


ds = load_dataset("openai/gsm8k", "main")
TRAIN = [(ex['question'], _gold(ex)) for ex in ds['train'].select(range(500))]
TEST = [(ex['question'], _gold(ex)) for ex in ds['test'].select(range(200))]  # 200-problem subset; eval ~5 min on Mac. Swap to ds['test'] for the full 1319.


def extract_answer(text):
    t = text.replace(',', '').replace('$', '')
    m = re.search(r'answer is\s*(-?\d+\.?\d*)', t, re.IGNORECASE)
    if m:
        return m.group(1)
    nums = re.findall(r'-?\d+\.?\d*', t)
    return nums[-1] if nums else None


def reward_fn(completion, answer):
    pred = extract_answer(completion)
    if pred is None:
        return 0.0
    try:
        return 1.0 if abs(float(pred) - float(answer)) < 1e-6 else 0.0
    except ValueError:
        return 0.0


class GRPO:
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct", reward_fn=reward_fn):
        self.num_rollouts = 4
        # Training reward — pass a different callable to shape the signal (format / partial
        # credit / length penalty). Eval always uses the frozen module-level `reward_fn` metric
        # for cross-experiment comparability.
        self.reward_fn = reward_fn

        # bf16 halves memory + ~2x compute on Apple/Ampere/Hopper. No grad-scaler needed (unlike fp16).
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Left-pad so the prompt-end positions align across batched generation.
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_new_tokens = 256
        self.max_seqlen = 512
        # No ref model, no PPO clip — plain REINFORCE with group baseline. With one
        # update per rollout batch (mb=1), the PPO ratio is ≡1 and the clip never fires,
        # so the whole IS/clip machinery was vestigial. Just policy gradient × advantage.

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-6)

    @torch.no_grad()
    def _generate(self, prompts, temperature=1.0):
        # Batched sampling via HF generate. num_return_sequences=G gets all G rollouts
        # of all B prompts in one call. Result is shape (B*G, T) with left-padded prompts.
        B, G = len(prompts), self.num_rollouts
        inputs = self.tokenizer(prompts, return_tensors='pt', padding=True).to(DEVICE)
        prompt_len = inputs.input_ids.shape[1]   # left-pad: prompts all end at this index
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=G,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        # Pad/truncate to max_seqlen so downstream shape is fixed.
        if out.shape[1] < self.max_seqlen:
            out = F.pad(out, (0, self.max_seqlen - out.shape[1]), value=self.tokenizer.pad_token_id)
        else:
            out = out[:, :self.max_seqlen]

        # Build masks: 1 only on completion tokens. Generated region is [prompt_len, eos_or_end).
        gen = out[:, prompt_len:]
        is_pad = (gen == self.tokenizer.pad_token_id)
        first_pad = is_pad.float().argmax(dim=-1)
        first_pad[~is_pad.any(dim=-1)] = gen.size(1)   # ran to max_new_tokens with no pad
        seqlens = prompt_len + first_pad
        pos = torch.arange(self.max_seqlen, device=DEVICE)
        masks = ((pos >= prompt_len) & (pos < seqlens[:, None])).long()

        outputs = out.view(B, G, self.max_seqlen)
        masks = masks.view(B, G, self.max_seqlen)
        return outputs, masks

    def step(self, outputs, masks, rewards):
        # Group-baseline advantage. +1e-8 zeros the advantage for zero-variance groups
        # cleanly (when all rollouts in a group got the same reward).
        advantage = (rewards - rewards.mean(-1, keepdim=True)) / (rewards.std(-1, keepdim=True, correction=0) + 1e-8)
        B, G, T = outputs.shape
        mask = masks[..., 1:]

        # log-prob per token via cross_entropy (fuses log_softmax+gather; avoids 9.4GB softmax tensor).
        flat = outputs.view(B * G, T)
        logits = self.model(flat).logits
        shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
        shift_targets = flat[:, 1:].reshape(-1)
        log_probs = (-F.cross_entropy(shift_logits, shift_targets, reduction='none')).view(B, G, T - 1)

        # REINFORCE: -log π(a) · A.  Dr-GRPO normalization (/max_seqlen) avoids length bias.
        total_loss = -(log_probs * advantage.view(B, G, 1) * mask).sum() / self.max_seqlen / B / G

        self.optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()

        return dict(loss=total_loss.detach(), reward=rewards.detach().mean(), grad_norm=grad_norm)

    @torch.no_grad()
    def evaluate(self, problems, batch_size=16):
        # Batched greedy decoding via HF generate. Much faster than the per-prompt loop.
        n_correct = 0
        for i in range(0, len(problems), batch_size):
            chunk = problems[i:i + batch_size]
            prompts = [system_prompt + q for q, _ in chunk]
            inputs = self.tokenizer(prompts, return_tensors='pt', padding=True).to(self.model.device)
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            new_tokens = out[:, inputs.input_ids.shape[1]:]
            texts = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            for text, (_, a) in zip(texts, chunk):
                # Eval always uses the frozen `reward_fn` metric, not `self.reward_fn`,
                # so accuracy is comparable across runs that train with shaped rewards.
                n_correct += int(reward_fn(text, a) == 1.0)
            if i == 0:
                print(f"  sample: {texts[0].strip()[:200]}", flush=True)
            done = min(i + batch_size, len(problems))
            if (done // batch_size) % 10 == 0 or done == len(problems):
                print(f"  eval {done}/{len(problems)}: {n_correct} correct so far", flush=True)
        return n_correct / len(problems)


grpo = GRPO()

# Single post-train eval per program.md — val_acc is the metric. Skipping pre-eval keeps each
# experiment inside the timeout (20 min train + ~5 min eval + load + buffer).
TRAIN_BUDGET_S = 20 * 60  # 20-min wall-clock training budget.
BATCH_SIZE = 4            # prompts per training step. With G=4, each step uses 16 trajectories.
step, skipped, reward_hist, best_ema = 0, 0, [], 0.0
t_train_start = time.monotonic()
while time.monotonic() - t_train_start < TRAIN_BUDGET_S:
    batch = random.sample(TRAIN, k=BATCH_SIZE)
    prompts = [system_prompt + q for q, a in batch]
    answers = [a for q, a in batch]
    outputs, masks = grpo._generate(prompts)   # default temp=1.0

    B, G, T = outputs.shape
    rewards = []
    for i in range(B):
        for j in range(G):
            text = grpo.tokenizer.decode(outputs[i][j], skip_special_tokens=True)
            rewards.append(grpo.reward_fn(text, answers[i]))
    rewards = torch.tensor(rewards, device=outputs.device).view(B, G)

    # DAPO-style filter: skip when no group has any reward variance. With B>1 this is
    # rare since at least one prompt usually has mixed outcomes across its rollouts.
    if rewards.std(-1, correction=0).max() < 1e-6:
        skipped += 1
        continue

    metrics = grpo.step(outputs, masks, rewards)
    reward_hist.append(metrics['reward'].item())
    ema = sum(reward_hist[-20:]) / min(len(reward_hist), 20)  # running mean over last 20 steps
    best_ema = max(best_ema, ema)
    print(f"step {step:>3} (skipped={skipped}): "
          + " ".join(f"{k}={v:.2f}" for k, v in metrics.items())
          + f" ema={ema:.2f}",
          flush=True)
    step += 1

val_acc = grpo.evaluate(TEST)
print(f"\n=== val_acc: {val_acc:.4f} ({int(val_acc * len(TEST))}/{len(TEST)}) ===")
print(f"=== best training EMA: {best_ema:.3f} (final EMA: {ema if reward_hist else 0.0:.3f}) ===")

# Peak resident memory — used for the autoresearch loop's `peak_gb` column.
peak_gb = 0.0
if DEVICE == 'mps':
    torch.mps.synchronize()
    peak_gb = torch.mps.driver_allocated_memory() / 1024**3
elif DEVICE == 'cuda':
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
print(f"=== peak_gb: {peak_gb:.1f} ===")
