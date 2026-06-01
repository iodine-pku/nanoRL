"""GRPO on GSM8K — textbook setup (ref model + KL penalty + PPO clip). Eval baseline → train → eval.

This is the readable starting point: the same GRPO as minimal_grpo.py, scaled from toy
arithmetic to real grade-school math word problems. gsm8k_grpo_autoresearch.py is what an
autonomous research loop (program.md) converged to after 82 experiments — a stripped-down
REINFORCE that matches this file's accuracy with far less machinery.
"""
import random
import re
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F

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
TEST = [(ex['question'], _gold(ex)) for ex in ds['test']]  # full 1319-problem test set


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
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct"):
        self.num_rollouts = 4

        # bf16 halves memory + ~2x compute on Apple/Ampere/Hopper. No grad-scaler needed (unlike fp16).
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(DEVICE)
        self.ref_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Left-pad so the prompt-end positions align across batched generation.
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_new_tokens = 256
        self.max_seqlen = 512
        self.epsilon = 0.2
        self.beta = 0.04
        self.minibatch = 2

        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.ref_model.eval()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-6)

    @torch.no_grad()
    def _generate(self, prompts):
        # Batched sampling via HF generate. num_return_sequences=G gets all G rollouts
        # of all B prompts in one call. Result is shape (B*G, T) with left-padded prompts.
        B, G = len(prompts), self.num_rollouts
        inputs = self.tokenizer(prompts, return_tensors='pt', padding=True).to(DEVICE)
        prompt_len = inputs.input_ids.shape[1]   # left-pad: prompts all end at this index
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=1.0,
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
        # With B>1, individual groups can have zero variance even when the batch-level
        # filter passed. The +1e-8 zeros their advantage cleanly without div-by-zero.
        advantage = (rewards - rewards.mean(-1, keepdim=True)) / (rewards.std(-1, keepdim=True, correction=0) + 1e-8)
        B, G, T = outputs.shape
        mask = masks[..., 1:]

        def get_logprob(m):
            # Materializing (B*G, T, V) logits costs 9.4GB at B=16,G=4,T=512,V=151k,bf16.
            # Compute log-prob of just the target token per position via cross_entropy
            # (which fuses log_softmax + gather without keeping full softmax around).
            flat = outputs.view(B * G, T)
            logits = m(flat).logits  # (B*G, T, V)
            shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))  # (B*G*(T-1), V)
            shift_targets = flat[:, 1:].reshape(-1)
            token_logp = -F.cross_entropy(shift_logits, shift_targets, reduction='none')
            return token_logp.view(B, G, T - 1)

        old_log_probs = get_logprob(self.model).detach()
        ref_log_probs = get_logprob(self.ref_model).detach()

        for _ in range(self.minibatch):
            log_probs = get_logprob(self.model)
            ratio = torch.exp(log_probs - old_log_probs)
            loss1 = ratio * advantage.view(B, G, 1)
            loss2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage.view(B, G, 1)
            surrogate_loss = -torch.minimum(loss1, loss2)
            r = ref_log_probs - log_probs
            kl = (torch.exp(r) - 1) - r
            total_loss = surrogate_loss + kl * self.beta
            total_loss = (total_loss * mask).sum() / self.max_seqlen / B / G  # dr-grpo norm

            self.optimizer.zero_grad()
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.optimizer.step()

        return dict(
            loss=surrogate_loss.detach().mean(),
            kl=kl.detach().mean(),
            reward=rewards.detach().mean(),
            grad_norm=grad_norm,
        )

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
                n_correct += int(reward_fn(text, a) == 1.0)
            if i == 0:
                print(f"  sample: {texts[0].strip()[:200]}", flush=True)
            done = min(i + batch_size, len(problems))
            if (done // batch_size) % 10 == 0 or done == len(problems):
                print(f"  eval {done}/{len(problems)}: {n_correct} correct so far", flush=True)
        return n_correct / len(problems)


grpo = GRPO()

acc_before = grpo.evaluate(TEST)
print(f"\n=== Accuracy before RL: {acc_before:.3f} ({int(acc_before * len(TEST))}/{len(TEST)}) ===\n", flush=True)

MAX_STEPS = 200
BATCH_SIZE = 4    # prompts per training step. With G=4, each step uses 16 trajectories.
step, skipped, reward_hist, best_ema = 0, 0, [], 0.0
while step < MAX_STEPS:
    batch = random.sample(TRAIN, k=BATCH_SIZE)
    prompts = [system_prompt + q for q, a in batch]
    answers = [a for q, a in batch]
    outputs, masks = grpo._generate(prompts)

    B, G, T = outputs.shape
    rewards = []
    for i in range(B):
        for j in range(G):
            text = grpo.tokenizer.decode(outputs[i][j], skip_special_tokens=True)
            rewards.append(reward_fn(text, answers[i]))
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

acc_after = grpo.evaluate(TEST)
print(f"\n=== Accuracy before RL: {acc_before:.3f} ({int(acc_before * len(TEST))}/{len(TEST)}) ===")
print(f"=== Accuracy after RL:  {acc_after:.3f} ({int(acc_after  * len(TEST))}/{len(TEST)}) ===")
print(f"=== best training EMA:  {best_ema:.3f}  (final EMA: {ema:.3f}) ===")
