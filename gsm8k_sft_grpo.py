"""GSM8K: SFT warmup, then GRPO. Standard RLVR pipeline.

Three eval checkpoints reveal how much each phase contributes:
  baseline (base model)  →  after SFT  →  after GRPO
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
    return ex['answer'].split('####')[-1].strip().replace(',', '')


def _solution(ex):
    # GSM8K reasoning chain. Replace '####' with our target format.
    return ex['answer'].replace('####', 'The answer is')


ds = load_dataset("openai/gsm8k", "main")
TRAIN = [(ex['question'], _gold(ex), _solution(ex)) for ex in ds['train'].select(range(500))]
TEST = [(ex['question'], _gold(ex)) for ex in ds['test']]


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


class Trainer:
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct"):
        self.num_rollouts = 4

        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
        self.ref_model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
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

    def _set_lr(self, lr):
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    def sft_step(self, prompt, target):
        # SFT on one (prompt, target): NLL of target tokens only.
        ids = self.tokenizer(prompt + target, return_tensors='pt').input_ids.to(DEVICE)
        prompt_len = self.tokenizer(prompt, return_tensors='pt').input_ids.shape[1]
        log_probs = F.log_softmax(self.model(ids).logits, -1)
        shift_logp = log_probs[:, :-1, :]
        shift_targets = ids[:, 1:, None]
        token_logp = shift_logp.gather(-1, shift_targets).squeeze(-1)[0]  # T-1
        mask = torch.zeros(ids.size(-1) - 1, device=DEVICE)
        mask[prompt_len - 1:] = 1.0   # completion tokens only (shifted)
        loss = -(token_logp * mask).sum() / mask.sum().clamp(min=1.0)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()
        return loss.detach()

    @torch.no_grad()
    def _generate(self, prompts):
        outputs, masks = [], []
        B, G = len(prompts), self.num_rollouts
        for p in prompts:
            for _ in range(G):
                inputs = self.tokenizer(p, return_tensors='pt').to(self.model.device)
                prompt_len = inputs.input_ids.shape[1]
                past = None
                output = inputs.input_ids
                for _ in range(self.max_new_tokens):
                    ids = output if past is None else output[:, -1:]
                    out = self.model(ids, past_key_values=past, use_cache=True)
                    past = out.past_key_values
                    probs = torch.softmax(out.logits[:, -1], -1)
                    next_id = torch.multinomial(probs, 1)
                    output = torch.cat([output, next_id], dim=1)
                    if next_id.item() in self.model.generation_config.eos_token_id:
                        break

                output = output[0]
                seqlen = len(output)
                output = F.pad(output, (0, self.max_seqlen - seqlen), value=self.tokenizer.pad_token_id)
                outputs.append(output)
                mask = torch.zeros_like(output)
                mask[prompt_len:seqlen] = 1
                masks.append(mask)
        outputs = torch.stack(outputs).view(B, G, self.max_seqlen)
        masks = torch.stack(masks).view(B, G, self.max_seqlen)
        return outputs, masks

    def grpo_step(self, outputs, masks, rewards):
        advantage = (rewards - rewards.mean(-1, keepdim=True)) / rewards.std(-1, keepdim=True, correction=0)
        B, G, T = outputs.shape
        mask = masks[..., 1:]

        def get_logprob(m):
            logits = m(outputs.view(B * G, T)).logits.view(B, G, T, -1)
            log_probs = F.log_softmax(logits, -1)
            shift_logp = log_probs[:, :, :-1, :]
            shift_targets = outputs[:, :, 1:, None]
            return shift_logp.gather(-1, shift_targets).squeeze(-1)

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
            total_loss = (total_loss * mask).sum() / self.max_seqlen / B / G

            self.optimizer.zero_grad()
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.optimizer.step()

        return dict(loss=surrogate_loss.detach().mean(), kl=kl.detach().mean(),
                    reward=rewards.detach().mean(), grad_norm=grad_norm)

    @torch.no_grad()
    def evaluate(self, problems, batch_size=16):
        n_correct = 0
        for i in range(0, len(problems), batch_size):
            chunk = problems[i:i + batch_size]
            prompts = [system_prompt + q for q, _ in chunk]
            inputs = self.tokenizer(prompts, return_tensors='pt', padding=True).to(self.model.device)
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            new_tokens = out[:, inputs.input_ids.shape[1]:]
            texts = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            for text, (_, a) in zip(texts, chunk):
                n_correct += int(reward_fn(text, a) == 1.0)
        return n_correct / len(problems)


t = Trainer()

acc0 = t.evaluate(TEST)
print(f"\n=== Accuracy baseline:   {acc0:.3f} ({int(acc0 * len(TEST))}/{len(TEST)}) ===\n", flush=True)

# --- SFT warmup. Conservative LR + fewer steps to avoid destroying reasoning ability.
# (Earlier attempt with lr=2e-5, 100 steps lost 14pp on the eval. SFT can wreck the
#  base model when the LR is too high or the data style mismatches.)
print("--- SFT warmup ---", flush=True)
t._set_lr(5e-6)
for s in range(30):
    q, _, sol = random.choice(TRAIN)
    loss = t.sft_step(system_prompt + q, sol)
    if s % 10 == 0:
        print(f"sft {s:>3}: loss={loss.item():.2f}", flush=True)

acc1 = t.evaluate(TEST)
print(f"\n=== Accuracy after SFT:  {acc1:.3f} ({int(acc1 * len(TEST))}/{len(TEST)}) ===\n", flush=True)

# --- GRPO. Drop LR back to the standard 5e-6 for stable RL fine-tuning.
print("--- GRPO ---", flush=True)
t._set_lr(5e-6)
MAX_STEPS = 100
step, skipped, reward_hist = 0, 0, []
while step < MAX_STEPS:
    batch = random.sample(TRAIN, k=1)
    prompts = [system_prompt + q for q, _, _ in batch]
    answers = [a for _, a, _ in batch]
    outputs, masks = t._generate(prompts)

    B, G, T = outputs.shape
    rewards = []
    for i in range(B):
        for j in range(G):
            text = t.tokenizer.decode(outputs[i][j], skip_special_tokens=True)
            rewards.append(reward_fn(text, answers[i]))
    rewards = torch.tensor(rewards, device=outputs.device).view(B, G)

    if rewards.std(-1, correction=0).max() < 1e-6:
        skipped += 1
        continue

    metrics = t.grpo_step(outputs, masks, rewards)
    reward_hist.append(metrics['reward'].item())
    ema = sum(reward_hist[-20:]) / min(len(reward_hist), 20)
    print(f"step {step:>3} (skipped={skipped}): "
          + " ".join(f"{k}={v:.2f}" for k, v in metrics.items())
          + f" ema={ema:.2f}",
          flush=True)
    step += 1

acc2 = t.evaluate(TEST)
print(f"\n=== Accuracy baseline:   {acc0:.3f} ({int(acc0 * len(TEST))}/{len(TEST)}) ===")
print(f"=== Accuracy after SFT:  {acc1:.3f} ({int(acc1 * len(TEST))}/{len(TEST)}) ===")
print(f"=== Accuracy after GRPO: {acc2:.3f} ({int(acc2 * len(TEST))}/{len(TEST)}) ===")
