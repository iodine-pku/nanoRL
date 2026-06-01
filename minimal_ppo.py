import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'

system_prompt = "write answer inside <answer></answer>\n"
DATASET = [
    ("What is 1 + 1?", "2"),
    ("What is 2 + 3?", "5"),
    ("What is 4 + 4?", "8"),
    ("What is 5 + 7?", "12"),
    ("What is 6 + 9?", "15"),
    ("What is 3 + 8?", "11"),
]

def reward_fn(completion, answer):
    pred = completion.split('<answer>')[-1].split('</answer>')[0]
    return 1.0 if pred == answer else 0.0

class PPO:
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
        # Separate critic transformer (own parameters end-to-end), like InstructGPT.
        self.value_model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
        self.value_head = nn.Linear(self.value_model.config.hidden_size, 1).to(DEVICE).to(self.value_model.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_new_tokens = 32
        self.max_seqlen = 64
        self.epsilon = 0.2
        self.minibatch = 2

        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

        params = (list(self.model.parameters())
                  + list(self.value_model.parameters())
                  + list(self.value_head.parameters()))
        self.optimizer = torch.optim.AdamW(params, lr=1e-6)

    @torch.no_grad()
    def _generate(self, prompts):
        outputs = []
        masks = []
        for p in prompts:
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

            # Mask = 1 only on completion tokens (not prompt, not pad).
            mask = torch.zeros_like(output)
            mask[prompt_len:seqlen] = 1
            masks.append(mask)

        return torch.stack(outputs), torch.stack(masks)

    def _policy_forward(self, m, ids):
        # Returns token-level log probs only. Used for policy gradient.
        log_probs = F.log_softmax(m(ids).logits, -1)
        shift_logp = log_probs[:, :-1, :]
        shift_targets = ids[:, 1:, None]
        return shift_logp.gather(-1, shift_targets).squeeze(-1)  # N, T-1

    def _values(self, ids):
        # V(s) from the separate critic transformer. Gradient flows through value_model end-to-end.
        out = self.value_model(ids, output_hidden_states=True)
        return self.value_head(out.hidden_states[-1]).squeeze(-1)[:, :-1]  # N, T-1

    def step(self, outputs, masks, rewards):
        B, T = outputs.shape
        mask = masks[:, 1:].float()  # B, T-1; 1 on completion tokens only

        # Old log probs (for ratio) and old values (for advantage) — both cached and frozen.
        with torch.no_grad():
            old_log_probs = self._policy_forward(self.model, outputs)
            old_values = self._values(outputs)

            # Per-token reward stream: terminal R at the last completion position, 0 elsewhere.
            last_pos = mask.size(-1) - 1 - mask.flip(-1).argmax(-1)
            r_per_token = torch.zeros_like(mask)
            r_per_token[torch.arange(B, device=outputs.device), last_pos] = rewards

            # GAE backward pass: A_t = δ_t + γλ · A_{t+1}, with δ_t = r_t + γ·V_{t+1} − V_t.
            # next_in_traj zeros out V and gae at the trajectory boundary.
            gamma, lam = 1.0, 0.95
            advantages = torch.zeros_like(mask)
            gae = torch.zeros(B, device=outputs.device)
            Tm1 = mask.size(-1)
            for t in reversed(range(Tm1)):
                next_v = old_values[:, t + 1] if t + 1 < Tm1 else torch.zeros(B, device=outputs.device)
                next_in_traj = mask[:, t + 1] if t + 1 < Tm1 else torch.zeros(B, device=outputs.device)
                delta = r_per_token[:, t] + gamma * next_v * next_in_traj - old_values[:, t]
                gae = delta + gamma * lam * next_in_traj * gae
                advantages[:, t] = gae

            advantages = advantages * mask
            returns = (advantages + old_values) * mask  # λ-blended value target

        value_params = list(self.value_model.parameters()) + list(self.value_head.parameters())
        for _ in range(self.minibatch):
            values = self._values(outputs)
            value_loss = ((values - returns) ** 2 * mask).sum() / mask.sum().clamp(min=1.0)

            log_probs = self._policy_forward(self.model, outputs)
            ratio = torch.exp(log_probs - old_log_probs)
            loss1 = ratio * advantages
            loss2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            policy_loss = -torch.minimum(loss1, loss2)
            policy_loss_mean = (policy_loss * mask).sum() / mask.sum().clamp(min=1.0)

            total_loss = policy_loss_mean + value_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(value_params, max_norm=1.0)
            self.optimizer.step()
        metrics = dict(
            policy_loss=policy_loss.detach().mean(),
            value_loss=value_loss.detach().mean(),
            reward=rewards.detach().mean(),
            ratio=ratio.detach().mean(),
            clipped_frac=(((ratio.detach() < 1 - self.epsilon) | (ratio.detach() > 1 + self.epsilon)) * mask).sum() / mask.sum().clamp(min=1.0),
            advantage_abs=advantages.detach().abs().mean(),
            grad_norm=grad_norm,
        )
        return metrics

ppo = PPO()

for step in range(30):
    batch = random.sample(DATASET, k=1)
    prompts = [system_prompt + p for p, a in batch]
    answers = [a for p, a in batch]
    outputs, masks = ppo._generate(prompts)

    rewards = []
    for i, a in enumerate(answers):
        outputs_text = ppo.tokenizer.decode(outputs[i], skip_special_tokens=True)
        rewards.append(reward_fn(outputs_text, a))

    rewards = torch.tensor(rewards, device=outputs.device)
    metrics = ppo.step(outputs, masks, rewards)

    print('-' * 10, 'step', step, '-' * 10)
    print(outputs_text)
    print(" ".join(f"{k}={v:.2f}" for k, v in metrics.items()))
