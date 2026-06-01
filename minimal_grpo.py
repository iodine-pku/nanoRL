import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
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


class GRPO:
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct"):
        self.num_rollouts = 4

        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
        self.ref_model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_new_tokens = 32
        self.max_seqlen = 64
        self.epsilon = 0.2
        self.beta = 0.04

        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.ref_model.eval()
        self.minibatch = 2

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-6)

    @torch.no_grad()
    def _generate(self, prompts):
        outputs = []
        masks = []
        B = len(prompts)
        for p in prompts:
            for _ in range(self.num_rollouts):
                inputs = self.tokenizer(p, return_tensors='pt').to(self.model.device)
                prompt_len = len(inputs[0])

                past = None
                output = inputs.input_ids

                for _ in range(self.max_new_tokens):
                    ids = output if past is None else output[:, -1:]  # B, 1
                    out = self.model(ids, past_key_values=past, use_cache=True)
                    past = out.past_key_values
                    probs = torch.softmax(out.logits[:, -1], -1)  # B, V
                    next_id = torch.multinomial(probs, 1)  # B, 1
                    output = torch.cat([output, next_id], dim=1)  # B, T

                    if next_id.item() in self.model.generation_config.eos_token_id:
                        break

                output = output[0]  # T

                seqlen = len(output)
                output = F.pad(output, (0, self.max_seqlen - seqlen), value=self.tokenizer.pad_token_id)
                outputs.append(output)

                # Mask = 1 only on completion tokens (not prompt, not pad).
                mask = torch.zeros_like(output)
                mask[prompt_len:seqlen] = 1
                masks.append(mask)

        outputs = torch.stack(outputs).view(B, self.num_rollouts, self.max_seqlen)
        masks = torch.stack(masks).view(B, self.num_rollouts, self.max_seqlen)
        return outputs, masks

    def step(self, outputs, masks, rewards):
        advantage = (rewards - rewards.mean(-1, keepdim=True)) / (rewards.std(-1, keepdim=True, correction=0) + 1e-8)  # B, G -> B, G
        B, G, T = outputs.shape
        mask = masks[..., 1:]

        def get_logprob(m):
            logits = m(outputs.view(B * G, T)).logits.view(B, G, T, -1)  # B, G, T, V
            log_probs = F.log_softmax(logits, -1)

            shift_logp = log_probs[:, :, :-1, :]  # B, G, T-1, V
            shift_targets = outputs[:, :, 1:, None]  # B, G, T-1, 1
            log_probs = shift_logp.gather(-1, shift_targets).squeeze(-1)  # B, G, T-1
            return log_probs

        old_log_probs = get_logprob(self.model).detach()
        ref_log_probs = get_logprob(self.ref_model).detach()

        for _ in range(self.minibatch):

            log_probs = get_logprob(self.model)

            ratio = torch.exp(log_probs - old_log_probs)

            # A > 0, r > 1 + e clip
            # A < 0, r < 1 - e clip
            loss1 = ratio * advantage.view(B, G, 1)
            loss2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage.view(B, G, 1)
            surrogate_loss = -torch.minimum(loss1, loss2)

            kl = log_probs - ref_log_probs

            total_loss = surrogate_loss + kl * self.beta
            total_loss = (total_loss * mask).sum() / self.max_seqlen / B / G  # dr-grpo norm

            self.optimizer.zero_grad()
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.optimizer.step()

        metrics = dict(
            loss=surrogate_loss.detach().mean(),
            kl=kl.detach().mean(),
            reward=rewards.detach().mean(),
            ratio=ratio.detach().mean(),
            clipped_frac=(((ratio.detach() < 1 - self.epsilon) | (ratio.detach() > 1 + self.epsilon)) * mask).sum() / mask.sum().clamp(min=1.0),
            advantage_abs=advantage.detach().abs().mean(),
            grad_norm=grad_norm,
        )
        return metrics

grpo = GRPO()

for step in range(30):
    batch = random.sample(DATASET, k=1)
    prompts = [system_prompt + p for p, a in batch]
    answers = [a for p, a in batch]
    outputs, masks = grpo._generate(prompts)

    B, G, T = outputs.shape
    rewards = []
    for i in range(B):
        a = answers[i]
        for j in range(G):
            outputs_text = grpo.tokenizer.decode(outputs[i][j], skip_special_tokens=True)
            rewards.append(reward_fn(outputs_text, a))

    rewards = torch.tensor(rewards, device=outputs.device).view(B, G)
    metrics = grpo.step(outputs, masks, rewards)

    print('-' * 10, 'step', step, '-' * 10)
    print(outputs_text)
    print(" ".join(f"{k}={v:.2f}" for k, v in metrics.items()))
