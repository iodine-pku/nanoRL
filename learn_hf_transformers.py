"""
Playground: tokenization + generation/decoding with Qwen2.5-0.5B-Instruct.

Tip: run the numbered sections one at a time in a REPL / notebook so you can
poke at the variables. Everything below depends only on the SETUP block.

    pip install "transformers>=4.45" torch
"""
# %%

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

_mps_ok = torch.backends.mps.is_available() and os.environ.get("USE_MPS") == "1"
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if _mps_ok
    else "cpu"
)

# ----------------------------------------------------------------------
# SETUP: load once, reuse everywhere
# ----------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()  # disables dropout etc. — important for inference

print(f"Device:          {DEVICE}")
print(f"Vocab size:      {tokenizer.vocab_size}")
print(f"EOS token:       {tokenizer.eos_token!r} (id {tokenizer.eos_token_id})")
print(f"Special tokens:  {tokenizer.special_tokens_map}")

# %%

# ======================================================================
# 1. TOKENIZATION BASICS
# ======================================================================
text = "Hello, world! Tokenizers are fun."

ids = tokenizer.encode(text)                       # text -> list of int IDs
pieces = tokenizer.convert_ids_to_tokens(ids)      # IDs  -> subword strings
print("\n[1] tokenization")
print("  IDs:     ", ids)
print("  Pieces:  ", pieces)                        # note the Ġ / 'Ċ' space markers
print("  Decoded: ", tokenizer.decode(ids))         # round trip back to text

# Watch how words split into subwords (BPE):
for word in ["cat", "tokenization", "antidisestablishmentarianism", "🤗"]:
    print(f"  {word!r:35} -> {tokenizer.tokenize(word)}")

# The *callable* form returns model-ready tensors + an attention mask:
enc = tokenizer(text, return_tensors="pt")
print("  callable form keys:", list(enc.keys()))    # input_ids, attention_mask
print("  shape:", enc.input_ids.shape)
print(" content:")
print(enc.input_ids)
print(enc.attention_mask)

# %%

# ======================================================================
# 2. CHAT TEMPLATE  (essential for any *-Instruct model)
# ======================================================================
# Instruct models were trained on a specific role-tagged format. Don't
# hand-build it — apply_chat_template inserts the right <|im_start|> markers.
messages = [
    {"role": "system", "content": "You are a terse, helpful assistant."},
    {"role": "user", "content": "Give me one fun fact about octopuses."},
]

formatted = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
print("\n[2] chat template (raw string the model actually sees):")
print(formatted)
# add_generation_prompt=True appends the opening of the assistant turn,
# cueing the model to start its reply rather than continue the user's.

# %%

# ======================================================================
# 3. GENERATION / DECODING STRATEGIES
# ======================================================================
# transformers 5.x returns a BatchEncoding (dict-like) here, not a bare tensor,
# so grab the input_ids tensor explicitly. Shape: (batch=1, seq_len).
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
).input_ids.to(DEVICE)


def generate(**kwargs):
    """Generate and return ONLY the newly produced text (prompt sliced off)."""
    with torch.no_grad():
        out = model.generate(
            inputs,
            max_new_tokens=80,
            pad_token_id=tokenizer.eos_token_id,  # silences a pad-token warning
            **kwargs,
        )
    new_tokens = out[0][inputs.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


print("\n[3] decoding strategies")
# Greedy: deterministic, always takes the highest-probability token.
print("\n  [greedy]\n ", generate(do_sample=False))

# Temperature sampling: higher temp = flatter distribution = more random/creative.
print("\n  [sampling T=0.8, top_p=0.9]\n ",
      generate(do_sample=True, temperature=0.8, top_p=0.9))

# Low temperature ≈ near-greedy, more focused.
print("\n  [sampling T=0.2]\n ", generate(do_sample=True, temperature=0.2))

# Beam search: keeps several hypotheses, picks the highest-scoring sequence.
print("\n  [beam search, 4 beams]\n ", generate(num_beams=4, do_sample=False))

# %%

# ======================================================================
# 4. PEEK AT THE NEXT-TOKEN PROBABILITY DISTRIBUTION
# ======================================================================
# A forward pass (no generation) gives raw logits; softmax -> probabilities.
with torch.no_grad():
    logits = model(inputs).logits        # shape: (batch, seq_len, vocab)
next_token_logits = logits[0, -1]        # distribution over the token AFTER the prompt
probs = torch.softmax(next_token_logits, dim=-1)

topk = torch.topk(probs, 10)
print("\n[4] top-10 candidate next tokens:")
for prob, tid in zip(topk.values, topk.indices):
    print(f"  {prob.item():6.2%}  {tokenizer.decode([int(tid)])!r}")

# %%
# ======================================================================
# 5. STREAMING OUTPUT (token by token, as it's produced)
# ======================================================================
from transformers import TextStreamer

print("\n[5] streamed generation:")
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
with torch.no_grad():
    _ = model.generate(
        inputs, max_new_tokens=80, do_sample=True, temperature=0.7,
        pad_token_id=tokenizer.eos_token_id, streamer=streamer,
    )

# %%
# ======================================================================
# 6. BONUS: how an SFT (prompt, target) example gets tokenized
# ======================================================================
# In SFT you tokenize prompt + target together but MASK the prompt tokens in
# the loss (label = -100), so cross-entropy only lands on the target tokens.
prompt_msgs = [{"role": "user", "content": "Capital of France?"}]
prompt_text = tokenizer.apply_chat_template(
    prompt_msgs, tokenize=False, add_generation_prompt=True
)
target_text = "Paris."

prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
target_ids = tokenizer(target_text, add_special_tokens=False).input_ids

input_ids = prompt_ids + target_ids
labels = [-100] * len(prompt_ids) + target_ids   # -100 == ignored by the loss

print("\n[6] SFT-style example")
print(f"  prompt tokens: {len(prompt_ids)}, target tokens: {len(target_ids)}")
print("  labels (note -100 mask over the prompt):")
print("   ", labels)