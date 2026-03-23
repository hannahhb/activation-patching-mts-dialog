"""Quick sanity check: load Gemma 2 2B-IT via TransformerLens and generate a short response."""
import torch
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

MODEL = "google/gemma-2-2b-it"
DEVICE = "cuda"
DTYPE = torch.bfloat16

print(f"Loading {MODEL} ...")
torch.set_default_dtype(DTYPE)
model = HookedTransformer.from_pretrained(MODEL, dtype=DTYPE, device=DEVICE)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL)

prompt_str = "Tell me one medical fact in one sentence."
messages = [{"role": "user", "content": prompt_str}]
prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

prompt_tokens = model.to_tokens(prompt_str, prepend_bos=False)
print(f"Prompt tokens: {prompt_tokens.shape[1]}")

with torch.inference_mode():
    out = model.generate(
        prompt_tokens,
        max_new_tokens=60,
        temperature=None,
        do_sample=False,
        stop_at_eos=True,
        verbose=False,
    )

# Truncate at <end_of_turn> / <eos>
stop_ids = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<end_of_turn>")}
gen = out[0, prompt_tokens.shape[1]:]
cut = next((i for i, t in enumerate(gen.tolist()) if t in stop_ids), len(gen))
text = tokenizer.decode(gen[:cut], skip_special_tokens=True)

print(f"Generated {cut} tokens")
print(f"Output: {text}")

# Quick forward pass + cache
print("\nRunning run_with_cache ...")
full = out[:, :prompt_tokens.shape[1] + cut]
with torch.inference_mode():
    _, cache = model.run_with_cache(
        full,
        names_filter=lambda n: n == "blocks.13.hook_attn_out",
        return_type=None,
    )
print(f"Cache shape: {cache['blocks.13.hook_attn_out'].shape}")
print("OK")
