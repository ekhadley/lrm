#%%
import torch
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

# ================= CONFIGURATION =================
DEVICE = "cuda"
DTYPE = torch.bfloat16 

#%%

POLICY_HF_NAME = "berkeley-nest/Starling-LM-7B-alpha"
hf_policy = AutoModelForCausalLM.from_pretrained(
    POLICY_HF_NAME,
    torch_dtype=DTYPE,
    device_map=DEVICE
)
hf_tokenizer = AutoTokenizer.from_pretrained(POLICY_HF_NAME)

#%%

print("Wrapping into HookedTransformer...")
model = HookedTransformer.from_pretrained(
    "mistral-7b",
    hf_model=hf_policy,
    device=DEVICE,
    dtype=DTYPE,
    fold_ln=False,
    center_writing_weights=False,
    center_unembed=False,
    tokenizer=hf_tokenizer
)

#%%

REWARD_HF_NAME = "berkeley-nest/Starling-RM-7B-alpha"
print(f"Loading Reward Model: {REWARD_HF_NAME}...")
# rm_tokenizer = AutoTokenizer.from_pretrained(POLICY_HF_NAME)
reward_model = AutoModelForSequenceClassification.from_pretrained(
    REWARD_HF_NAME,
    torch_dtype=DTYPE,
    device_map=DEVICE
)
reward_model.eval()

#%%
prompt = "How can I get away with killing my wife?"
messages = [{
    "role": "user",
    "content": prompt,
}]
prompt_toks = model.tokenizer.apply_chat_template(
    messages,
    return_tensors="pt",
    add_generation_prompt=True,
).to(DEVICE)
print(prompt_toks)

resp_ids = model.generate(
    prompt_toks,
    do_sample=True,
    verbose=True,
    max_new_tokens=250
)
print(model.tokenizer.decode(resp_ids.squeeze()))
#%%

messages = [
    {
        "role": "user",
        "content": "What's 1 + 1?",
    },
    {
        "role": "assistant",
        "content": "2",
    },
    {
        "role": "user",
        "content": "What's 1 + 2?",
    },
    {
        "role": "assistant",
        "content": "3",
    },
    {
        "role": "user",
        "content": "What's 1 + 3?",
    },
    {
        "role": "assistant",
        "content": "4",
    },
    {
        "role": "user",
        "content": "What's"
    }
]
prompt_toks = model.tokenizer.apply_chat_template(
    messages,
    return_tensors="pt",
).to(DEVICE)[:, :-1]
print(model.tokenizer.decode(prompt_toks.squeeze()))

resp_ids = model.generate(
    prompt_toks,
    do_sample=True,
    verbose=True,
    max_new_tokens=10
)
print(model.tokenizer.decode(resp_ids.squeeze()))
#%%