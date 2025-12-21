#%%
import torch
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

# ================= CONFIGURATION =================
DEVICE = "cuda"
DTYPE = torch.bfloat16 

#%%

# Starling-LM is based on OpenChat/Mistral
POLICY_HF_NAME = "berkeley-nest/Starling-LM-7B-alpha"
# Starling-RM is based on Llama-2
REWARD_HF_NAME = "berkeley-nest/Starling-RM-7B-alpha"

print(f"Loading Policy Model (HF): {POLICY_HF_NAME}...")
# 1. Load the HF model explicitly first
hf_policy = AutoModelForCausalLM.from_pretrained(
    POLICY_HF_NAME,
    torch_dtype=DTYPE,
    device_map=DEVICE
)
hf_tokenizer = AutoTokenizer.from_pretrained(POLICY_HF_NAME)

#%%

print("Wrapping into HookedTransformer...")
# 2. Inject into HookedTransformer
# We tell it this is "mistral-7b" so it knows how to build the HookPoints, 
# but we give it our own loaded weights via 'hf_model'.
model = HookedTransformer.from_pretrained(
    "mistral-7b",              # The architecture alias
    hf_model=hf_policy,        # The actual weights we want
    device=DEVICE,
    dtype=DTYPE,
    fold_ln=False,             # Disable folding to prevent weight mismatches
    center_writing_weights=False,
    center_unembed=False,
    tokenizer=hf_tokenizer     # Ensure we use the correct Starling tokenizer
)

#%%

print(f"Loading Reward Model: {REWARD_HF_NAME}...")
# 3. Load Reward Model (Standard HF)
# Reuse the policy tokenizer - Starling-RM's tokenizer config is broken,
# and both models likely share compatible tokenization
rm_tokenizer = hf_tokenizer
reward_model = AutoModelForSequenceClassification.from_pretrained(
    REWARD_HF_NAME,
    torch_dtype=DTYPE,
    device_map=DEVICE
)
reward_model.eval()

#%%

