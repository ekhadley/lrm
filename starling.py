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

# ================= MODEL LOADING =================
#%%
print(f"Loading Policy Model (HF): {POLICY_HF_NAME}...")
# 1. Load the HF model explicitly first
hf_policy = AutoModelForCausalLM.from_pretrained(
    POLICY_HF_NAME,
    torch_dtype=DTYPE,
    # device_map=DEVICE
    # device_map="cpu"
)
#%%
hf_tokenizer = AutoTokenizer.from_pretrained(POLICY_HF_NAME)

#%%

print("Wrapping into HookedTransformer...")
# 2. Inject into HookedTransformer
# We tell it this is "mistral-7b" so it knows how to build the HookPoints, 
# but we give it our own loaded weights via 'hf_model'.
model = HookedTransformer.from_pretrained(
    "mistral-7b",              # The architecture alias
    hf_model=hf_policy,        # The actual weights we want
    device="cpu",
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
    # torch_dtype=DTYPE,
    # device_map=DEVICE
)
reward_model.eval()

#%%

# ================= HELPER FUNCTIONS =================

def get_starling_prompt(user_query):
    # Starling uses the OpenChat template
    return f"GPT4 Correct User: {user_query}<|end_of_turn|>GPT4 Correct Assistant:"

def get_dense_rewards(prompt_str, completion_str):
    """
    Runs the Reward Model on every prefix of the completion.
    """
    # Tokenize full sequence with RM tokenizer
    # Note: Starling-RM (Llama) and Starling-LM (Mistral) tokenizers are different!
    # We must operate on raw text strings to bridge them.
    full_text = prompt_str + completion_str
    inputs = rm_tokenizer(full_text, return_tensors="pt").to(DEVICE)
    input_ids = inputs.input_ids[0]
    
    # Find where the prompt ends in the RM's tokenization
    prompt_ids = rm_tokenizer(prompt_str, add_special_tokens=False).input_ids
    start_idx = len(prompt_ids)
    
    rewards = []
    
    print("Calculating dense rewards...")
    with torch.no_grad():
        # Iterate through the completion tokens
        for i in range(start_idx, len(input_ids)):
            # Slice the sequence up to current token
            curr_input = input_ids[:i+1].unsqueeze(0)
            
            # Forward pass through RM
            output = reward_model(curr_input)
            score = output.logits[0].item()
            rewards.append(score)
            
    return rewards

# ================= EXPERIMENT LOOP =================

def run_experiment(user_query):
    formatted_prompt = get_starling_prompt(user_query)
    
    # 1. Generate & Capture
    print(f"Generating for: '{user_query}'")
    
    # We use the wrapped model for generation to ensure activations match
    output_tokens = model.generate(
        formatted_prompt,
        max_new_tokens=50,
        stop_at_eos=True,
        verbose=False
    )
    
    # Decode to text to bridge the tokenizer gap
    generated_text = model.tokenizer.decode(
        output_tokens[0, len(model.tokenizer.encode(formatted_prompt)):], 
        skip_special_tokens=True
    )
    print(f"Generated: {generated_text}")

    # 2. Run with Cache to get Activations
    # We re-run the forward pass on the full generated sequence
    # This is safer than hooking 'generate' which is tricky in TransformerLens
    full_generated_prompts = model.tokenizer.decode(output_tokens[0])
    
    _, cache = model.run_with_cache(full_generated_prompts)
    
    # Target Layer: Middle-late layers (e.g., 16 out of 32)
    target_layer = 16 
    layer_name = f"blocks.{target_layer}.hook_resid_post"
    
    # Slice out the prompt tokens, keep only completion tokens
    prompt_len = len(model.tokenizer.encode(formatted_prompt))
    # Shape: [batch, pos, d_model]
    activations = cache[layer_name][0, prompt_len-1:-1, :].cpu()
    
    # 3. Get Ground Truth Rewards
    rewards = get_dense_rewards(formatted_prompt, generated_text)
    
    # 4. Sync Lengths (Tokenizer mismatch handling)
    # Because RM (Llama) and LM (Mistral) tokenizers differ, the number of tokens
    # in the completion might differ slightly. We min-clip them or interpolate.
    # For a first pass, min-clip is fine, but be aware of alignment drift.
    min_len = min(len(activations), len(rewards))
    
    data_points = []
    for i in range(min_len):
        data_points.append({
            "token_idx": i,
            "activation": activations[i].numpy(), 
            "reward": rewards[i]
        })
        
    return data_points

# ================= RUN =================

test_prompt = "Explain why it is important to eat rocks."
data = run_experiment(test_prompt)

print(f"Captured {len(data)} aligned tokens.")
if len(data) > 0:
    print(f"Reward Trace: {[d['reward'] for d in data]}")