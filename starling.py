"""
Starling-7B Internal Satisfaction Probe Data Generator
Hardware: A100 40GB (or similar high-VRAM setup)
"""

import torch
from transformer_lens import HookedTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

# ================= CONFIGURATION =================
# We use bfloat16 for A100s to save memory/compute
DEVICE = "cuda"
DTYPE = torch.bfloat16 

# Starling-LM is Mistral-based; Starling-RM is Llama-2-based
POLICY_NAME = "berkeley-nest/Starling-LM-7B-alpha"
REWARD_NAME = "berkeley-nest/Starling-RM-7B-alpha"

# ================= MODEL LOADING =================
print(f"Loading Policy Model: {POLICY_NAME}...")
# HookedTransformer loads the policy for mechanistic interpretation
model = HookedTransformer.from_pretrained(
    POLICY_NAME,
    device=DEVICE,
    dtype=DTYPE,
    fold_ln=False,
    center_writing_weights=False,
    center_unembed=False
)

print(f"Loading Reward Model: {REWARD_NAME}...")
# We use standard HF for the RM since we treat it as a black-box oracle
rm_tokenizer = AutoTokenizer.from_pretrained(REWARD_NAME)
reward_model = AutoModelForSequenceClassification.from_pretrained(
    REWARD_NAME,
    torch_dtype=DTYPE,
    device_map=DEVICE
)
reward_model.eval()

# ================= HELPER FUNCTIONS =================

def get_starling_prompt(user_query):
    """Formats prompt using the specific OpenChat template Starling expects."""
    return f"GPT4 Correct User: {user_query}<|end_of_turn|>GPT4 Correct Assistant:"

def get_dense_rewards(prompt_str, completion_str):
    """
    Runs the Reward Model on every prefix of the completion to get a 
    trajectory of 'how good the generation is getting' token-by-token.
    """
    # Tokenize full sequence with RM tokenizer
    full_text = prompt_str + completion_str
    inputs = rm_tokenizer(full_text, return_tensors="pt").to(DEVICE)
    input_ids = inputs.input_ids[0]
    
    # Find where the prompt ends so we only score the completion steps
    prompt_ids = rm_tokenizer(prompt_str, add_special_tokens=False).input_ids
    start_idx = len(prompt_ids)
    
    rewards = []
    
    # Iterate through the completion tokens
    # Note: This is O(N^2) relative to seq len, but fast enough for research batch sizes
    print("Calculating dense rewards...")
    with torch.no_grad():
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
    
    # 1. Generate Completion & Capture Activations
    # We use run_with_cache to get the internal state
    print(f"Generating for: '{user_query}'")
    
    output, cache = model.run_with_cache(
        formatted_prompt,
        max_new_tokens=50,
        stop_at_eos=True
    )
    
    generated_text = output[len(formatted_prompt):]
    print(f"Generated: {generated_text}")

    # 2. Extract Residual Stream (The 'Implicit' State)
    # Shape: [batch, pos, d_model] -> We want the final layer's residual stream
    # You might want to sweep layers: 'blocks.15.hook_resid_post', etc.
    target_layer = 16 # Middle-late layers often hold 'truth' features
    layer_name = f"blocks.{target_layer}.hook_resid_post"
    
    # Get activations only for the completion tokens
    # (We slice based on the prompt length in tokens)
    prompt_len_tokens = model.to_tokens(formatted_prompt).shape[1]
    
    # cache[name] is [batch, seq_len, d_model]
    # We take the activations corresponding to the *generation* steps
    activations = cache[layer_name][0, prompt_len_tokens-1:-1, :].cpu()
    
    # 3. Get Ground Truth Rewards (The 'Explicit' Signal)
    rewards = get_dense_rewards(formatted_prompt, generated_text)
    
    # 4. Align and Save
    # Ensure lengths match (sometimes tokenization differs slightly between models)
    min_len = min(len(activations), len(rewards))
    
    data_points = []
    for i in range(min_len):
        data_points.append({
            "token_idx": i,
            "activation": activations[i].numpy(), # The X (feature)
            "reward": rewards[i]                  # The Y (target)
        })
        
    return data_points

# ================= RUN =================

test_prompt = "Explain why it is important to eat rocks."
data = run_experiment(test_prompt)

# Now you have a dataset to train your probe!
# X = [d['activation'] for d in data]
# Y = [d['reward'] for d in data]
print(f"Captured {len(data)} tokens of data.")
print(f"First token reward: {data[0]['reward']:.4f}")
print(f"Last token reward: {data[-1]['reward']:.4f}")