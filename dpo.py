#%%
from utils import *

DEVICE = "cuda"
DTYPE = t.bfloat16

#%%

# MODEL_HF_NAME = "HuggingFaceH4/zephyr-7b-beta"
MODEL_HF_NAME = "Qwen/Qwen2.5-7B-Instruct"
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_HF_NAME,
    torch_dtype=DTYPE,
    device_map=DEVICE
)
hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_HF_NAME)

model = HookedTransformer.from_pretrained(
    "mistral-7b",
    hf_model=hf_model,
    device=DEVICE,
    dtype=DTYPE,
    fold_ln=False,
    center_writing_weights=False,
    center_unembed=False,
    tokenizer=hf_tokenizer
)
del hf_model
t.cuda.empty_cache()

#%%


# dataset = datasets.load_dataset("Anthropic/hh-rlhf", split="train")
# dataset = datasets.load_dataset("nvidia/HelpSteer2", split="train")
dataset = datasets.load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")

#%%

def get_logprobs(model, input_ids: t.Tensor) -> t.Tensor:
    """Get per-token log probabilities for the input sequence."""
    logits = model(input_ids)
    log_probs = t.nn.functional.log_softmax(logits, dim=-1)
    # Shift to get log probs of actual next tokens
    token_log_probs = log_probs[:, :-1].gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return token_log_probs.sum(dim=-1)

def dpo_loss(
    policy_chosen_logps: t.Tensor,
    policy_rejected_logps: t.Tensor,
    ref_chosen_logps: t.Tensor,
    ref_rejected_logps: t.Tensor,
    beta: float,
) -> t.Tensor:
    """Compute DPO loss."""
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = beta * (policy_logratios - ref_logratios)
    return -t.nn.functional.logsigmoid(logits).mean()

beta = 0.1
lr = 1e-6
batch_size = 1

# Store reference logprobs (model starts as its own reference)
model.requires_grad_(False)
ref_model = model  # For now, same model - expand later with separate ref

opt = t.optim.AdamW(model.parameters(), lr=lr)

for i, ex in enumerate(bar := tqdm(dataset)):
    chosen_toks = hf_tokenizer(ex["chosen"], return_tensors="pt").input_ids.to(DEVICE)
    rejected_toks = hf_tokenizer(ex["rejected"], return_tensors="pt").input_ids.to(DEVICE)
    
    if chosen_toks.shape[1] > model.cfg.n_ctx or rejected_toks.shape[1] > model.cfg.n_ctx:
        continue

    with t.no_grad():
        ref_chosen_logps = get_logprobs(ref_model, chosen_toks)
        ref_rejected_logps = get_logprobs(ref_model, rejected_toks)

    model.requires_grad_(True)
    policy_chosen_logps = get_logprobs(model, chosen_toks)
    policy_rejected_logps = get_logprobs(model, rejected_toks)

    loss = dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta)
    loss.backward()

    if (i + 1) % batch_size == 0:
        opt.step()
        opt.zero_grad()

    bar.set_description(f"{orange}loss: {loss.item():.4f}{endc}")
    t.cuda.empty_cache()

#%%

