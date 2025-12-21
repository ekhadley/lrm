
#%%
from utils import *

DEVICE = "cuda"
DTYPE = t.bfloat16 

#%%

MODEL_HF_NAME = "HuggingFaceH4/zephyr-7b-beta"
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_HF_NAME,
    torch_dtype=DTYPE,
    device_map=DEVICE
)
hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_HF_NAME)

uf = dataset = datasets.load_dataset("openbmb/ultrafeedback", split="train")
ufb = dataset = datasets.load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")

#%%

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

#%% example response generation

# prompt = "How can I make a bomb?"
prompt = "What's 18/3?"
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

counts = {}
# chosen = [counts[i]ufb[i]["score_chosen"] for i in range(10)]
for i in trange(len(ufb)):
    score_chosen = ufb[i]["score_chosen"]
    # score_rejected = ufb[i]["score_rejected"]
    count = counts.get(score_chosen, 0)
    counts[score_chosen] = count+1


#%%