
#%%
from utils import *

DEVICE = "cuda"
DTYPE = t.bfloat16 
# DTYPE = t.float32 

#%%

MODEL_HF_NAME = "HuggingFaceH4/zephyr-7b-beta"
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

# uf = dataset = datasets.load_dataset("openbmb/ultrafeedback", split="train")
# ufb = dataset = datasets.load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
dataset = datasets.load_dataset("eekay/ultrafeedback-binarized-balanced", split="train")

#%% example response generation

do_example_generation = True
if do_example_generation:
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
    print(model.tokenizer.decode(prompt_toks.squeeze()))

    resp_ids = model.generate(
        prompt_toks,
        do_sample=True,
        verbose=True,
        max_new_tokens=30
    )
    print(model.tokenizer.decode(resp_ids.squeeze()))

#%%

generate_probe_dataset = False
if generate_probe_dataset:
    dataset = make_probe_dataset(balance_ratings=True)
    print(dataset)
    print(len(dataset))
    print(dataset[0])
    print(dataset[1])
    print(dataset[2])

#%%

train_rating_probe = True
if train_rating_probe:
    probe_layer = 25
    probe_act_name = f"blocks.{probe_layer}.hook_resid_pre"

    train_dtype = DTYPE
    probe = t.randn((model.cfg.d_model), dtype=train_dtype, requires_grad=True)
    probe_b = t.randn((1), dtype=train_dtype, requires_grad=True)

    for ex in tqdm(dataset.shuffle()):
        messages = [{"role":"user","content": ex["prompt"]}, {"role":"assistant","content":ex["response"]}]
        prompt_toks = model.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
        ).squeeze().to(DEVICE)
        score = ex["score"]

        _, cache = model.run_with_cache(
            prompt_toks,
            stop_at_layer = probe_layer+1,
            names_filter=[probe_act_name]
        )
        act = cache[probe_act_name].squeeze()
        print(act.shape)

        normalized_score = (score / 10.0)

        break
    
    t.cuda.empty_cache()

#%%

for tok in prompt_toks:
    print(f"{tok}: {repr(model.tokenizer.decode(tok))}")
# %%

asst_special_tok_ids = [523, 28766, 489, 11143, 28766, 28767] # this is how '<|assistant|>' is tokenized. :/
def find_assistant_start(input):
    toks = input.tolist()
    for i in range(len(toks)):
        if toks[i:i+len(asst_special_tok_ids)] == asst_special_tok_ids:
            return i
    else:
        return -1

def to_str_toks(input: str, tokenizer) -> list[int]:
    toks = tokenizer(input)
    str_toks = [model.tokenizer.decode(tok) for tok in toks]

# z = model.tokenizer.decode(prompt_toks)
# i = find_assistant_start(prompt_toks)
# print(i)

print(to_str_toks(z), model.tokenizer)