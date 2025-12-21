
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
model.requires_grad_(False)
del hf_model
t.cuda.empty_cache()

# uf = dataset = datasets.load_dataset("openbmb/ultrafeedback", split="train")
# ufb = dataset = datasets.load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
dataset = datasets.load_dataset("eekay/ultrafeedback-binarized-balanced", split="train")

#%% example response generation

do_example_generation = False
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
    probe_layer = 30
    probe_act_name = f"blocks.{probe_layer}.hook_resid_pre"
    lr = 3e-4
    batch_size = 16
    epochs = 3

    train_dtype = t.float32
    probe = t.zeros((model.cfg.d_model), dtype=train_dtype, device=DEVICE, requires_grad=True)

    opt = t.optim.AdamW([probe], lr=lr, weight_decay=0.0)

    run_cfg = {"lr":lr, "batch_size":batch_size, "act_name":probe_act_name, "dtype":str(train_dtype)}
    wandb.init(project="reward_probing", config=run_cfg)

    grad_norm = 0.0
    step = 0
    for e in range(epochs):
        for ex in (bar:=tqdm(dataset.shuffle())):
            messages = [{"role":"user","content": ex["prompt"]}, {"role":"assistant","content":ex["response"]}]
            prompt_toks = model.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
            ).squeeze().to(DEVICE)
            seq_len = prompt_toks.shape[0]
            if seq_len >= model.cfg.n_ctx: continue

            score = ex["score"]
            normalized_score = (score / 10.0)

            _, cache = model.run_with_cache(
                prompt_toks,
                stop_at_layer = probe_layer+1,
                names_filter=[probe_act_name]
            )
            act = cache[probe_act_name].squeeze().to(train_dtype)
            last_act = act[-1]

            probe_pred = probe @ last_act
            loss = t.sqrt((normalized_score - probe_pred)**2) / batch_size
            loss.backward()
            
            if (step+1) % batch_size == 0:
                grad_norm = probe.grad.detach().norm().item()
                opt.step()
                opt.zero_grad()

            with t.inference_mode():
                probe_norm = probe.clone().detach().norm().item()
                loss = loss.detach().item() * batch_size
                pred_acc = 1 if round((probe_pred*10).detach().item()) == score else 0
                wandb.log({"loss":loss, "norm": probe_norm,  "acc":pred_acc})
                bar.set_description(f"{orange}[{e}] loss: {loss:.3f}, probe_norm: {probe_norm:.3f} probe grad norm: {grad_norm:.3f}, probe_pred: {pred_acc} {endc}")

            step += 1
            
            t.cuda.empty_cache()
    
    wandb.finish()
    t.cuda.empty_cache()
    

#%%
