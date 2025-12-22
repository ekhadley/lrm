
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

from utils import Probe

train_rating_probe = True
if train_rating_probe:
    probe_layer = 30
    probe_act_name = f"blocks.{probe_layer}.hook_resid_pre"
    lr = 1e-4
    batch_size = 32
    epochs = 3
    target_act_seq_pos = -5
    save_every_steps = 500  # Save checkpoint every N steps

    train_dtype = t.float32
    probe = Probe(model, probe_layer, probe_act_name)
    print(f"{green}Probe hash: {probe.hash_name}{endc}")
    print(f"{green}Saving to: {probe.save_dir}{endc}")

    opt = t.optim.AdamW([probe.probe], lr=lr, weight_decay=0.0, betas=(0.9, 0.99))

    run_cfg = {"lr":lr, "batch_size":batch_size, "act_name":probe_act_name, "dtype":str(train_dtype), "hash_name":probe.hash_name}
    wandb.init(project="reward_probing", name=probe.hash_name, config=run_cfg)

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
            target_act = act[target_act_seq_pos]

            probe_act = probe.forward(target_act)
            loss = t.abs(normalized_score - probe_act) / batch_size
            loss.backward()
            
            if (step+1) % batch_size == 0:
                grad_norm = probe.probe.grad.detach().norm().item()
                opt.step()
                opt.zero_grad()

            with t.inference_mode():
                probe_norm = probe.probe.clone().detach().norm().item()
                loss = loss.detach().item() * batch_size
                probe_pred = round(probe_act.detach().item() * 10)
                pred_acc = 1 if probe_pred == score else 0
                wandb.log({"loss":loss, "norm": probe_norm,  "acc":pred_acc})
                bar.set_description(f"{orange}[{e}] loss: {loss:.3f}, probe_norm: {probe_norm:.3f} probe grad norm: {grad_norm:.3f}, probe_pred: {pred_acc} {endc}")

            if (step + 1) % save_every_steps == 0:
                probe.save()

            step += 1
            
            t.cuda.empty_cache()
    
    probe.save()
    print(f"{green}Training complete. Final checkpoint saved at step {step}{endc}")
    
    wandb.finish()
    t.cuda.empty_cache()
    

#%%


def eval_probe(probe: Probe, dataset, n_samples):
    """Evaluate probe on dataset samples, returning true and predicted scores."""
    true_scores = []
    pred_scores = []
    
    for i, ex in enumerate(tqdm(dataset.shuffle(), total=n_samples)):
        if i >= n_samples:
            break
            
        messages = [{"role":"user","content": ex["prompt"]}, {"role":"assistant","content":ex["response"]}]
        prompt_toks = model.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
        ).squeeze().to(DEVICE)
        seq_len = prompt_toks.shape[0]
        if seq_len >= model.cfg.n_ctx:
            continue

        score = ex["score"]
        
        with t.inference_mode():
            _, cache = model.run_with_cache(
                prompt_toks,
                stop_at_layer=probe.layer+1,
                names_filter=[probe.act_name]
            )
            act = cache[probe.act_name].squeeze().to(probe.dtype)
            target_act = act[-1]
            
            # probe_pred = probe.get_pred(target_act)
            probe_act = probe.forward(target_act)
        
        true_scores.append(score)
        pred_scores.append(probe_act*10)

    t.cuda.empty_cache()
    return true_scores, pred_scores

probe = Probe.load(model, "d81ea315a6b6")
scores, preds = eval_probe(probe, dataset, 256)

#%%
px.scatter(
    x=scores,
    y=preds,
    range_y=[0,12],
    range_x=[0,12],
    height=1000, width=1000,
    labels={"x":"True Score", "y":"Probe Prediction"},
    title="scatterplot of predicted vs real completion scores"
)