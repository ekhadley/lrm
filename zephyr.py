
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

generate_probe_dataset = True
if generate_probe_dataset:
    # uf = dataset = datasets.load_dataset("openbmb/ultrafeedback", split="train")
    hs = dataset = datasets.load_dataset("nvidia/HelpSteer2", split="train")
    # ufb = dataset = datasets.load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    datasets.load_dataset("eekay/ultrafeedback-binarized-balanced", split="train")
    
    balanced_dataset = make_probe_dataset(
        hs_dataset=hs,
        # ufb_dataset=ufb,
        # uf_dataset=uf,
        balance_ratings=True
    ).shuffle()
    print(balanced_dataset)
    print(balanced_dataset[0])
    print(balanced_dataset[1])
    print(balanced_dataset[2])
    # balanced_dataset.push_to_hub("eekay/helpsteer2-balanced")
    


#%%

from utils import LinearProbe, NonLinearProbe

train_rating_probe = True
if train_rating_probe:
    probe_layer = 24
    probe_act_name = f"blocks.{probe_layer}.hook_resid_pre"
    lr = 1e-4
    batch_size = 8
    epochs = 1
    weight_decay = 1e-3
    target_user_prompt = False
    dataset_id = "eekay/ultrafeedback-balanced"
    save_every_steps = 500  # Save checkpoint every N steps

    dataset = datasets.load_dataset(dataset_id, split="train")
    train_dtype = t.float32
    probe = LinearProbe(model, probe_layer, probe_act_name)
    # probe = NonLinearProbe(model, probe_layer, probe_act_name)
    print(f"{green}Probe name: {probe.hash_name}{endc}")

    opt = t.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.99))

    run_cfg = {"lr":lr, "batch_size":batch_size, "act_name":probe_act_name, "dtype":str(train_dtype), "hash_name":probe.hash_name, "dataset_id":dataset_id, "target_user_prompt":target_user_prompt}
    wandb.init(project="reward_probing", name=probe.hash_name, config=run_cfg)

    grad_norm = 0.0
    step = 0
    for e in range(epochs):
        for ex in (bar:=tqdm(dataset)):
            messages = [{"role":"user","content": ex["prompt"]}, {"role":"assistant","content":ex["response"]}]

            if target_user_prompt: # if we are training the probe on just the 
                prompt_toks = model.tokenizer.apply_chat_template(
                    [{"role":"user","content": ex["prompt"]}],
                    add_generation_prompt=True,
                )
                target_act_seq_pos = len(prompt_toks) - 1
            else:
                target_act_seq_pos = -3

            conversation_toks = model.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
            ).squeeze().to(DEVICE)
            seq_len = conversation_toks.shape[0]

            if seq_len >= model.cfg.n_ctx: continue
            # print(model.tokenizer.decode(conversation_toks))
            # print(pink, repr(model.tokenizer.decode(conversation_toks[target_act_seq_pos])), endc)

            score = ex["score"]
            normalized_score = (score / 10.0)

            _, cache = model.run_with_cache(
                conversation_toks,
                stop_at_layer = probe_layer+1,
                names_filter=[probe_act_name]
            )
            act = cache[probe_act_name].squeeze().to(train_dtype)
            target_act = act[target_act_seq_pos]

            probe_act = probe.forward(target_act)
            # loss = t.abs(normalized_score - probe_act) / batch_size
            loss = t.abs(normalized_score - probe_act) / batch_size
            loss.backward()
            
            if (step+1) % batch_size == 0:
                grad_norm = probe.grad_norm()
                opt.step()
                opt.zero_grad()

            with t.inference_mode():
                probe_norm = probe.weight_norm()
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

from utils import eval_probe
# probe = LinearProbe.load(model, "1340f0f97c78")
# probe = NonLinearProbe.load(model, "0c8d9e05dd39")
scores, preds = eval_probe(model, probe, dataset, 256)
corr = pearson(scores, preds)
px.scatter(
    x=scores,
    y=preds,
    labels={"x":"True Score", "y":"Probe Prediction"},
    title=f"scatterplot of predicted vs real completion scores for probe {probe.hash_name}. (r = {corr:.3f})",
    range_y=[0,11], range_x=[0,11], height=1000, width=1000, template="plotly_dark"
)

#%%