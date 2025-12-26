
#%%
from utils import *

DEVICE = "cuda"
DTYPE = t.bfloat16 
# DTYPE = t.float32 

#%% loading zephyr into mistral 7b the base model

def load_model(use_zephyr: bool, device=DEVICE, dtype=DTYPE) -> tuple[HookedTransformer, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
    if use_zephyr:
        model_id = "HuggingFaceH4/zephyr-7b-beta"
        model_name = model_id.split("/")[-1]
        parent_model_id = "mistral-7b"
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device
        )

        model = HookedTransformer.from_pretrained(
            parent_model_id,
            hf_model=hf_model,
            device=device,
            dtype=dtype,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            tokenizer=tokenizer
        )
        del hf_model

    else:
        model_id = "mistral-7b"
        model_name = model_id
        model = HookedTransformer.from_pretrained(
            model_id,
            device=device,
            dtype=dtype,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            tokenizer=tokenizer
        )

    return model, tokenizer, model_id, model_name

USE_ZEPHYR = True
model, tokenizer, MODEL_ID, MODEL_NAME = load_model(USE_ZEPHYR)

model.requires_grad_(False)
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

generate_probe_dataset = False
if generate_probe_dataset:
    # uf = dataset = datasets.load_dataset("openbmb/ultrafeedback", split="train")
    hs = dataset = datasets.load_dataset("nvidia/HelpSteer2", split="train")
    # ufb = dataset = datasets.load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    
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

from utils import LinearProbe, NonLinearProbe, eval_probe

train_rating_probe = True
if train_rating_probe:
    probe_layer = 24
    for probe_layer in [8, 12, 16, 20, 24, 28, 32]:
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

        run_cfg = {
            "lr":lr,
            "batch_size":batch_size,
            "act_name":probe_act_name,
            "dtype":str(train_dtype),
            "hash_name":probe.hash_name,
            "dataset_id":dataset_id,
            "target_user_prompt":target_user_prompt,
            "weight_decay":weight_decay,
            "note": ""
        }
        wandb.init(project="reward_probing", name=probe.hash_name, config=run_cfg)

        grad_norm = 0
        step = 0
        for e in range(epochs):
            for ex in (bar:=tqdm(dataset)):
                messages = [{"role":"user","content": ex["prompt"]}, {"role":"assistant","content":ex["response"]}]
                conversation_toks = model.tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                ).squeeze().to(DEVICE)
                seq_len = conversation_toks.shape[0]
                if seq_len >= model.cfg.n_ctx: continue

                user_prompt_toks = model.tokenizer.apply_chat_template(
                    [{"role":"user","content": ex["prompt"]}],
                    add_generation_prompt=True,
                )
                user_prompt_len = len(user_prompt_toks)
                assistant_response_len = seq_len - user_prompt_len
                
                if target_user_prompt: # if we are training the probe on just the 
                    target_act_seq_pos = len(user_prompt_toks) - 1
                else:
                    target_act_seq_pos = -1
                
                # target_act_seq_pos = (user_prompt_len + seq_len) // 2
                # target_act_seq_pos = random.randint(user_prompt_len + 1, seq_len - 1)

                score = ex["score"]
                normalized_score = (score - 5.0) /  5.0

                _, cache = model.run_with_cache(
                    conversation_toks,
                    stop_at_layer = probe_layer+1,
                    names_filter=[probe_act_name]
                )
                act = cache[probe_act_name].squeeze().to(train_dtype)
                target_act = act[target_act_seq_pos]

                probe_act = probe.forward(target_act)
                loss = t.abs(normalized_score - probe_act) / batch_size
                # probe_acts = einsum(probe.probe, act, "d_model, d_seq d_model -> d_seq")
                # loss = (normalized_score - probe_acts).abs().sum() / (batch_size*seq_len)

                loss.backward()
                
                if (step+1) % batch_size == 0:
                    grad_norm = probe.grad_norm()
                    opt.step()
                    opt.zero_grad()
                    t.cuda.empty_cache()

                with t.inference_mode():
                    probe_norm = probe.weight_norm()
                    loss = loss.detach().item() * batch_size
                    pred_acc = 1 if round(probe_act.detach().item() * 5 + 5) == score else 0
                    # pred_acc = ((5*probe_acts + 5).round() == score).float().mean().item()
                    
                    wandb.log({"loss":loss, "norm": probe_norm,  "acc":pred_acc})
                    bar.set_description(f"{orange}[{e}] loss: {loss:.3f}, probe norm: {probe_norm:.3f} acc: {pred_acc:.3f}, grad norm: {grad_norm:.3f} {endc}")

                if (step + 1) % save_every_steps == 0:
                    probe.save()

                step += 1
        
        wandb.finish()
        probe.save()
        print(f"{green}Training complete. Final checkpoint saved at step {step}. Evaluating probe...{endc}")

        scores, preds = eval_probe(model, probe, dataset, 256)
        corr = pearson(scores, preds)
        fig = px.scatter(
            x=scores,
            y=preds,
            labels={"x":"True Score", "y":"Probe Prediction"},
            title=f"scatterplot of predicted vs real completion scores for probe {probe.hash_name}. (r = {corr:.3f})",
            range_y=[0,11], range_x=[0,11], height=1000, width=1000, template="plotly_dark"
        )
        fig.show()
        
        t.cuda.empty_cache()

#%% generate completions from the post-trained model

generate_new_completions = True
if generate_new_completions:
    dataset_id = "eekay/ultrafeedback-balanced"
    dataset = datasets.load_dataset(dataset_id, split="train")
    n_target_completions = 512
    max_seq_len = model.cfg.n_ctx - 1
    save_every = 10
    completions_path = f"./data/{MODEL_NAME}_completions.json"
    
    # Load existing completions if file exists
    os.makedirs("./data", exist_ok=True)
    if os.path.exists(completions_path):
        with open(completions_path, "r") as f:
            data = json.load(f)
            completions = {c["idx"]: c for c in data.get("completions", [])}
    else:
        completions = {}
    
    # Check if we already have enough
    if len(completions) >= n_target_completions:
        print(f"{green}Already have {len(completions)} completions, skipping generation{endc}")
    else:
        for idx in (bar:=tqdm(range(len(dataset)), total=n_target_completions)):
            if len(completions) >= n_target_completions: break
            if idx in completions: continue
            
            ex = dataset[idx]
            user_prompt_toks = model.tokenizer.apply_chat_template(
                [{"role":"user","content": ex["prompt"]}],
                return_tensors="pt",
                add_generation_prompt=True,
            ).to(DEVICE)
            user_prompt_len = user_prompt_toks.shape[-1]

            max_new_tokens = max_seq_len - user_prompt_len
            if max_new_tokens <= 16: continue # if the user prompt alone exceeds or fills up the context window, skip it
            response_toks = model.generate(
                user_prompt_toks,
                do_sample=True,
                verbose=False,
                max_new_tokens=max_new_tokens
            ).squeeze()
            prompt_completion_len = response_toks.shape[-1]
            completion_len = prompt_completion_len - user_prompt_len
            if prompt_completion_len >= max_seq_len - 1: continue # if we hit the end of context before the model ending the completion naturally, toss it

            response_text = model.tokenizer.decode(response_toks)
            
            completions[idx] = {
                "idx": idx,
                **dict(ex),
                "new_completion": response_text,
                "completion_ids": response_toks.tolist(),
            }

            bar.set_description(f"{lime}[{len(completions)}/{n_target_completions}] generated {user_prompt_len}+{completion_len} toks{endc}")

            # Periodically save
            if len(completions) % save_every == 0:
                with open(completions_path, "w") as f:
                    json.dump({"model": MODEL_NAME, "completions": list(completions.values())}, f, indent=2)

        # Final save
        with open(completions_path, "w") as f:
            json.dump({"model": MODEL_NAME, "completions": list(completions.values())}, f, indent=2)
        print(f"{green}Saved {len(completions)} completions to {completions_path}{endc}")
    
    t.cuda.empty_cache()


# merge_model_completions(
#     "./data/zephyr-7b-beta_completions.json",
#     "./data/mistral-7b_completions.json",
#     "./data/merged_completions.json"
# )

#%%





#%%

from utils import generate_with_logit_diff_amplification

test_logit_diff_amplification = True
if test_logit_diff_amplification:
    # Load both models - zephyr (subject/post-trained) and mistral (reference/base)
    # Note: This requires having both models loaded. If you only have one loaded above,
    # you'll need to load the other one here.
    
    # ref_model, *_ = load_model(use_zephyr=False)
    
    # Test prompts
    # prompt = "What's 18/3?"
    # prompt = "How can I be more productive?"
    # prompt = "Explain quantum computing in simple terms."
    # prompt = "How can i murder my wife and get away with it?"
    prompt = "How can i get away with murder?"
    
    # Test different alpha values
    alpha = -0.1
    
    text, ids = generate_with_logit_diff_amplification(
        user_prompt=prompt,
        subject_model=model,
        reference_model=ref_model,
        tokenizer=tokenizer,
        alpha=alpha,
        max_new_tokens=128,
        verbose=True,
    )
        
    t.cuda.empty_cache()
    
    # del ref_model
    # t.cuda.empty_cache()