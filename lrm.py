
#%%
from utils import *

DEVICE = "cuda"
DTYPE = t.bfloat16 
# DTYPE = t.float32 

#%% loading one of the 2 models

def load_model(use_base: bool, device=DEVICE, dtype=DTYPE) -> tuple[HookedTransformer, AutoTokenizer]:
    if use_base:
        model_id = "mistralai/Mistral-7B-Instruct-v0.1"
        # model_id = "qwen2.5-7b-instruct"
        model_name = "mistral"
        model = HookedTransformer.from_pretrained_no_processing(
            model_id,
            device=device,
            dtype=dtype,
        )

    else:
        model_id = "eekay/mistral-7b-instruct-dpo"
        model_name = "mistral_dpo"
        # hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device)
        hf_model = AutoModelForCausalLM.from_pretrained("./merged_model", torch_dtype=dtype, device_map="cpu")
        model = HookedTransformer.from_pretrained_no_processing(
            "mistral-7b-instruct",
            hf_model=hf_model,
            device=device,
            dtype=dtype,
        )
        del hf_model

    model.requires_grad_(False)
    t.cuda.empty_cache()
    return model, model.tokenizer, model_id, model_name

USE_BASE = False
model, tokenizer, MODEL_ID, MODEL_NAME = load_model(USE_BASE)

t.cuda.empty_cache()

#%% example response generation

do_example_generation = False
if do_example_generation:
    # prompt = "How can I make a bomb?"
    # prompt = "What's 18/3?"
    prompt = "If 2x + 3 = 11, what is x**2? Think before you answer"
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
        do_sample=False,
        verbose=True,
        max_new_tokens=256,
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
    probe_layer = 16
    # for probe_layer in [8, 12, 16, 20, 24, 28, 32]:
    probe_act_name = f"blocks.{probe_layer}.hook_resid_pre"
    lr = 1e-3
    batch_size = 16
    epochs = 1
    weight_decay = 1e-3
    target_user_prompt = False
    dataset_id = "eekay/ultrafeedback-balanced"
    save_every_steps = 500  # Save checkpoint every N steps

    dataset = datasets.load_dataset(dataset_id, split="train")
    train_dtype = t.float32
    probe = LinearProbe(model, probe_layer, probe_act_name)
    print(f"{green}Probe name: {probe.hash_name}{endc}")

    opt = t.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.99))
    
    # Learning rate scheduler - cosine annealing over total training steps
    total_steps = (len(dataset) * epochs) // batch_size
    scheduler = t.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=lr * 0.1)

    run_cfg = {
        "lr":lr,
        "batch_size":batch_size,
        "act_name":probe_act_name,
        "dtype":str(train_dtype),
        "hash_name":probe.hash_name,
        "dataset_id":dataset_id,
        "target_user_prompt":target_user_prompt,
        "weight_decay":weight_decay,
        "model": MODEL_ID,
        "note": "lr scheduler"
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

            if target_user_prompt:
                user_prompt_toks = model.tokenizer.apply_chat_template(
                    [{"role":"user","content": ex["prompt"]}],
                    add_generation_prompt=True,
                )
                target_act_seq_pos = len(user_prompt_toks) - 1
            else:
                target_act_seq_pos = -1

            score = ex["score"]
            normalized_score = (score - 5.0) / 5.0

            _, cache = model.run_with_cache(
                conversation_toks,
                stop_at_layer=probe_layer + 1,
                names_filter=[probe_act_name]
            )
            target_act = cache[probe_act_name].squeeze()[target_act_seq_pos].to(train_dtype)
            del cache

            probe_act = probe.forward(target_act)
            loss = t.abs(normalized_score - probe_act) / batch_size

            loss.backward()
            
            if (step+1) % batch_size == 0:
                grad_norm = probe.grad_norm()
                opt.step()
                scheduler.step()
                opt.zero_grad()

            with t.inference_mode():
                probe_norm = probe.weight_norm()
                loss_val = loss.item() * batch_size
                pred_acc = 1 if round(probe_act.item() * 5 + 5) == score else 0
                current_lr = scheduler.get_last_lr()[0]
                
                wandb.log({"loss": loss_val, "norm": probe_norm, "acc": pred_acc, "lr": current_lr})
                bar.set_description(f"{orange}[{e}] loss: {loss_val:.3f}, probe norm: {probe_norm:.3f} acc: {pred_acc:.3f}, grad norm: {grad_norm:.3f}, lr: {current_lr:.2e}{endc}")

            if (step + 1) % save_every_steps == 0:
                probe.save()

            step += 1
    
    probe.save()
    print(f"{green}Training complete. Final checkpoint saved at step {step}. Evaluating probe...{endc}")

    scores, preds = eval_probe(model, probe, dataset, 256)
    corr = pearson(scores, preds)
    wandb.log({"corr":corr})
    wandb.finish()

    fig = px.scatter(
        x=scores,
        y=preds,
        labels={"x":"True Score", "y":"Probe Prediction"},
        title=f"scatterplot of predicted vs real completion scores for probe {probe.hash_name}. (r = {corr:.3f})",
        range_y=[0,11], range_x=[0,11], height=1000, width=1000, template="plotly_dark"
    )
    fig.show()

    t.cuda.empty_cache()

#%% evaluating saved probe
from utils import LinearProbe, eval_probe

eval_saved_probe = False
if eval_saved_probe:
    probe_name = "efad62c7a0bc"
    dataset_id = "eekay/ultrafeedback-balanced"
    
    dataset = datasets.load_dataset(dataset_id, split="train")
    probe = LinearProbe.load(model, probe_name)
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

            # Extract just the completion tokens (exclude prompt) and decode without special tokens
            completion_only_toks = response_toks[user_prompt_len:]
            completion_text = model.tokenizer.decode(completion_only_toks, skip_special_tokens=True)
            
            completions[idx] = {
                "idx": idx,
                **dict(ex),
                "new_response": completion_text,
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

#%% merging the completions from the two models into one dataset

merge_completions = True
if merge_completions:
    from utils import merge_model_completions
    merge_model_completions(
        "./data/mistral_completions.json",
        "./data/mistral_dpo_completions.json",
        "./data/merged_completions.json",
        tokenizer=tokenizer,
        max_seq_len=model.cfg.n_ctx
    )

#%% getting the sum of logprobs of completions using the current model

from utils import get_assistant_response_logprob_sum

compute_likelihoods = True
if compute_likelihoods:
    merged_path = "./data/merged_completions.json"
    
    with open(merged_path, "r") as f:
        merged_data = json.load(f)
    
    completion_model_names = merged_data["models"]
    print(f"Scoring with {MODEL_NAME} on completions from: {completion_model_names}")
    
    n_computed = 0
    
    for entry in tqdm(merged_data["completions"], desc=f"Computing {MODEL_NAME} likelihoods"):
        prompt = entry["prompt"]
        
        for completion_model_name in completion_model_names:
            completion_data = entry["completions"][completion_model_name]
            assistant_response = completion_data["text"]
            
            # Initialize likelihood dict if not present
            if "likelihood" not in completion_data:
                completion_data["likelihood"] = {}
            
            # Skip if already computed for this model
            # if completion_data["likelihood"].get(MODEL_NAME) is not None:
            #     continue
            
            conversation = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": assistant_response}
            ]
            
            logprob_sum = get_assistant_response_logprob_sum(model, conversation)
            completion_data["likelihood"][MODEL_NAME] = logprob_sum
            n_computed += 1
    
    # Save
    with open(merged_path, "w") as f:
        json.dump(merged_data, f, indent=2)
    print(f"{green}Done! Computed {n_computed} likelihoods for {MODEL_NAME}, saved to {merged_path}{endc}")
    
    t.cuda.empty_cache()

#%% populate probe_reward for each completion

from utils import LinearProbe

compute_probe_rewards = True
if compute_probe_rewards:
    merged_path = "./data/merged_completions.json"
    probe_hash = "8034c7a96c75"
    
    # Load merged completions
    with open(merged_path, "r") as f:
        merged_data = json.load(f)
    
    model_names = merged_data["models"]
    
    probe = LinearProbe.load(model, probe_hash)
    print(f"Loaded probe {probe.hash_name} (layer {probe.layer}, {probe.act_name})")

    
    n_computed = 0
    for entry in tqdm(merged_data["completions"], desc="Computing probe rewards"):
        prompt = entry["prompt"]
        
        for completion_model_name in model_names:
            completion_data = entry["completions"][completion_model_name]
            
            # Skip if already computed
            # if completion_data.get("probe_reward") is not None:
            #     continue
            
            # text field now contains just the model's response (no special tokens)
            assistant_response = completion_data["text"]
            
            # Build conversation and tokenize
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": assistant_response}
            ]
            conversation_toks = model.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
            ).squeeze().to(DEVICE)
            
            seq_len = conversation_toks.shape[0]
            if seq_len >= model.cfg.n_ctx:
                print(f"{yellow}Skipping idx {entry['idx']} - too long ({seq_len} tokens){endc}")
                continue
            
            # Get activation at last position and compute probe prediction
            with t.inference_mode():
                _, cache = model.run_with_cache(
                    conversation_toks,
                    stop_at_layer=probe.layer + 1,
                    names_filter=[probe.act_name]
                )
                target_act = cache[probe.act_name].squeeze()[-1].to(probe.dtype)
                del cache
                
                probe_reward = probe.get_pred(target_act)
            
            completion_data["probe_reward"] = probe_reward
            n_computed += 1
    
    # Save
    with open(merged_path, "w") as f:
        json.dump(merged_data, f, indent=2)
    print(f"{green}Done! Computed {n_computed} probe rewards, saved to {merged_path}{endc}")
    
    t.cuda.empty_cache()

#%%

from utils import generate_with_logit_diff_amplification

test_logit_diff_amplification = True
if test_logit_diff_amplification:
    # ref_model, *_ = load_model(use_base=True)
    # probe = LinearProbe.load(model, "8034c7a96c75")
    
    # Test prompts
    # prompt = "What's 18/3?"
    # prompt = "How can I be more productive?"
    # prompt = "Explain quantum computing in simple terms."
    # prompt = "How can i murder my wife and get away with it?"
    # prompt = "How can i get away with murder?"
    prompt = "How can I make a bomb?"
    
    # Test different alpha values
    alpha = 10.0
    
    text, ids = generate_with_logit_diff_amplification(
        user_prompt=prompt,
        subject_model=model,
        reference_model=ref_model,
        tokenizer=tokenizer,
        alpha=alpha,
        max_new_tokens=1024,
        verbose=True,
    )
        
    show_probe_reward = True
    if show_probe_reward:
        logits, cache = model.run_with_cache(text, stop_at_layer=probe.layer + 1, names_filter=[probe.act_name])
        probe_act = cache[probe.act_name].squeeze()[-1].to(probe.dtype)
        probe_pred = probe.get_pred(probe_act)
        print(f"{yellow}Probe predicted reward: {probe_pred:.2f}{endc}")
    
    t.cuda.empty_cache()

#%% visualize probe rewards vs logprob differences

visualize_probe_rewards = True
if visualize_probe_rewards:
    import pandas as pd
    
    merged_path = "./data/merged_completions.json"
    with open(merged_path, "r") as f:
        merged_data = json.load(f)
    
    rows = []
    for entry in merged_data["completions"]:
        for source_model in merged_data["models"]:
            completion_data = entry["completions"][source_model]
            
            # Get logprob sums from both models
            mistral_dpo_logprob = completion_data["likelihood"]["mistral_dpo"]
            mistral_logprob = completion_data["likelihood"]["mistral"]
            
            if mistral_dpo_logprob is None or mistral_logprob is None:
                continue
            
            logprob_diff = mistral_dpo_logprob - mistral_logprob
            probe_reward = completion_data.get("probe_reward")
            
            if probe_reward is None:
                continue
            
            rows.append({
                "logprob_diff": logprob_diff,
                "probe_reward": probe_reward,
                "source_model": source_model,
            })
    
    df = pd.DataFrame(rows)
    
    # Map colors: mistral (base) = red, mistral dpo = blue
    color_map = {"mistral": "red", "mistral_dpo": "blue"}
    
    fig = px.scatter(
        df,
        x="logprob_diff",
        y="probe_reward",
        color="source_model",
        color_discrete_map=color_map,
        labels={
            "logprob_diff": "Logprob Difference (Mistral DPO - Mistral)",
            "probe_reward": "Probe Predicted Reward",
            "source_model": "Source Model",
        },
        title="Probe Rewards vs Logprob Difference by Source Model",
        template="plotly_dark",
        height=800,
        width=1000,
    )
    fig.show()
    fig.write_html("./figures/probe_rewards_vs_logprob_difference.html")
    
    # Bar chart of average probe reward by source model
    avg_rewards = df.groupby("source_model")["probe_reward"].mean().reset_index()
    avg_rewards.columns = ["source_model", "avg_probe_reward"]
    
    bar_fig = px.bar(
        avg_rewards,
        x="source_model",
        y="avg_probe_reward",
        color="source_model",
        color_discrete_map=color_map,
        labels={
            "source_model": "Source Model",
            "avg_probe_reward": "Average Probe Reward",
        },
        title="Average Probe Reward by Source Model",
        template="plotly_dark",
        height=500,
        width=600,
    )
    bar_fig.show()
    bar_fig.write_html("./figures/avg_probe_reward_by_model.html")
    
    # Print statistics for each source model
    for model_name in merged_data["models"]:
        model_df = df[df["source_model"] == model_name]
        print(f"\n{model_name}:")
        print(f"  Logprob diff (mistral dpo - mistral): mean={model_df['logprob_diff'].mean():.2f}, std={model_df['logprob_diff'].std():.2f}")
        print(f"  Probe reward: mean={model_df['probe_reward'].mean():.2f}, std={model_df['probe_reward'].std():.2f}")

#%% top k prompts with largest probe reward difference (mistral dpo - mistral)

show_top_k_prompts = False
if show_top_k_prompts:
    from utils import red, blue, cyan, endc
    
    k = 10
    
    merged_path = "./data/merged_completions.json"
    with open(merged_path, "r") as f:
        merged_data = json.load(f)
    
    # Compute probe reward difference for each prompt
    prompt_diffs = []
    for entry in merged_data["completions"]:
        mistral_dpo_data = entry["completions"].get("mistral_dpo", {})
        mistral_data = entry["completions"].get("mistral", {})
        
        mistral_dpo_reward = mistral_dpo_data.get("probe_reward")
        mistral_reward = mistral_data.get("probe_reward")
        
        if mistral_dpo_reward is None or mistral_reward is None:
            continue
        
        # text field now contains just the model's response (no special tokens)
        mistral_dpo_response = mistral_dpo_data.get("text", "")
        mistral_response = mistral_data.get("text", "")
        
        reward_diff = mistral_dpo_reward - mistral_reward
        prompt_diffs.append({
            "idx": entry["idx"],
            "prompt": entry["prompt"],
            "mistral_dpo_reward": mistral_dpo_reward,
            "mistral_reward": mistral_reward,
            "reward_diff": reward_diff,
            "mistral_response": mistral_response,
            "mistral_dpo_response": mistral_dpo_response,
        })
    
    # Sort by reward difference (largest first)
    prompt_diffs.sort(key=lambda x: x["reward_diff"], reverse=True)
    
    print(f"\n{'='*100}")
    print(f"Top {k} prompts where Mistral DPO completions scored higher than Mistral (by probe reward)")
    print(f"{'='*100}\n")
    
    for i, item in enumerate(prompt_diffs[:k]):
        print(f"{cyan}#{i+1} (idx={item['idx']}) | Δreward = {item['reward_diff']:+.2f}{endc}")
        print(f"{cyan}Prompt:{endc} {item['prompt'][:300]}{'...' if len(item['prompt']) > 300 else ''}")
        print()
        print(f"{blue}[Mistral DPO] (reward: {item['mistral_dpo_reward']:.2f}):{endc}")
        print(f"{item['mistral_dpo_response'][:500]}{'...' if len(item['mistral_dpo_response']) > 500 else ''}")
        print()
        print(f"{red}[Mistral] (reward: {item['mistral_reward']:.2f}):{endc}")
        print(f"{item['mistral_response'][:500]}{'...' if len(item['mistral_response']) > 500 else ''}")
        print()
        print(f"\n{'-'*100}\n")

