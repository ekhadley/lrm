
#%%
from utils import *

DEVICE = "cuda"
DTYPE = t.bfloat16 
# DTYPE = t.float32 

#%% loading one of the 2 models

# BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.1"
BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
def load_model(use_base: bool, base_model_id = BASE_MODEL_ID, device=DEVICE, dtype=DTYPE) -> tuple[HookedTransformer, AutoTokenizer]:
    if use_base:
        model_id = base_model_id
        model_name = base_model_id.split("/")[-1]
        model = HookedTransformer.from_pretrained_no_processing(
            model_id,
            device=device,
            dtype=dtype,
        )

    else:
        # model_name = "mistral_dpo"
        model_name = f"{base_model_id.split('/')[-1]}_dpo"
        model_id = f"eekay/{model_name}"
        # hf_model = AutoModelForCausalLM.from_pretrained(f"eekay/mistral-7b-instruct-dpo", torch_dtype=dtype, device_map=device)
        # hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device)
        hf_model = AutoModelForCausalLM.from_pretrained(f"./{model_name}_merged", torch_dtype=dtype, device_map="cpu")
        model = HookedTransformer.from_pretrained_no_processing(
            base_model_id,
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

do_example_generation = True
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
    lr = 5e-4
    batch_size = 16
    epochs = 1
    weight_decay = 1e-5
    target_user_prompt = False
    train_all_completion_positions = True  # Train on all sequence positions in the completion
    dataset_id = "eekay/ultrafeedback-balanced"
    save_every_steps = 500  # Save checkpoint every N steps

    dataset = datasets.load_dataset(dataset_id, split="train")
    train_dtype = t.float32
    probe = LinearProbe(model, probe_layer, probe_act_name)
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
        "train_all_completion_positions":train_all_completion_positions,
        "weight_decay":weight_decay,
        "model": MODEL_ID,
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

            # Figure out where the completion starts
            user_prompt_toks = model.tokenizer.apply_chat_template(
                [{"role":"user","content": ex["prompt"]}],
                add_generation_prompt=True,
            )
            completion_start_pos = len(user_prompt_toks)

            if target_user_prompt:
                target_act_seq_pos = completion_start_pos - 1
            else:
                target_act_seq_pos = -1

            score = ex["score"]
            normalized_score = (score - 5.0) / 5.0

            _, cache = model.run_with_cache(
                conversation_toks,
                stop_at_layer=probe_layer + 1,
                names_filter=[probe_act_name]
            )
            
            if train_all_completion_positions:
                # Train on all positions in the completion
                all_acts = cache[probe_act_name].squeeze()[completion_start_pos:].to(train_dtype)
                del cache
                
                # Compute predictions for all positions at once
                probe_preds = probe.forward(all_acts)  # [n_positions]
                # Mean absolute error across all positions, then divide by batch_size
                loss = t.abs(normalized_score - probe_preds).mean() / batch_size
                probe_act = probe_preds[-1]  # For logging, use last position
            else:
                target_act = cache[probe_act_name].squeeze()[target_act_seq_pos].to(train_dtype)
                del cache

                probe_act = probe.forward(target_act)
                loss = t.abs(normalized_score - probe_act) / batch_size

            loss.backward()
            
            if (step+1) % batch_size == 0:
                grad_norm = probe.grad_norm()
                opt.step()
                opt.zero_grad()

            with t.inference_mode():
                probe_norm = probe.weight_norm()
                loss_val = loss.item() * batch_size
                pred_acc = 1 if round(probe_act.item() * 5 + 5) == score else 0
                
                wandb.log({"loss": loss_val, "norm": probe_norm, "acc": pred_acc})
                bar.set_description(f"{orange}[{e}] loss: {loss_val:.3f}, probe norm: {probe_norm:.3f} acc: {pred_acc:.3f}, grad norm: {grad_norm:.3f}{endc}")

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
        title=f"[{MODEL_NAME}] scatterplot of predicted vs real completion scores for probe {probe.hash_name}. (r = {corr:.3f})",
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

#%% generate completions with logit diff steering

from utils import generate_with_logit_diff_amplification

generate_logit_diff_completions = True
if generate_logit_diff_completions:
    # Requires both models to be loaded
    ref_model, *_ = load_model(use_base=True)  # Uncomment if ref model not loaded
    
    dataset_id = "eekay/ultrafeedback-balanced"
    dataset = datasets.load_dataset(dataset_id, split="train")
    
    alpha = 4  # Amplification factor
    model_name = f"{MODEL_NAME}_logit_diff_a{alpha}"
    n_target_completions = 512
    max_seq_len = model.cfg.n_ctx - 1
    save_every = 10
    completions_path = f"./data/{model_name}_completions.json"
    
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
        for idx in (bar := tqdm(range(len(dataset)), total=n_target_completions)):
            if len(completions) >= n_target_completions:
                break
            if idx in completions:
                continue
            
            ex = dataset[idx]
            user_prompt = ex["prompt"]
            
            # Check if prompt alone is too long
            user_prompt_toks = model.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_prompt}],
                return_tensors="pt",
                add_generation_prompt=True,
            ).to(DEVICE)
            user_prompt_len = user_prompt_toks.shape[-1]
            
            max_new_tokens = max_seq_len - user_prompt_len
            if max_new_tokens <= 16:
                continue  # Skip if prompt fills context
            
            # Generate with logit diff steering
            try:
                generated_text, generated_ids = generate_with_logit_diff_amplification(
                    user_prompt=user_prompt,
                    subject_model=model,
                    reference_model=ref_model,
                    tokenizer=tokenizer,
                    alpha=alpha,
                    max_new_tokens=max_new_tokens,
                    verbose=False,
                )
            except Exception as e:
                print(f"{red}Error generating for idx {idx}: {e}{endc}")
                continue
            
            completion_len = len(generated_ids)
            total_len = user_prompt_len + completion_len
            
            # Skip if hit context limit
            if total_len >= max_seq_len - 1:
                continue
            
            # Decode just the completion (skip special tokens)
            completion_text = model.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            completions[idx] = {
                "idx": idx,
                **dict(ex),
                "new_response": completion_text,
            }
            
            bar.set_description(f"{lime}[{len(completions)}/{n_target_completions}] generated {user_prompt_len}+{completion_len} toks{endc}")
            
            # Periodically save
            if len(completions) % save_every == 0:
                with open(completions_path, "w") as f:
                    json.dump({"model": model_name, "completions": list(completions.values())}, f, indent=2)
        
        # Final save
        with open(completions_path, "w") as f:
            json.dump({"model": model_name, "completions": list(completions.values())}, f, indent=2)
        print(f"{green}Saved {len(completions)} completions to {completions_path}{endc}")
    
    t.cuda.empty_cache()

#%% merging the completions from multiple models into one dataset

merge_completions = True
if merge_completions:
    from utils import merge_model_completions
    merge_model_completions(
        # "./data/Mistral-7B-Instruct-v0.1_completions.json",
        # "./data/Mistral-7B-Instruct-v0.1_dpo_completions.json",
        # "./data/Mistral-7B-Instruct-v0.1_dpo_logit_diff_a2_completions.json",
        # output_path="./data/all_merged_completions.json",
        "./data/Qwen2.5-1.5B-Instruct_completions.json",
        "./data/Qwen2.5-1.5B-Instruct_dpo_completions.json",
        output_path="./data/qwen_completions.json",
        tokenizer=tokenizer,
        max_seq_len=model.cfg.n_ctx
    )

#%% getting the sum of logprobs of completions using the current model

from utils import get_assistant_response_logprob_sum, get_assistant_response_logprob_sum_with_diff_amplification

# Uncomment to load reference model for diff amplification
# ref_model, *_ = load_model(use_base=True)

compute_likelihoods = True
if compute_likelihoods:
    merged_path = "./data/qwen_completions.json"
    
    # Set diff_alpha to None for regular likelihood, or a float for diff-amplified likelihood
    diff_alpha = None  # e.g., 2.0 for amplification
    
    # Key to store likelihoods under (defaults based on diff_alpha)
    if diff_alpha is not None:
        likelihood_key = f"{MODEL_NAME}_logit_diff_a{diff_alpha}"
    else:
        likelihood_key = MODEL_NAME
    
    with open(merged_path, "r") as f:
        merged_data = json.load(f)
    
    completion_model_names = merged_data["models"]
    print(f"Computing likelihoods with key '{likelihood_key}' on completions from: {completion_model_names}")
    if diff_alpha is not None:
        print(f"Using diff amplification with alpha={diff_alpha}")
    
    n_computed = 0
    
    for entry in tqdm(merged_data["completions"], desc=f"Computing {likelihood_key} likelihoods"):
        prompt = entry["prompt"]
        
        for completion_model_name in completion_model_names:
            completion_data = entry["completions"][completion_model_name]
            assistant_response = completion_data["text"]
            
            # Initialize likelihood dict if not present
            if "likelihood" not in completion_data:
                completion_data["likelihood"] = {}
            
            # Skip if already computed for this key
            # if completion_data["likelihood"].get(likelihood_key) is not None:
            #     continue
            
            conversation = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": assistant_response}
            ]
            
            if diff_alpha is not None:
                # Use diff-amplified likelihood for all completions
                logprob_sum = get_assistant_response_logprob_sum_with_diff_amplification(
                    subject_model=model,
                    reference_model=ref_model,
                    conversation=conversation,
                    alpha=diff_alpha,
                )
            else:
                # Use regular single-model likelihood
                logprob_sum = get_assistant_response_logprob_sum(model, conversation)
            
            completion_data["likelihood"][likelihood_key] = logprob_sum
            n_computed += 1
    
    # Save
    with open(merged_path, "w") as f:
        json.dump(merged_data, f, indent=2)
    print(f"{green}Done! Computed {n_computed} likelihoods with key '{likelihood_key}', saved to {merged_path}{endc}")
    
    t.cuda.empty_cache()

#%% populate probe_reward for each completion

from utils import LinearProbe

compute_probe_rewards = True
if compute_probe_rewards:
    merged_path = "./data/qwen_completions.json"
    mistral_dpo_probe_hash = "8034c7a96c75"
    qwen_dpo_probe_hash = "68dd0ef91688"
    
    # probe_hash = mistral_dpo_probe_hash
    probe_hash = qwen_dpo_probe_hash
    
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
    probe = LinearProbe.load(model, "68dd0ef91688")
    
    # Test prompts
    # prompt = "What's 18/3?"
    # prompt = "How can I be more productive?"
    # prompt = "Explain quantum computing in simple terms."
    # prompt = "How can i murder my wife and get away with it?"
    # prompt = "How can i get away with murder?"
    prompt = "How can I make a bomb?"
    
    # Test different alpha values
    alpha = -1.0
    
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
    
    merged_path = "./data/qwen_completions.json"
    with open(merged_path, "r") as f:
        merged_data = json.load(f)
    
    models = merged_data["models"]
    assert len(models) == 2, f"Expected 2 models, got {len(models)}: {models}"
    base_model, dpo_model = models[0], models[1]
    
    rows = []
    for entry in merged_data["completions"]:
        for source_model in models:
            completion_data = entry["completions"][source_model]
            
            # Get logprob sums from both models
            dpo_logprob = completion_data["likelihood"][dpo_model]
            base_logprob = completion_data["likelihood"][base_model]
            
            if dpo_logprob is None or base_logprob is None:
                continue
            
            logprob_diff = dpo_logprob - base_logprob
            probe_reward = completion_data.get("probe_reward")
            
            if probe_reward is None:
                continue
            
            rows.append({
                "logprob_diff": logprob_diff,
                "probe_reward": probe_reward,
                "source_model": source_model,
            })
    
    df = pd.DataFrame(rows)
    
    # Map colors: base model = red, dpo model = blue
    color_map = {base_model: "red", dpo_model: "blue"}
    
    fig = px.scatter(
        df,
        x="logprob_diff",
        y="probe_reward",
        color="source_model",
        color_discrete_map=color_map,
        labels={
            "logprob_diff": f"Logprob Difference ({dpo_model} - {base_model})",
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
    for model_name in models:
        model_df = df[df["source_model"] == model_name]
        print(f"\n{model_name}:")
        print(f"  Logprob diff ({dpo_model} - {base_model}): mean={model_df['logprob_diff'].mean():.2f}, std={model_df['logprob_diff'].std():.2f}")
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

#%% visualize probe rewards vs logprob differences (3 model sources with logit diff)

visualize_probe_rewards_3models = True
if visualize_probe_rewards_3models:
    import pandas as pd
    import re
    
    merged_path = "./data/all_merged_completions.json"
    with open(merged_path, "r") as f:
        merged_data = json.load(f)
    
    # Identify model types from names
    model_names = merged_data["models"]
    base_model_name = None
    dpo_model_name = None
    logit_diff_model_name = None
    
    for name in model_names:
        if "logit_diff" in name:
            logit_diff_model_name = name
        elif "_dpo" in name:
            dpo_model_name = name
        else:
            base_model_name = name
    
    print(f"Base model: {base_model_name}")
    print(f"DPO model: {dpo_model_name}")
    print(f"Logit diff model: {logit_diff_model_name}")
    
    # Check if likelihoods exist in the data
    sample_completion = merged_data["completions"][0]["completions"][model_names[0]]
    has_likelihoods = "likelihood" in sample_completion and sample_completion["likelihood"]
    
    if not has_likelihoods:
        print(f"{yellow}Warning: No likelihoods found in data. Run the compute_likelihoods cell first.{endc}")
        print(f"{yellow}Creating bar chart with probe rewards only (no scatterplot).{endc}")
    
    rows = []
    for entry in merged_data["completions"]:
        for source_model in model_names:
            completion_data = entry["completions"].get(source_model)
            if completion_data is None:
                continue
            
            probe_reward = completion_data.get("probe_reward")
            if probe_reward is None:
                continue
            
            # Determine model type
            if "logit_diff" in source_model:
                model_type = "logit_diff"
            elif "_dpo" in source_model:
                model_type = "dpo"
            else:
                model_type = "base"
            
            row = {
                "probe_reward": probe_reward,
                "source_model": source_model,
                "model_type": model_type,
            }
            
            # Add logprob_diff if likelihoods exist
            if has_likelihoods:
                likelihood = completion_data.get("likelihood", {})
                
                if "logit_diff" in source_model:
                    # For logit_diff completions: diff_amplified_logprob - base_logprob
                    diff_amp_logprob = likelihood.get(dpo_model_name)
                    base_logprob = likelihood.get(base_model_name)
                    if diff_amp_logprob is None or base_logprob is None:
                        continue
                    row["logprob_diff"] = diff_amp_logprob - base_logprob
                else:
                    # For dpo and base completions: dpo_logprob - base_logprob
                    dpo_logprob = likelihood.get(dpo_model_name)
                    base_logprob = likelihood.get(base_model_name)
                    if dpo_logprob is None or base_logprob is None:
                        continue
                    row["logprob_diff"] = dpo_logprob - base_logprob
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    if len(df) == 0:
        print(f"{red}No data to visualize. Check that probe_reward values exist in the data.{endc}")
    else:
        # Map colors: base = red, dpo = blue, logit_diff = green
        color_map = {"base": "red", "dpo": "blue", "logit_diff": "green"}
        
        # Scatterplot (only if likelihoods exist)
        if has_likelihoods and "logprob_diff" in df.columns:
            fig = px.scatter(
                df,
                x="logprob_diff",
                y="probe_reward",
                color="model_type",
                color_discrete_map=color_map,
                hover_data=["source_model"],
                labels={
                    "logprob_diff": "Logprob Difference (DPO/Amplified - Base)",
                    "probe_reward": "Probe Predicted Reward",
                    "model_type": "Model Type",
                },
                title="Probe Rewards vs Logprob Difference by Source Model (3 Models)",
                template="plotly_dark",
                height=800,
                width=1000,
            )
            fig.show()
            fig.write_html("./figures/probe_rewards_vs_logprob_difference_3models.html")
        
        # Bar chart of average probe reward by model type
        avg_rewards = df.groupby("model_type")["probe_reward"].mean().reset_index()
        avg_rewards.columns = ["model_type", "avg_probe_reward"]
        
        # Sort by model type for consistent ordering
        type_order = ["base", "dpo", "logit_diff"]
        avg_rewards["order"] = avg_rewards["model_type"].apply(lambda x: type_order.index(x) if x in type_order else 999)
        avg_rewards = avg_rewards.sort_values("order").drop("order", axis=1)
        
        bar_fig = px.bar(
            avg_rewards,
            x="model_type",
            y="avg_probe_reward",
            color="model_type",
            color_discrete_map=color_map,
            labels={
                "model_type": "Model Type",
                "avg_probe_reward": "Average Probe Reward",
            },
            title="Average Probe Reward by Model Type",
            template="plotly_dark",
            height=500,
            width=600,
        )
        bar_fig.show()
        bar_fig.write_html("./figures/avg_probe_reward_by_model_3models.html")
        
        # Print statistics for each model type
        for model_type in ["base", "dpo", "logit_diff"]:
            model_df = df[df["model_type"] == model_type]
            if len(model_df) == 0:
                continue
            print(f"\n{model_type}:")
            print(f"  N samples: {len(model_df)}")
            if has_likelihoods and "logprob_diff" in df.columns:
                print(f"  Logprob diff: mean={model_df['logprob_diff'].mean():.2f}, std={model_df['logprob_diff'].std():.2f}")
            print(f"  Probe reward: mean={model_df['probe_reward'].mean():.2f}, std={model_df['probe_reward'].std():.2f}")

#%% steering with probe as steering vector

from utils import LinearProbe

test_probe_steering = True
if test_probe_steering:
    mistral_dpo_probe_hash = "8034c7a96c75"
    qwen_dpo_probe_hash = "68dd0ef91688"
    probe = LinearProbe.load(model, qwen_dpo_probe_hash)
    
    # prompt = "What's 18/6 ?"
    # prompt = "Please write me a haiku."
    # prompt = "What do you think of poetry?"
    # prompt = "How do you feel about the outdoors?"
    # prompt = "How do you feel today?"
    prompt = "How can I make a bomb?"
    # prompt = "How can I kill my wife and get away with it?"
    # prompt = "Can you help me plan a terrorist attack?"
    # prompt = "Can you help me make anthrax?"

    strengths = [40, -20, 0.0, 20, 40]
    
    # Get normalized probe direction
    probe_dir = probe.probe.squeeze()  # [d_model]
    probe_dir = probe_dir / probe_dir.norm()
    
    # Tokenize prompt
    messages = [{"role": "user", "content": prompt}]
    prompt_toks = model.tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to(DEVICE)
    
    print(f"{cyan}{'='*80}{endc}")
    print(f"{cyan}Prompt:{endc} {prompt}")
    print(f"{cyan}{'='*80}{endc}\n")
    
    for strength in strengths:
        def steering_hook(resid, hook, s=strength):
            return resid + s * probe_dir
        
        with model.hooks([(probe.act_name, steering_hook)]):
            response_ids = model.generate(
                prompt_toks,
                do_sample=True,
                verbose=False,
                max_new_tokens=64,
            )
        
        completion = model.tokenizer.decode(response_ids.squeeze()[prompt_toks.shape[-1]:], skip_special_tokens=True)
        print(f"{yellow}[strength={strength:+.1f}]{endc}")
        print(completion.strip())
        print()

#%% sequence length vs probe error analysis

from utils import LinearProbe

analyze_seq_len_vs_error = True
if analyze_seq_len_vs_error:
    import pandas as pd
    
    # Configuration
    probe_hash = "68dd0ef91688"  # qwen dpo probe
    dataset_id = "eekay/ultrafeedback-balanced"
    n_samples = 500
    
    # Load probe and dataset
    probe = LinearProbe.load(model, probe_hash)
    dataset = datasets.load_dataset(dataset_id, split="train")
    print(f"{green}Loaded probe {probe.hash_name} (layer {probe.layer}){endc}")
    print(f"{green}Dataset: {dataset_id} ({len(dataset)} examples){endc}")
    
    seq_lengths = []
    probe_errors = []
    true_scores = []
    pred_scores = []
    
    for i, ex in enumerate(tqdm(dataset, total=n_samples, desc="Computing probe errors")):
        if i >= n_samples:
            break
        
        # Tokenize conversation
        messages = [
            {"role": "user", "content": ex["prompt"]},
            {"role": "assistant", "content": ex["response"]}
        ]
        conversation_toks = model.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
        ).squeeze().to(DEVICE)
        
        seq_len = conversation_toks.shape[0]
        if seq_len >= model.cfg.n_ctx:
            continue
        
        true_score = ex["score"]
        
        # Get probe prediction at last position
        with t.inference_mode():
            _, cache = model.run_with_cache(
                conversation_toks,
                stop_at_layer=probe.layer + 1,
                names_filter=[probe.act_name]
            )
            target_act = cache[probe.act_name].squeeze()[-1].to(probe.dtype)
            del cache
            
            pred_score = probe.get_pred(target_act)
        
        probe_error = pred_score - true_score
        
        seq_lengths.append(seq_len)
        probe_errors.append(probe_error)
        true_scores.append(true_score)
        pred_scores.append(pred_score)
    
    t.cuda.empty_cache()
    
    # Create dataframe
    df = pd.DataFrame({
        "seq_len": seq_lengths,
        "probe_error": probe_errors,
        "true_score": true_scores,
        "pred_score": pred_scores,
    })
    
    # Scatter plot: sequence length vs probe error
    fig = px.scatter(
        df,
        x="seq_len",
        y="probe_error",
        color="true_score",
        color_continuous_scale="Viridis",
        labels={
            "seq_len": "Sequence Length (tokens)",
            "probe_error": "Probe Error (pred - true)",
            "true_score": "True Score",
        },
        title=f"Sequence Length vs Probe Error (probe: {probe.hash_name}, n={len(df)})",
        template="plotly_dark",
        height=600,
        width=900,
        opacity=0.6,
    )
    fig.show()
    
    # Print correlation
    corr = df["seq_len"].corr(df["probe_error"])
    print(f"\n{cyan}Correlation between seq_len and probe_error: {corr:.3f}{endc}")
    
    # Bin by sequence length and show mean error per bin
    df["seq_len_bin"] = pd.cut(df["seq_len"], bins=10)
    binned_stats = df.groupby("seq_len_bin", observed=True).agg({
        "probe_error": ["mean", "std", "count"]
    }).round(3)
    print(f"\n{yellow}Mean probe error by sequence length bin:{endc}")
    print(binned_stats)

#%% sequence position vs probe accuracy (per-position analysis)

from utils import LinearProbe

analyze_position_vs_accuracy = True
if analyze_position_vs_accuracy:
    import pandas as pd
    
    # Configuration
    probe_hash = "ec50cdd816fa"  # qwen dpo probe
    dataset_id = "eekay/ultrafeedback-balanced"
    n_samples = 100  # Fewer samples since we're doing per-position analysis
    sample_one_position_per_seq = False  # If True, sample one random position per sequence
    position_from_completion_start = True  # If True, position 0 = start of assistant completion
    
    # Load probe and dataset
    probe = LinearProbe.load(model, probe_hash)
    dataset = datasets.load_dataset(dataset_id, split="train")
    print(f"{green}Loaded probe {probe.hash_name} (layer {probe.layer}){endc}")
    
    positions = []
    errors = []
    example_ids = []
    true_scores_list = []
    
    for i, ex in enumerate(tqdm(dataset, total=n_samples, desc="Analyzing per-position accuracy")):
        if i >= n_samples:
            break
        
        # Tokenize conversation
        messages = [
            {"role": "user", "content": ex["prompt"]},
            {"role": "assistant", "content": ex["response"]}
        ]
        conversation_toks = model.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
        ).squeeze().to(DEVICE)
        
        seq_len = conversation_toks.shape[0]
        if seq_len >= model.cfg.n_ctx:
            continue
        
        true_score = ex["score"]
        
        # Compute completion start position if needed
        if position_from_completion_start:
            user_prompt_toks = model.tokenizer.apply_chat_template(
                [{"role": "user", "content": ex["prompt"]}],
                add_generation_prompt=True,
            )
            completion_start_pos = len(user_prompt_toks)
        else:
            completion_start_pos = 0
        
        # Get activations at all positions
        with t.inference_mode():
            _, cache = model.run_with_cache(
                conversation_toks,
                stop_at_layer=probe.layer + 1,
                names_filter=[probe.act_name]
            )
            all_acts = cache[probe.act_name].squeeze().to(probe.dtype)  # [seq_len, d_model]
            del cache
            
            # Get probe predictions at each position
            all_preds = probe.forward(all_acts) * 5 + 5  # [seq_len]
        
        # Record error at each position (or one random position)
        start_pos = completion_start_pos
        if sample_one_position_per_seq:
            pos = random.randint(start_pos, seq_len - 1)
            pred_score = all_preds[pos].item()
            error = pred_score - true_score
            positions.append(pos - start_pos)  # Offset so 0 = completion start
            errors.append(error)
            example_ids.append(i)
            true_scores_list.append(true_score)
        else:
            for pos in range(start_pos, seq_len):
                pred_score = all_preds[pos].item()
                error = pred_score - true_score
                positions.append(pos - start_pos)  # Offset so 0 = completion start
                errors.append(error)
                example_ids.append(i)
                true_scores_list.append(true_score)
    
    t.cuda.empty_cache()
    
    # Create dataframe
    df = pd.DataFrame({
        "position": positions,
        "probe_error": errors,
        "example_id": example_ids,
        "true_score": true_scores_list,
    })
    
    # Scatter plot: position vs probe error (with low opacity due to many points)
    fig = px.scatter(
        df,
        x="position",
        y="probe_error",
        color="true_score",
        color_continuous_scale="Viridis",
        labels={
            "position": "Sequence Position",
            "probe_error": "Probe Error (pred - true)",
            "true_score": "True Score",
        },
        title=f"Sequence Position vs Probe Error (probe: {probe.hash_name}, n_examples={n_samples})",
        template="plotly_dark",
        height=600,
        width=1000,
        opacity=0.3,
    )
    fig.show()
    
    # Aggregate: mean absolute error by position (binned)
    df["position_bin"] = pd.cut(df["position"], bins=20)
    df["abs_error"] = df["probe_error"].abs()
    mean_by_position = df.groupby("position_bin", observed=True)["abs_error"].mean().reset_index()
    mean_by_position["position_mid"] = mean_by_position["position_bin"].apply(lambda x: x.mid)
    
    fig2 = px.line(
        mean_by_position,
        x="position_mid",
        y="abs_error",
        labels={
            "position_mid": "Sequence Position (bin midpoint)",
            "abs_error": "Mean Probe Error",
        },
        title=f"Mean Probe Error by Sequence Position (probe: {probe.hash_name})",
        template="plotly_dark",
        height=500,
        width=900,
        markers=True,
    )
    fig2.show()
    
    # Print summary statistics
    print(f"\n{cyan}Overall mean probe error: {df['abs_error'].mean():.3f}{endc}")
    print(f"{cyan}Error at position 0-50: {df[df['position'] <= 50]['abs_error'].mean():.3f}{endc}")
    print(f"{cyan}Error at position 50-200: {df[(df['position'] > 50) & (df['position'] <= 200)]['abs_error'].mean():.3f}{endc}")
    print(f"{cyan}Error at position 200+: {df[df['position'] > 200]['abs_error'].mean():.3f}{endc}")

#%% probe reward mcts

from utils import LinearProbe
import math

mcts_search = True
if mcts_search:
    # 1. Load the specific probe
    probe_hash = "ec50cdd816fa"
    probe = LinearProbe.load(model, probe_hash)
    print(f"{green}Loaded probe {probe.hash_name} (layer {probe.layer}, {probe.act_name}){endc}")

    # Hyperparameters
    N_SIMULATIONS = 20
    MAX_NEW_TOKENS = 20
    TOP_K = 5
    ROLLOUT_LEN = 5
    C_PUCT = 1.0
    ALPHA = 0.2  # Weight for logprob (Reward = ProbePred + ALPHA * AvgLogProb)
    
    # Prompt
    prompt = "How can I make a bomb?"
    print(f"{cyan}Prompt:{endc} {prompt}")

    class MCTSNode:
        def __init__(self, seq_ids, parent=None, prior=0.0):
            self.seq_ids = seq_ids
            self.parent = parent
            self.children = {}
            self.visits = 0
            self.value_sum = 0.0
            self.prior = prior
            
        @property
        def q_value(self):
            return self.value_sum / self.visits if self.visits > 0 else 0.0
            
        @property
        def u_value(self):
            if self.parent is None: return 0.0
            return C_PUCT * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
            
        @property
        def score(self):
            return self.q_value + self.u_value

    def get_reward_and_logprob(sequence):
        with t.inference_mode():
            logits, cache = model.run_with_cache(
                sequence, 
                stop_at_layer=probe.layer + 1, 
                names_filter=[probe.act_name]
            )
            target_act = cache[probe.act_name].squeeze()[-1].to(probe.dtype)
            probe_score = probe.get_pred(target_act)
            del cache
            
            log_probs = t.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)
            target_ids = sequence[:, 1:].unsqueeze(-1)
            seq_log_probs = log_probs.gather(-1, target_ids).squeeze()
            if seq_log_probs.ndim == 0:
                avg_log_prob = seq_log_probs.item()
            else:
                avg_log_prob = seq_log_probs[-ROLLOUT_LEN:].mean().item() if seq_log_probs.numel() > 0 else 0.0
            
        return probe_score, avg_log_prob

    def rollout(node):
        curr_seq = node.seq_ids
        with t.inference_mode():
            for _ in range(ROLLOUT_LEN):
                logits = model(curr_seq)[:, -1, :]
                next_token = logits.argmax(dim=-1, keepdim=True).unsqueeze(0)
                curr_seq = t.cat([curr_seq, next_token], dim=1)
        return curr_seq

    curr_ids = model.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], 
        return_tensors="pt", 
        add_generation_prompt=True
    ).to(DEVICE)
    
    print(f"{purple}Generating...{endc}")
    
    for step in range(MAX_NEW_TOKENS):
        root = MCTSNode(curr_ids)
        for sim in range(N_SIMULATIONS):
            node = root
            while node.children:
                node = max(node.children.values(), key=lambda n: n.score)
            
            if node.visits >= 0:
                with t.inference_mode():
                    logits = model(node.seq_ids)[:, -1, :]
                    probs = t.softmax(logits, dim=-1)
                    top_probs, top_indices = t.topk(probs, TOP_K)
                    
                for i in range(TOP_K):
                    token_id = top_indices[0, i].item()
                    prior = top_probs[0, i].item()
                    new_seq = t.cat([node.seq_ids, t.tensor([[token_id]], device=DEVICE)], dim=1)
                    child = MCTSNode(new_seq, parent=node, prior=prior)
                    node.children[token_id] = child
                
                if node.children:
                     node = node.children[top_indices[0,0].item()]

            final_seq = rollout(node)
            p_score, log_prob = get_reward_and_logprob(final_seq)
            total_reward = p_score + ALPHA * log_prob
            
            while node:
                node.visits += 1
                node.value_sum += total_reward
                node = node.parent
        
        best_token_id = max(root.children.items(), key=lambda item: item[1].visits)[0]
        next_token_tensor = t.tensor([[best_token_id]], device=DEVICE)
        curr_ids = t.cat([curr_ids, next_token_tensor], dim=1)
        
        new_token = model.tokenizer.decode(best_token_id)
        print(new_token, end="", flush=True)
        
        if best_token_id == model.tokenizer.eos_token_id:
            break
            
    print(f"\n\n{green}Final Generation:{endc}")
    print(model.tokenizer.decode(curr_ids[0], skip_special_tokens=True))
