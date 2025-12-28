import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import wandb

# ============================= Config ============================= #

DEVICE = "cuda"
DTYPE = t.bfloat16

# Model config
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.1"
USE_QLORA = True  # Use QLoRA for memory efficiency

# Training config
# OUTPUT_DIR = "./dpo_output"
# HUB_REPO_ID_MERGED = "eekay/mistral-7b-instruct-dpo"  # Hub repo for merged model
# HUB_REPO_ID_ADAPTER = "eekay/mistral-7b-instruct-dpo-adapter"  # Hub repo for LoRA adapter
OUTPUT_DIR = "./short_dpo_output"
HUB_REPO_ID_MERGED = "eekay/mistral-7b-instruct-short-dpo"  # Hub repo for merged model
HUB_REPO_ID_ADAPTER = "eekay/mistral-7b-instruct-short-dpo-adapter"  # Hub repo for LoRA adapter
ADAPTER_PATH = "./dpo_output/checkpoint-500"  # Path to existing adapter to resume from, or None to init new

LEARNING_RATE = 5e-6
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
NUM_TRAIN_EPOCHS = 1
MAX_LENGTH = 1024
MAX_PROMPT_LENGTH = 512
BETA = 0.05  # DPO beta parameter (controls deviation from reference model)
WARMUP_RATIO = 0.1
LOGGING_STEPS = 10
SAVE_STEPS = 100
EVAL_STEPS = 100

# ============================= Load Model ============================= #

def load_model_and_tokenizer(model_id: str = MODEL_ID, use_qlora: bool = USE_QLORA, adapter_path: str | None = ADAPTER_PATH):
    """Load the model and tokenizer for DPO training."""
    
    print(f"Loading model: {model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # DPO requires left padding
    
    if use_qlora:
        # QLoRA config for memory efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=DTYPE,
            bnb_4bit_use_double_quant=True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=DTYPE,
            # attn_implementation="flash_attention_2",
        )
        model = prepare_model_for_kbit_training(model)
        
        # Load existing adapter or create new LoRA config
        if adapter_path is not None:
            print(f"Loading existing adapter from: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
            peft_config = None  # Already applied, don't pass to trainer
        else:
            # LoRA config
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=DTYPE,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        peft_config = None
    
    return model, tokenizer, peft_config

def load_preference_dataset(dataset=None):
    dataset_id = "HuggingFaceH4/ultrafeedback_binarized" if dataset is None else dataset
    print(f"Loading dataset: '{dataset_id}'")
    
    dataset = load_dataset(dataset_id, split="train_prefs" if dataset is None else "train")
    
    # The dataset has 'chosen' and 'rejected' columns which contain message lists
    # DPO trainer expects: prompt, chosen, rejected columns
    
    def format_example(example):
        """Format the example for DPO training."""
        # The dataset structure has:
        # - 'chosen': list of messages (conversation with chosen response)
        # - 'rejected': list of messages (conversation with rejected response)
        # - 'prompt': the user prompt
        
        # Extract the prompt (should be the same in both)
        prompt = example["prompt"]
        
        # Extract chosen and rejected responses from the message lists
        # The messages are in format [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]
        chosen_messages = example["chosen"]
        rejected_messages = example["rejected"]
        
        # Get the assistant's response (last message in each conversation)
        chosen_response = chosen_messages[-1]["content"] if chosen_messages else ""
        rejected_response = rejected_messages[-1]["content"] if rejected_messages else ""
        
        return {
            "prompt": prompt,
            "chosen": chosen_response,
            "rejected": rejected_response,
        }
    
    formatted_dataset = dataset.map(
        format_example,
        remove_columns=dataset.column_names,
        desc="Formatting dataset for DPO",
    )
    
    # Filter out examples where chosen == rejected
    formatted_dataset = formatted_dataset.filter(
        lambda x: x["chosen"] != x["rejected"],
        desc="Filtering identical responses",
    )
    
    print(f"Dataset size: {len(formatted_dataset)}")
    print(f"Example: {formatted_dataset[0]}")
    
    return formatted_dataset

def train_dpo(
    model,
    tokenizer,
    dataset,
    peft_config=None,
    output_dir: str = OUTPUT_DIR,
    use_wandb: bool = True,
    hub_repo_merged: str = HUB_REPO_ID_MERGED,
    hub_repo_adapter: str = HUB_REPO_ID_ADAPTER,
):
    """Run DPO training."""
    
    # Initialize wandb
    if use_wandb:
        wandb.init(
            project="mistral-dpo",
            name="mistral-7b-instruct-dpo",
            config={
                "model_id": MODEL_ID,
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                "num_epochs": NUM_TRAIN_EPOCHS,
                "max_length": MAX_LENGTH,
                "beta": BETA,
            }
        )
    
    # DPO training config
    training_args = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        beta=BETA,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_strategy="no",  # Set to "steps" if you have eval dataset
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        lr_scheduler_type=None,
        report_to="wandb" if use_wandb else "none",
        remove_unused_columns=False,
        loss_type="sigmoid",  # Standard DPO loss
    )
    
    # Create DPO trainer
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    
    # Test push before training (adapter only - merged requires destructive merge_and_unload)
    if hub_repo_adapter:
        print("Testing adapter push before training...")
        trainer.model.push_to_hub(hub_repo_adapter, commit_message="Pre-training test (adapter)")
        tokenizer.push_to_hub(hub_repo_adapter, commit_message="Add tokenizer")
        print(f"Adapter test push complete: https://huggingface.co/{hub_repo_adapter}")
    
    # Train
    print("Starting DPO training...")
    trainer.train()
    
    # Save final model
    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Push to Hub
    if hub_repo_adapter:
        print(f"Pushing adapter to Hub: {hub_repo_adapter}")
        trainer.model.push_to_hub(hub_repo_adapter, commit_message="DPO training complete (adapter)")
        tokenizer.push_to_hub(hub_repo_adapter, commit_message="Add tokenizer")
        print(f"Adapter pushed to https://huggingface.co/{hub_repo_adapter}")
    
    if hub_repo_merged:
        print(f"Merging adapter into base model...")
        merged_model = trainer.model.merge_and_unload()
        
        print(f"Pushing merged model to Hub: {hub_repo_merged}")
        merged_model.push_to_hub(hub_repo_merged, commit_message="DPO training complete (merged)")
        tokenizer.push_to_hub(hub_repo_merged, commit_message="Add tokenizer")
        print(f"Merged model pushed to https://huggingface.co/{hub_repo_merged}")
    
    if use_wandb:
        wandb.finish()
    
    return trainer

if __name__ == "__main__":
    # Load model and tokenizer
    model, tokenizer, peft_config = load_model_and_tokenizer()
    
    # Load dataset
    # dataset = load_preference_dataset()
    dataset = load_preference_dataset(dataset="eekay/ultrafeedback-binarized-short-pref")
    
    # Train
    trainer = train_dpo(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        peft_config=peft_config,
        use_wandb=True,
    )
    
    print("DPO training complete!")

def merge_adapter_locally(
    base_model_id: str = MODEL_ID,
    adapter_id: str = HUB_REPO_ID_ADAPTER,
    output_dir: str = "./merged_model",
    push_to_hub: bool = False,
    hub_repo_id: str = HUB_REPO_ID_MERGED,
):
    """Load the base model and adapter, merge them, and save locally."""
    
    print(f"Loading base model: {base_model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=DTYPE,
        device_map="auto",
    )
    
    print(f"Loading adapter: {adapter_id}")
    model_with_adapter = PeftModel.from_pretrained(base_model, adapter_id)
    
    print("Merging adapter into base model...")
    merged_model = model_with_adapter.merge_and_unload()
    
    print(f"Saving merged model to {output_dir}")
    merged_model.save_pretrained(output_dir)
    
    # Also save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.save_pretrained(output_dir)
    
    if push_to_hub:
        print(f"Pushing merged model to Hub: {hub_repo_id}")
        merged_model.push_to_hub(hub_repo_id, commit_message="Merged model")
        tokenizer.push_to_hub(hub_repo_id, commit_message="Add tokenizer")
        print(f"Pushed to https://huggingface.co/{hub_repo_id}")
    
    print("Merge complete!")
    return merged_model, tokenizer

# Run merge
# merged_model, tokenizer = merge_adapter_locally()
