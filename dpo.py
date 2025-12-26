#%%

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import wandb

# ============================= Config ============================= #

#%%

DEVICE = "cuda"
DTYPE = t.bfloat16

# Model config
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.1"
USE_QLORA = True  # Use QLoRA for memory efficiency

# Training config
OUTPUT_DIR = "./dpo_output"
HUB_REPO_ID = "eekay/mistral-7b-instruct-dpo"  # Hub repo to push trained model
LEARNING_RATE = 5e-7
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
NUM_TRAIN_EPOCHS = 1
MAX_LENGTH = 1024
MAX_PROMPT_LENGTH = 512
BETA = 0.1  # DPO beta parameter (controls deviation from reference model)
WARMUP_RATIO = 0.1
LOGGING_STEPS = 10
SAVE_STEPS = 500
EVAL_STEPS = 500

# ============================= Load Model ============================= #

def load_model_and_tokenizer(model_id: str = MODEL_ID, use_qlora: bool = USE_QLORA):
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

#%%

def load_ultrafeedback_dataset():
    """Load and format the UltraFeedback binarized dataset for DPO."""
    
    print("Loading UltraFeedback binarized dataset...")
    
    dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    
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

#%%

def train_dpo(
    model,
    tokenizer,
    dataset,
    peft_config=None,
    output_dir: str = OUTPUT_DIR,
    use_wandb: bool = True,
    hub_repo_id: str = HUB_REPO_ID,
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
        lr_scheduler_type="cosine",
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
    
    # Train
    print("Starting DPO training...")
    trainer.train()
    
    # Save final model
    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Push to Hub
    if hub_repo_id:
        print(f"Pushing model to Hub: {hub_repo_id}")
        trainer.push_to_hub(repo_id=hub_repo_id, commit_message="DPO training complete")
        tokenizer.push_to_hub(repo_id=hub_repo_id, commit_message="Add tokenizer")
        print(f"Model pushed to https://huggingface.co/{hub_repo_id}")
    
    if use_wandb:
        wandb.finish()
    
    return trainer

#%%

if __name__ == "__main__":
    # Load model and tokenizer
    model, tokenizer, peft_config = load_model_and_tokenizer()
    
    # Load dataset
    dataset = load_ultrafeedback_dataset()
    
    # Train
    trainer = train_dpo(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        peft_config=peft_config,
        use_wandb=True,
    )
    
    print("DPO training complete!")

#%%