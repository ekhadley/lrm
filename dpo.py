#%%
import torch as t
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

# ================= CONFIGURATION =================
DEVICE = "cuda"
DTYPE = t.bfloat16


#%%

MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
    device_map=DEVICE,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

#%%

dataset = datasets.load_dataset(
    # "HuggingFaceH4/ultrafeedback_binarized",
    "HuggingFaceH4/ultrafeedback_binarized",
    split="train_prefs"
)

#%%

training_args = DPOConfig(
    output_dir="./dpo_output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-7,
    beta=0.1,
    max_length=512,
    max_prompt_length=256,
    num_train_epochs=1,
    logging_steps=10,
    bf16=True,
)

#%%

trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

#%%

run_training = False
if run_training:
    trainer.train()
    trainer.save_model("./dpo_final")

