#%%
import torch as t
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

# ================= CONFIGURATION =================
DEVICE = "cuda"
DTYPE = t.bfloat16

#%%

# MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
# MODEL_NAME = "mistralai/Mistral-7B-v0.1"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=DTYPE,
    # device_map=DEVICE,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

#%%

from utils import convert_hs_to_dpo_format

rating_dataset = datasets.load_dataset("nvidia/HelpSteer2", split="train")
dpo_dataset = convert_hs_to_dpo_format(rating_dataset)

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

trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dpo_dataset,
    processing_class=tokenizer,
)

#%%

trainer.train()
trainer.save_model("./dpo_final")

