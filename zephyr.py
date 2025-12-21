
#%%
from utils import *

DEVICE = "cuda"
DTYPE = t.bfloat16 

#%%

MODEL_HF_NAME = "HuggingFaceH4/zephyr-7b-beta"
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_HF_NAME,
    torch_dtype=DTYPE,
    device_map=DEVICE
)
hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_HF_NAME)

uf = dataset = datasets.load_dataset("openbmb/ultrafeedback", split="train")
ufb = dataset = datasets.load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")

#%%

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

#%% example response generation

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

resp_ids = model.generate(
    prompt_toks,
    do_sample=True,
    verbose=True,
    max_new_tokens=250
)
print(model.tokenizer.decode(resp_ids.squeeze()))

#%%

def make_probe_dataset(ufb_dataset=None, split="train_prefs", balance_ratings=False):
    if ufb_dataset is None:
        ufb_dataset = datasets.load_dataset(
            "HuggingFaceH4/ultrafeedback_binarized", 
            split=split
        )
    
    prompts = []
    responses = []
    scores = []
    
    for example in tqdm(ufb_dataset, desc="Building probe dataset"):
        # Extract prompt from the first message (user message)
        # The chosen/rejected fields are lists of message dicts
        chosen_messages = example["chosen"]
        rejected_messages = example["rejected"]
        
        # Get prompt from user message (first message in the conversation)
        prompt = chosen_messages[0]["content"]
        
        # Get chosen response and score
        chosen_response = chosen_messages[1]["content"]
        chosen_score = example["score_chosen"]
        
        prompts.append(prompt)
        responses.append(chosen_response)
        scores.append(round(chosen_score))
        
        # Get rejected response and score (skip if duplicate of chosen)
        rejected_response = rejected_messages[1]["content"]
        rejected_score = example["score_rejected"]
        
        if rejected_response != chosen_response:
            prompts.append(prompt)
            responses.append(rejected_response)
            scores.append(round(rejected_score))
    
    probe_dataset = Dataset.from_dict({
        "prompt": prompts,
        "response": responses,
        "score": scores,
    })
    
    if balance_ratings:
        # Group indices by rating
        rating_indices = {r: [] for r in range(1, 11)}
        for i, score in enumerate(scores):
            if 1 <= score <= 10:
                rating_indices[score].append(i)
        
        # Find minimum count across all ratings that have at least one example
        counts = [len(indices) for indices in rating_indices.values() if len(indices) > 0]
        min_count = min(counts) if counts else 0
        
        # Sample min_count indices from each rating
        balanced_indices = []
        for r in range(1, 11):
            if len(rating_indices[r]) >= min_count:
                balanced_indices.extend(random.sample(rating_indices[r], min_count))
        
        # Shuffle and select
        random.shuffle(balanced_indices)
        probe_dataset = probe_dataset.select(balanced_indices)
    
    return probe_dataset

dataset = make_probe_dataset(balance_ratings=True)
print(dataset)
print(len(dataset))
print(dataset[0])
print(dataset[1])
print(dataset[2])


#%%