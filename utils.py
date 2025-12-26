import json
import os
import random
import re
import hashlib
import time
from pathlib import Path

import numpy as np
import plotly.express as px
from IPython import get_ipython
import functools
from tabulate import tabulate
import wandb

import torch as t
from torch import Tensor
from tqdm import trange, tqdm
import datasets
from datasets import Dataset

from transformer_lens import HookedTransformer, ActivationCache, HookedTransformerConfig
from transformer_lens.hook_points import HookPoint

import transformers
from transformers import AutoModelForCausalLM

from transformers import AutoTokenizer

purple = '\x1b[38;2;255;0;255m'
blue = '\x1b[38;2;0;0;255m'
brown = '\x1b[38;2;128;128;0m'
cyan = '\x1b[38;2;0;255;255m'
lime = '\x1b[38;2;0;255;0m'
yellow = '\x1b[38;2;255;255;0m'
red = '\x1b[38;2;255;0;0m'
pink = '\x1b[38;2;255;51;204m'
orange = '\x1b[38;2;255;51;0m'
green = '\x1b[38;2;5;170;20m'
gray = '\x1b[38;2;127;127;127m'
magenta = '\x1b[38;2;128;0;128m'
white = '\x1b[38;2;255;255;255m'
bold = '\033[1m'
underline = '\033[4m'
endc = '\033[0m'

def tec(): t.cuda.empty_cache()

def pearson(x, y):
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    std_x = (sum((xi - mean_x) ** 2 for xi in x)) ** 0.5
    std_y = (sum((yi - mean_y) ** 2 for yi in y)) ** 0.5
    return cov / (std_x * std_y)

PROBES_DIR = Path("./probes")

IPYTHON = get_ipython()
if IPYTHON is not None:
    IPYTHON.run_line_magic('load_ext', 'autoreload')
    IPYTHON.run_line_magic('autoreload', '2')

# ============================= model stuff ============================= #

class LinearProbe:
    def __init__(self, model, layer, act_name, device="cuda", hash_name=None):
        self.model = model
        self.layer = layer
        self.act_name = act_name
        self.probe = t.zeros((model.cfg.d_model), dtype=t.float32, device=device, requires_grad=True)
        
        # Generate unique hash name if not provided
        if hash_name is None:
            timestamp = str(time.time_ns())
            hash_input = f"{layer}_{act_name}_{timestamp}"
            self.hash_name = hashlib.sha256(hash_input.encode()).hexdigest()[:12]
        else:
            self.hash_name = hash_name

    @property
    def dtype(self):
        return self.probe.dtype

    def parameters(self):
        return iter([self.probe])

    def grad_norm(self):
        if self.probe.grad is None:
            return 0.0
        return self.probe.grad.norm().item()

    def weight_norm(self):
        return self.probe.norm().item()

    def forward(self, act: Tensor) -> Tensor:
        return self.probe @ act
    
    def get_pred(self, act: Tensor) -> Tensor:
        probe_pred = self.forward(act).item() * 10
        return probe_pred
    
    def save(self):
        """Save probe state to disk."""
        save_path = PROBES_DIR / f"{self.hash_name}.pt"
        state = {
            "probe": self.probe.detach().cpu(),
            "layer": self.layer,
            "act_name": self.act_name,
            "hash_name": self.hash_name,
        }
        t.save(state, save_path)
        return save_path
    
    @classmethod
    def load(cls, model, hash_name, device="cuda"):
        """Load probe from disk."""
        load_path = PROBES_DIR / f"{hash_name}.pt"
        if not load_path.exists():
            raise FileNotFoundError(f"Probe checkpoint not found: {load_path}")
        
        state = t.load(load_path, map_location=device)
        
        probe = cls(
            model=model,
            layer=state["layer"],
            act_name=state["act_name"],
            hash_name=state["hash_name"],
        )
        probe.probe = state["probe"].to(device).requires_grad_(True)
        
        return probe

class NonLinearProbe:
    def __init__(self, model, layer, act_name, device="cuda", hash_name=None):
        self.model = model
        self.layer = layer
        self.act_name = act_name
        self.l1 = t.nn.Linear(model.cfg.d_model, model.cfg.d_model, device=device)
        self.l2 = t.nn.Linear(model.cfg.d_model, 1, device=device)
        
        # Generate unique hash name if not provided
        if hash_name is None:
            timestamp = str(time.time_ns())
            hash_input = f"{layer}_{act_name}_{timestamp}"
            self.hash_name = hashlib.sha256(hash_input.encode()).hexdigest()[:12]
        else:
            self.hash_name = hash_name

    @property
    def dtype(self):
        return self.l1.weight.dtype

    def parameters(self):
        yield from self.l1.parameters()
        yield from self.l2.parameters()

    def grad_norm(self):
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
        return total_norm ** 0.5

    def weight_norm(self):
        total_norm = 0.0
        for p in self.parameters():
            total_norm += p.norm().item() ** 2
        return total_norm ** 0.5

    def forward(self, act: Tensor) -> Tensor:
        return self.l2(t.relu(self.l1(act))).squeeze()
    
    def get_pred(self, act: Tensor) -> Tensor:
        probe_pred = round(self.forward(act).detach().item() * 10)
        return probe_pred
    
    def save(self):
        """Save probe state to disk."""
        save_path = PROBES_DIR / f"{self.hash_name}.pt"
        state = {
            "l1": self.l1.state_dict(),
            "l2": self.l2.state_dict(),
            "layer": self.layer,
            "act_name": self.act_name,
            "hash_name": self.hash_name,
        }
        t.save(state, save_path)
        return save_path
    
    @classmethod
    def load(cls, model, hash_name, device="cuda"):
        """Load probe from disk."""
        load_path = PROBES_DIR / f"{hash_name}.pt"
        
        if not load_path.exists():
            raise FileNotFoundError(f"Probe checkpoint not found: {load_path}")
        
        state = t.load(load_path, map_location=device)
        
        probe = cls(
            model=model,
            layer=state["layer"],
            act_name=state["act_name"],
            hash_name=state["hash_name"],
        )
        probe.l1.load_state_dict(state["l1"])
        probe.l2.load_state_dict(state["l2"])
        
        return probe

def eval_probe(model, probe: LinearProbe, dataset, n_samples):
    """Evaluate probe on dataset samples, returning true and predicted scores."""
    true_scores = []
    pred_scores = []
    
    for i, ex in enumerate(tqdm(dataset, total=n_samples)):
        if i >= n_samples:
            break
            
        messages = [{"role":"user","content": ex["prompt"]}, {"role":"assistant","content":ex["response"]}]
        prompt_toks = model.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
        ).squeeze().to(model.cfg.device)
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
            probe_act = probe.forward(target_act).item()
        
        true_scores.append(score)
        pred_scores.append(probe_act*10)

    t.cuda.empty_cache()
    return true_scores, pred_scores

# =========================== dataset stuff =========================== #

def convert_hs_to_dpo_format(hs_dataset):
    """
    Convert HelpSteer2 dataset to HuggingFace conversational preference format.
    
    Args:
        hs_dataset: HelpSteer2 dataset (nvidia/HelpSteer2)
    
    Returns:
        Dataset with columns: prompt, chosen, rejected
        - prompt: [{"role": "user", "content": ...}]
        - chosen: [{"role": "assistant", "content": ...}]
        - rejected: [{"role": "assistant", "content": ...}]
    """
    from collections import defaultdict
    
    # Group by prompt
    prompt_groups = defaultdict(list)
    for i, ex in enumerate(hs_dataset):
        prompt_groups[ex["prompt"]].append(i)
    
    prompts = []
    chosen = []
    rejected = []
    
    for prompt_text, indices in tqdm(prompt_groups.items(), desc="Converting to DPO format"):
        if len(indices) < 2:
            continue  # Need at least 2 responses for preference pair
        
        # Compute scores and collect responses
        scored_responses = []
        for idx in indices:
            ex = hs_dataset[idx]
            avg_score = (ex["helpfulness"] + ex["correctness"] + ex["coherence"]) / 3
            scored_responses.append((avg_score, ex["response"]))
        
        # Shuffle for random tiebreaking, then sort by score descending
        random.shuffle(scored_responses)
        scored_responses.sort(key=lambda x: x[0], reverse=True)
        
        # Best response is chosen, worst is rejected
        best_response = scored_responses[0][1]
        worst_response = scored_responses[-1][1]
        
        prompts.append([{"role": "user", "content": prompt_text}])
        chosen.append([{"role": "assistant", "content": best_response}])
        rejected.append([{"role": "assistant", "content": worst_response}])
    
    return Dataset.from_dict({
        "prompt": prompts,
        "chosen": chosen,
        "rejected": rejected,
    })


def make_probe_dataset(*, ufb_dataset=None, uf_dataset=None, hs_dataset=None, balance_ratings=True):
    """
    Build a probe dataset from ultrafeedback (binarized or unbinarized) or HelpSteer2.
    
    Args:
        ufb_dataset: Binarized dataset (HuggingFaceH4/ultrafeedback_binarized)
        uf_dataset: Unbinarized dataset (openbmb/UltraFeedback)
        hs_dataset: HelpSteer2 dataset (nvidia/HelpSteer2)
        balance_ratings: Whether to balance by rating scores
    
    Returns:
        Dataset with columns: prompt, response, score, is_winner
    
    Exactly one of ufb_dataset, uf_dataset, or hs_dataset must be provided.
    """
    provided = sum(x is not None for x in [ufb_dataset, uf_dataset, hs_dataset])
    if provided != 1:
        raise ValueError("Exactly one of ufb_dataset, uf_dataset, or hs_dataset must be provided")
    
    prompts = []
    responses = []
    scores = []
    is_winner = []
    
    if hs_dataset is not None:
        # Process HelpSteer2 dataset
        # Group by prompt first
        from collections import defaultdict
        prompt_groups = defaultdict(list)
        for i, ex in enumerate(hs_dataset):
            prompt_groups[ex["prompt"]].append(i)
        
        for prompt, indices in tqdm(prompt_groups.items(), desc="Building probe dataset (HelpSteer2)"):
            # Collect responses with computed scores
            completions = []
            for idx in indices:
                ex = hs_dataset[idx]
                # Score = average of helpfulness, correctness, coherence (0-4 scale)
                # Scale to 1-10: (avg / 4) * 9 + 1
                avg_score = (ex["helpfulness"] + ex["correctness"] + ex["coherence"]) / 3
                scaled_score = (avg_score / 4) * 9 + 1  # Maps 0-4 to 1-10
                completions.append((scaled_score, ex["response"]))
            
            if len(completions) == 0:
                continue
            
            # Shuffle for random tiebreaking, then sort by score descending
            random.shuffle(completions)
            completions.sort(key=lambda x: x[0], reverse=True)
            
            # Top half are winners (typically 1 of 2)
            n_winners = max(1, len(completions) // 2)
            for rank, (score, response) in enumerate(completions):
                prompts.append(prompt)
                responses.append(response)
                scores.append(round(score))
                is_winner.append(rank < n_winners)
    
    elif uf_dataset is not None:
        # Process unbinarized dataset
        for example in tqdm(uf_dataset, desc="Building probe dataset (unbinarized)"):
            prompt = example["instruction"]
            completions = example["completions"]
            
            # Filter completions with valid scores and collect (index, score, completion)
            valid_completions = []
            for i, comp in enumerate(completions):
                score = comp.get("overall_score")
                if score is not None:
                    valid_completions.append((i, score, comp))
            
            if len(valid_completions) == 0:
                continue
            
            # Shuffle first for random tiebreaking, then sort by score descending
            random.shuffle(valid_completions)
            valid_completions.sort(key=lambda x: x[1], reverse=True)
            
            # Top 2 are winners
            for rank, (idx, score, comp) in enumerate(valid_completions):
                prompts.append(prompt)
                responses.append(comp["response"])
                scores.append(round(score))
                is_winner.append(rank < 2)  # 0 and 1 are winners
    else:
        # Process binarized dataset
        for example in tqdm(ufb_dataset, desc="Building probe dataset (binarized)"):
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
            is_winner.append(True)
            
            # Get rejected response and score (skip if duplicate of chosen)
            rejected_response = rejected_messages[1]["content"]
            rejected_score = example["score_rejected"]
            
            if rejected_response != chosen_response:
                prompts.append(prompt)
                responses.append(rejected_response)
                scores.append(round(rejected_score))
                is_winner.append(False)
    
    probe_dataset = Dataset.from_dict({
        "prompt": prompts,
        "response": responses,
        "score": scores,
        "is_winner": is_winner,
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

# ======================= completion merging ========================== #

def merge_model_completions(file1: str, file2: str, output_path: str) -> dict:
    """
    Merge completions from two models into a single JSON file.
    
    Args:
        file1: Path to first model's completions JSON file
        file2: Path to second model's completions JSON file
        output_path: Path where the merged JSON should be saved
    
    Returns:
        The merged data dict
    
    Output format:
        {
            "models": ["model1_name", "model2_name"],
            "completions": [
                {
                    "idx": 0,
                    "prompt": "...",
                    "completions": {
                        "model1_name": "generated text...",
                        "model2_name": "generated text..."
                    }
                },
                ...
            ]
        }
    
    Only includes examples where both models have a completion.
    """
    with open(file1, "r") as f:
        data1 = json.load(f)
    with open(file2, "r") as f:
        data2 = json.load(f)
    
    model1_name = data1.get("model", "model1")
    model2_name = data2.get("model", "model2")
    
    # Index completions by idx
    completions1 = {c["idx"]: c for c in data1.get("completions", [])}
    completions2 = {c["idx"]: c for c in data2.get("completions", [])}
    
    # Find common indices
    common_indices = set(completions1.keys()) & set(completions2.keys())
    
    merged_completions = []
    for idx in sorted(common_indices):
        c1 = completions1[idx]
        c2 = completions2[idx]
        
        merged_completions.append({
            "idx": idx,
            "prompt": c1.get("prompt", c2.get("prompt")),
            "completions": {
                model1_name: c1.get("new_completion", ""),
                model2_name: c2.get("new_completion", ""),
            }
        })
    
    merged_data = {
        "models": [model1_name, model2_name],
        "completions": merged_completions
    }
    
    with open(output_path, "w") as f:
        json.dump(merged_data, f, indent=2)
    
    print(f"Merged {len(merged_completions)} completions from {model1_name} and {model2_name}")
    print(f"(Skipped {len(completions1) + len(completions2) - 2*len(common_indices)} examples with missing completions)")
    print(f"Saved to {output_path}")
    
    return merged_data

# ======================= random stuff ========================== #

asst_special_tok_ids = [523, 28766, 489, 11143, 28766, 28767] # this is how '<|assistant|>' is tokenized. :/
def find_assistant_start(input):
    toks = input.tolist()
    for i in range(len(toks)):
        if toks[i:i+len(asst_special_tok_ids)] == asst_special_tok_ids:
            return i
    else:
        return -1

def to_str_toks(input: str, tokenizer) -> list[int]:
    toks = tokenizer(input)
    str_toks = [model.tokenizer.decode()]
    

# ============================ plotting stuff ============================ #

# yaxis_range = [lower, upper]
def line(y, renderer=None, **kwargs):
    '''
    Edit to this helper function, allowing it to take args in update_layout (e.g. yaxis_range).
    '''
    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
    if ("size" in kwargs_pre) or ("shape" in kwargs_pre):
        size = kwargs_pre.pop("size", None) or kwargs_pre.pop("shape", None)
        kwargs_pre["height"], kwargs_pre["width"] = size
    return_fig = kwargs_pre.pop("return_fig", False)
    if "margin" in kwargs_post and isinstance(kwargs_post["margin"], int):
        kwargs_post["margin"] = dict.fromkeys(list("tblr"), kwargs_post["margin"])
    if "xaxis_tickvals" in kwargs_pre:
        tickvals = kwargs_pre.pop("xaxis_tickvals")
        kwargs_post["xaxis"] = dict(
            tickmode = "array",
            tickvals = kwargs_pre.get("x", np.arange(len(tickvals))),
            ticktext = tickvals
        )
    if "hovermode" not in kwargs_post:
        kwargs_post["hovermode"] = "closest"
    if "use_secondary_yaxis" in kwargs_pre and kwargs_pre["use_secondary_yaxis"]:
        del kwargs_pre["use_secondary_yaxis"]
        if "labels" in kwargs_pre:
            labels: dict = kwargs_pre.pop("labels")
            kwargs_post["yaxis_title_text"] = labels.get("y1", None)
            kwargs_post["yaxis2_title_text"] = labels.get("y2", None)
            kwargs_post["xaxis_title_text"] = labels.get("x", None)
        for k in ["title", "template", "width", "height"]:
            if k in kwargs_pre:
                kwargs_post[k] = kwargs_pre.pop(k)
        fig = make_subplots(specs=[[{"secondary_y": True}]]).update_layout(**kwargs_post)
        y0 = to_numpy(y[0])
        y1 = to_numpy(y[1])
        x0, x1 = kwargs_pre.pop("x", [np.arange(len(y0)), np.arange(len(y1))])
        name0, name1 = kwargs_pre.pop("names", ["yaxis1", "yaxis2"])
        fig.add_trace(go.Scatter(y=y0, x=x0, name=name0), secondary_y=False)
        fig.add_trace(go.Scatter(y=y1, x=x1, name=name1), secondary_y=True)
    else:
        y = list(map(to_numpy, y)) if isinstance(y, list) and not (isinstance(y[0], int) or isinstance(y[0], float)) else to_numpy(y)
        names = kwargs_pre.pop("names", None)
        hover_text = kwargs_pre.pop("hover_text", None)
        fig = px.line(y=y, **kwargs_pre).update_layout(**kwargs_post)
        if names is not None:
            fig.for_each_trace(lambda trace: trace.update(name=names.pop(0)))
        if hover_text is not None:
            # Update the hover template to show custom text
            fig.for_each_trace(lambda trace: trace.update(
                hovertemplate='<b>Token:</b> %{customdata}<br>' +
                              '<b>Value:</b> %{y}<br>' +
                              '<b>Index:</b> %{x}<extra></extra>',
                customdata=hover_text
            ))
    return fig if return_fig else fig.show(renderer=renderer)

update_layout_set = {"xaxis_range", "yaxis_range", "hovermode", "xaxis_title", "yaxis_title", "colorbar", "colorscale", "coloraxis", "title_x", "bargap", "bargroupgap", "xaxis_tickformat", "yaxis_tickformat", "title_y", "legend_title_text", "xaxis_showgrid", "xaxis_gridwidth", "xaxis_gridcolor", "yaxis_showgrid", "yaxis_gridwidth", "yaxis_gridcolor", "showlegend", "xaxis_tickmode", "yaxis_tickmode", "margin", "xaxis_visible", "yaxis_visible", "bargap", "bargroupgap", "coloraxis_showscale", "xaxis_tickangle", "yaxis_scaleanchor", "xaxis_tickfont", "yaxis_tickfont"}

update_traces_set = {"textposition"}

def to_numpy(tensor):
    """
    Helper function to convert a tensor to a numpy array. Also works on lists, tuples, and numpy arrays.
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, (list, tuple)):
        array = np.array(tensor)
        return array
    elif isinstance(tensor, (t.Tensor, t.nn.parameter.Parameter)):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, (int, float, bool, str)):
        return np.array(tensor)
    else:
        raise ValueError(f"Input to to_numpy has invalid type: {type(tensor)}")

def reorder_list_in_plotly_way(L: list, col_wrap: int):
    '''
    Helper function, because Plotly orders figures in an annoying way when there's column wrap.
    '''
    L_new = []
    while len(L) > 0:
        L_new.extend(L[-col_wrap:])
        L = L[:-col_wrap]
    return L_new

def imshow(tensor: t.Tensor, renderer=None, **kwargs):
    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
    if ("size" in kwargs_pre) or ("shape" in kwargs_pre):
        size = kwargs_pre.pop("size", None) or kwargs_pre.pop("shape", None)
        kwargs_pre["height"], kwargs_pre["width"] = size
    facet_labels = kwargs_pre.pop("facet_labels", None)
    border = kwargs_pre.pop("border", False)
    return_fig = kwargs_pre.pop("return_fig", False)
    text = kwargs_pre.pop("text", None)
    xaxis_tickangle = kwargs_post.pop("xaxis_tickangle", None)
    # xaxis_tickfont = kwargs_post.pop("xaxis_tickangle", None)
    static = kwargs_pre.pop("static", False)
    if "color_continuous_scale" not in kwargs_pre:
        kwargs_pre["color_continuous_scale"] = "RdBu"
    if "color_continuous_midpoint" not in kwargs_pre:
        kwargs_pre["color_continuous_midpoint"] = 0.0
    if "margin" in kwargs_post and isinstance(kwargs_post["margin"], int):
        kwargs_post["margin"] = dict.fromkeys(list("tblr"), kwargs_post["margin"])
    fig = px.imshow(to_numpy(tensor), **kwargs_pre).update_layout(**kwargs_post)
    if facet_labels:
        # Weird thing where facet col wrap means labels are in wrong order
        if "facet_col_wrap" in kwargs_pre:
            facet_labels = reorder_list_in_plotly_way(facet_labels, kwargs_pre["facet_col_wrap"])
        for i, label in enumerate(facet_labels):
            print(fig.layout.annotations)
            fig.layout.annotations[i]['text'] = label
    if border:
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    if text:
        if tensor.ndim == 2:
            # if 2D, then we assume text is a list of lists of strings
            assert isinstance(text[0], list)
            assert isinstance(text[0][0], str)
            text = [text]
        else:
            # if 3D, then text is either repeated for each facet, or different
            assert isinstance(text[0], list)
            if isinstance(text[0][0], str):
                text = [text for _ in range(len(fig.data))]
        for i, _text in enumerate(text):
            fig.data[i].update(
                text=_text, 
                texttemplate="%{text}", 
                textfont={"size": 12}
            )
    # Very hacky way of fixing the fact that updating layout with xaxis_* only applies to first facet by default
    if xaxis_tickangle is not None:
        n_facets = 1 if tensor.ndim == 2 else tensor.shape[0]
        for i in range(1, 1+n_facets):
            xaxis_name = "xaxis" if i == 1 else f"xaxis{i}"
            fig.layout[xaxis_name]["tickangle"] = xaxis_tickangle
    return fig if return_fig else fig.show(renderer=renderer, config={"staticPlot": static})

