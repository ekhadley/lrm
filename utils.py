import json
import os
import random
import re

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

IPYTHON = get_ipython()
if IPYTHON is not None:
    IPYTHON.run_line_magic('load_ext', 'autoreload')
    IPYTHON.run_line_magic('autoreload', '2')

# ============================= model stuff ============================= #





# =========================== dataset stuff =========================== #

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

