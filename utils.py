#  Copyright 2023 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from logging import getLogger
from typing import Optional, Union, List
from collections.abc import Iterable
import functools

import torch
from torch import nn
from transformers.pytorch_utils import Conv1D

from constants import (
    BLOCK_PATTERNS,
    SEQLEN_KEYS_TRANFORMERS,
    ATTN_QKV_PATTERNS,
    ATTN_OUT_PATTENRS,
    FC1_PATTERN,
    FC2_PATTERN,
)

logger = getLogger(__name__)


def get_layers(module: nn.Module,
               layers: List = [Conv1D, nn.Conv2d, nn.Linear],
               prefix: Optional[str] = None,
               skip: Optional[List] = None,
               name: str = ""):
    """
    Get all the layers with a specific prefix in the module
    Args:
	module (`nn.Module`):
            The module that contains our layers
        layers (`list`, defaults to `[Conv1D, nn.Conv2d, nn.Linear]`):
            Type of the layers that we want to get
        prefix (`Optional[str]`, defaults to `None`):
            Prefix of layers
        name (`str`, defaults to `""`):
            Used for recursion. Don't modify

    Returns:
	`Dict[str,Union[Conv1D, nn.Conv2d, nn.Linear]]`: Mapping of the name of the layer and the actual layer
    """
    if skip is None:
        skip = []
    for layer in layers:
        if isinstance(module, layer):
            #print(name, skip)
            if prefix is not None:
                if name.startswith(prefix) and all(pattern not in name for pattern in skip):
                    return {name: module}
            else:
                if all(pattern not in name for pattern in skip):
                    return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            get_layers(child,
                       layers=layers,
                       prefix=prefix,
                       skip=skip,
                       name=name + "." + name1 if name != "" else name1))
    return res

def get_block_name_with_pattern(model: nn.Module):
    """
    Get the name of the module that contains the transformers blocks by checking if any modules has a specific pattern

    Args:
        model (`nn.Module`):
        The input model
    Returns:
        `str`: The name of the module that contains the Transformer blocks.
    """
    modules_names = [n for n, _ in model.named_modules()]
    for pattern_candidate in BLOCK_PATTERNS:
        pattern_candidate = pattern_candidate
        if any(pattern_candidate in name for name in modules_names):
            return pattern_candidate
    raise ValueError(
        "Block pattern could not be match. Pass `block_name_to_quantize` argument in `quantize_model`"
    )


def get_preceding_modules(model: nn.Module,
                          module_name: str,
                          reverse: bool = False):
    previous_module_name = []
    stop_adding = False

    def _get_preceding_modules(model: nn.Module,
                               module_name: str,
                               name: str = ""):
        nonlocal stop_adding
        modules = model.named_children()
        if reverse:
            modules = reversed(list(modules))
        for name_bis, child in modules:
            new_name = name + "." + name_bis if name != "" else name_bis
            if new_name == module_name:
                stop_adding = True
                break
            _get_preceding_modules(child, module_name, name=new_name)
        if not stop_adding:
            previous_module_name.append(name)
        return previous_module_name

    return _get_preceding_modules(model, module_name)


def get_device(obj: Union[torch.Tensor, nn.Module]):
    if isinstance(obj, torch.Tensor):
        return obj.device
    return next(obj.parameters()).device


def get_seqlen(model: nn.Module):
    if hasattr(model, "config"):
        model_config = model.config.to_dict()
        if any(k in model_config for k in SEQLEN_KEYS_TRANFORMERS):
            for key in SEQLEN_KEYS_TRANFORMERS:
                if key in model_config:
                    return model_config[key]
    logger.info(
        "We couldn't get the model sequence length. Setting it to 2048. You can overwrite this value by passing `model_seqlen` in` GPTQQuantizer`"
    )
    return 2048


def recurse_getattr(obj, attr: str):
    """
    Recursive `getattr`.

    Args:
        obj:
            A class instance holding the attribute.
        attr (`str`):
            The attribute that is to be retrieved, e.g. 'attribute1.attribute2'.
    """

    def _getattr(obj, attr):
        if isinstance(obj, Iterable):
            return obj[int(attr)]
        return getattr(obj, attr)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def get_layers_for_scaling(model):
    model_name = str(model.__class__).lower()
    if "llama" in model_name or "mistral" in model_name:
        layers = [
            ("input_layernorm", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]),
            ("post_attention_layernorm", ["mlp.gate_proj", "mlp.up_proj"]),
            ("mlp.up_proj", ["mlp.down_proj"]),
        ]
        if model.config.num_key_value_heads == model.config.num_attention_heads:
            layers.append(("self_attn.v_proj", ["self_attn.o_proj"]))
    elif "qwen" in model_name:
        layers = [
            ("ln_1", ["attn.c_attn"]),
            ("ln_2", ["mlp.w2", "mlp.w1"]),
            ("mlp.w1", ["mlp.c_proj"]),
        ]
    elif "mixtral" in model_name:
        layers = [
            ("input_layernorm", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]),
            ("post_attention_layernorm", ["block_sparse_moe.gate"]),
        ]
        for i in range(model.config.num_local_experts):
            layers.append((f"block_sparse_moe.experts.{i}.w3", [f"block_sparse_moe.experts.{i}.w2"]))
            layers.append(("post_attention_layernorm", [f"block_sparse_moe.experts.{i}.w3", f"block_sparse_moe.experts.{i}.w1"]))
        if model.config.num_key_value_heads == model.config.num_attention_heads:
            layers.append(("self_attn.v_proj", ["self_attn.o_proj"]))
    elif "yi" in model_name:
        layers = [
            ("ln1", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]),
            ("ln2", ["mlp.gate_proj", "mlp.up_proj"]),
            ("mlp.up_proj", ["mlp.down_proj"]),
        ]
        if model.config.num_key_value_heads == model.config.num_attention_heads:
            layers.append(("self_attn.v_proj", ["self_attn.o_proj"]))
    else:
        raise ValueError(f"{model_name} not supported for merging SU/SV. Please set merge_suv to False")
    return layers


def split_block_to_sublayers(layers):
    qkv_layers = [name for name in layers if name in ATTN_QKV_PATTERNS]
    out_layers = [name for name in layers if name in ATTN_OUT_PATTENRS]
    fc1_layers = [name for name in layers if name in FC1_PATTERN]
    fc2_layers = [name for name in layers if name in FC2_PATTERN]
    if len(qkv_layers) + len(out_layers) + len(fc1_layers) + len(fc2_layers) != len(layers):
        logger.info("We could not infer the split for this model. will treating the block as a whole")
        return [layers]
    return [qkv_layers, out_layers, fc1_layers, fc2_layers]


def extract_susv_params(module):
    susv_params = []
    params = []
    for name, param in module.named_parameters():
        if param.requires_grad:
            if 'SU' in name or 'SV' in name:
                susv_params.append(param)
            else:
                params.append(param)
    return susv_params, params


def get_susv_adam(susv_params, params, ft_susv_lr, ft_lr):
    return torch.optim.Adam([
        {
            'params': susv_params,
            'lr': ft_susv_lr
        },
        {
            'params': params,
            'lr': ft_lr
        },
    ])

def calculate_mse_loss(layer, dataset):
    layer.eval()
    device = get_device(layer)
    total_loss = 0
    num_samples = 0
    with torch.no_grad():
        for layer_input, layer_input_kwargs, layer_output in dataset:
            layer_input = layer_input.to(device)
            layer_input_kwargs = {k: v.to(device) if isinstance(
                v, torch.Tensor) else v for k, v in layer_input_kwargs.items()}
            total_loss += nn.MSELoss()(
                layer(layer_input, **layer_input_kwargs)[0],
                layer_output.to(device)
            )
            num_samples += 1
    layer.train()
    return (total_loss / num_samples).cpu().item()


def calculate_ce_loss(layer, dataset):
    layer.eval()
    device = get_device(layer)
    total_loss = 0
    num_samples = 0
    with torch.no_grad():
        for layer_input, layer_output in dataset:
            layer_input = {k : v.to(device) for k, v in layer_input.items()}
            logits = layer(**layer_input)
            total_loss += nn.CrossEntropyLoss()(
                logits.view(-1, logits.shape[-1]),
                layer_output.to(device).view(-1, logits.shape[-1]),
            )
            num_samples += 1
    layer.train()
    return (total_loss / num_samples).cpu().item()