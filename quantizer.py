# Modified from https://github.com/huggingface/optimum/blob/main/optimum/gptq/quantizer.py
# Copyright 2023 HuggingFace Inc. team and GPTQ and AutoGPTQ authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import gc
import json
import os
import copy
from logging import getLogger
from typing import Any, Dict, List, Optional, Union

import torch
from torch import nn
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers.pytorch_utils import Conv1D
from accelerate import (
    Accelerator,
    cpu_offload_with_hook,
    init_empty_weights,
    load_checkpoint_and_dispatch
)
from accelerate.hooks import remove_hook_from_module
from safetensors.torch import load_file
from huggingface_hub import snapshot_download

import register_lib
from constants import QUIP_CONFIG
from data import get_dataset, prepare_dataset
from utils import (get_block_name_with_pattern, get_device, get_layers,
                   get_preceding_modules, get_seqlen, recurse_getattr,
                   get_layers_for_scaling, split_block_to_sublayers,
                   extract_susv_params, get_susv_adam, calculate_mse_loss,
                   calculate_ce_loss)
from quip import QUIP
from qlinear import QuantLinear
from codebook import codebook_id

logger = getLogger(__name__)


class QuipQuantizer(object):
    r"""
    A simple API for QUIP Quantization
    """

    def __init__(
        self,
        codebook: str,
        dataset: str = "",
        nsamples: int = 4096,
        model_seqlen: int = 2048,
        quip_tune_iters: int = 10,
        sigma_reg: float = 0.01,
        rescale_WH: bool = False,
        use_rand: bool = True,
        scale_override: float = -1,
        opt_resid_scale: float = -1,
        per_channel: bool = False,
        block_name_to_quantize: Optional[str] = None,
        module_name_preceding_first_block: Optional[List[str]] = None,
        batch_size: int = 4,
        inference: bool = False,
        cache_on_gpu: bool = False,
        modules_to_not_convert: Optional[List] = None,
        merge_suv: bool = False,
        ft_lr: float = 5e-5,
        ft_susv_lr: float = 5e-4,
        ft_epochs: int = 5,
        ft_train_size: int = 384,
        ft_valid_size: int = 128,
        ft_batch_size: int = 8,
        ft_valid_freq: int = 1,
        ft_early_stop: int = 3,
        ft_embedding: bool = False,
        *args,
        **kwargs,
    ):
        self.dataset = dataset
        self.nsamples = nsamples
        self.quip_tune_iters = quip_tune_iters
        self.sigma_reg = sigma_reg
        self.model_seqlen = model_seqlen
        self.rescale_WH = rescale_WH
        self.use_rand = use_rand
        self.scale_override = scale_override
        self.opt_resid_scale = opt_resid_scale
        self.per_channel = per_channel
        self.block_name_to_quantize = block_name_to_quantize
        self.module_name_preceding_first_block = module_name_preceding_first_block
        self.batch_size = batch_size
        self.cache_on_gpu = cache_on_gpu
        self.modules_to_not_convert = modules_to_not_convert
        self.merge_suv = merge_suv
        self.quant_method = "QUiP"
        self.ft_lr = ft_lr
        self.ft_susv_lr = ft_susv_lr
        self.ft_epochs = ft_epochs
        self.ft_train_size = ft_train_size
        self.ft_valid_size = ft_valid_size
        self.ft_batch_size = ft_batch_size
        self.ft_update_freq = ft_batch_size // batch_size
        self.ft_valid_freq = ft_valid_freq
        self.ft_early_stop = ft_early_stop
        self.ft_embedding = ft_embedding
        if self.ft_epochs > 0:
            self.all_samples = self.nsamples + self.ft_train_size + self.ft_valid_size
        else:
            self.all_samples = self.nsamples

        if self.ft_epochs > 0 and self.merge_suv:
            raise ValueError("finetune mode is incompatible with merge_suv")
        if codebook not in ["D4", "E8P12", "HI", "E8P12RVQ3B", "E8P12RVQ4B"]:
            raise ValueError("Invalid codebook, has to be D4 or E8P12 or HI")
        self.codebook = codebook_id[codebook](inference=inference,
                                              opt_resid_scale=opt_resid_scale)

        if not (0 < self.sigma_reg < 1):
            raise ValueError("damp_percent must between 0 and 1.")

    def to_dict(self):
        """
        Returns the args in dict format.
        """
        return {
            "quant_method": "QUiP",
            "rescale_WH": self.rescale_WH,
            "use_rand": self.use_rand,
            "codebook": self.codebook.id,
            "codesz": self.codebook.codesz,
            "idx_dtype": str(self.codebook.idx_dtype),
            "merge_suv": self.merge_suv,
            "per_channel": self.per_channel,
            "opt_resid_scale": self.opt_resid_scale,
            "modules_to_not_convert": self.modules_to_not_convert
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """
        Instantiates a `QuipQuantizer` using config_dict as kwargs

        Args:
            config_dict (`Dict[str,Any]`):
                quantization config

        Returns:
            `QuipQuantizer`:  The quantizer object instantiated from those parameters.
        """
        return cls(**config_dict)

    def convert_model(self, model: nn.Module, train: bool = False):
        """
        Convert the model to a Quip model by getting and replacing the layers.

        Args:
            model (`nn.Module`):
                Model to be converted
            train (`bool`):
                Finetune mode or inference mode

        """
        if self.block_name_to_quantize is None:
            self.block_name_to_quantize = get_block_name_with_pattern(model)
        block_name = self.block_name_to_quantize
        layers_to_be_replaced = get_layers(model, prefix=block_name, skip=self.modules_to_not_convert)
        self._replace_by_quant_layers(model, layers_to_be_replaced, train=train)

        return model

    def get_no_split_module_classes(self, model):
        """
        Get the modules that should not be split across multiple devices.
        Args:
            model (`nn.Module`):
                The input model
        """

        block_class_name = recurse_getattr(
            model, self.block_name_to_quantize)[0].__class__.__name__
        no_split_module_classes = [block_class_name]
        return no_split_module_classes

    def _replace_by_quant_layers(self,
                                 module: nn.Module,
                                 names: List[str],
                                 name: str = "",
                                 train: bool = False):
        """
        Replaces linear layers in `module` by `QuantLinear`

        Args:
            module (`nn.Module`):
                Module to quantize
            names (`List[str]`):
                List of names of the module to quantize
            name (`str`, defaults to `""`):
                To keep track of the name of the current module
            train (`bool`, defaults to False):
                Finetune mode or inference mode
        """
        if isinstance(module, QuantLinear):
            return
        for attr in dir(module):
            try:
                layer = getattr(module, attr)
            except:
                continue
            name1 = name + "." + attr if name != "" else attr
            if name1 in names:
                device = get_device(layer)
                delattr(module, attr)
                if isinstance(layer, nn.Linear):
                    in_features = layer.in_features
                    out_features = layer.out_features
                elif isinstance(layer, nn.Conv2d):
                    in_features = layer.in_channels
                    out_features = layer.out_channels
                elif isinstance(layer, Conv1D):
                    in_features = layer.weight.shape[0]
                    out_features = layer.weight.shape[1]
                cb = codebook_id[self.codebook.id](
                    inference=True,
                    opt_resid_scale=self.opt_resid_scale
                )
                new_layer = QuantLinear(in_features,
                                        out_features,
                                        cb,
                                        bias=(layer.bias is not None),
                                        use_rand=self.use_rand,
                                        per_channel=self.per_channel,
                                        weight_dtype=layer.weight.dtype,
                                        train=train)
                new_layer.device = device
                if device != torch.device("meta"):
                    new_layer =	new_layer.to(device)
                #setattr(module, attr, new_layer.to(device))
                setattr(module, attr, new_layer)
        for name1, child in module.named_children():
            self._replace_by_quant_layers(
                child, names, name + "." + name1 if name != "" else name1,
                train=train)

    @torch.no_grad()
    def quantize_model(self, model: nn.Module, tokenizer: Any):
        """
        Quantizes the model using the dataset

        Args:
            model (`nn.Module`):
                The model to quantize
            tokenizer (`Any`):
                The tokenizer to use in order to prepare the dataset. You can pass either:
                    - A custom tokenizer object.
                    - A string, the *model id* of a predefined tokenizer hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                      user or organization name, like `dbmdz/bert-base-german-cased`.
                    - A path to a *directory* containing vocabulary files required by the tokenizer, for instance saved
                      using the [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.
        Returns:
            `nn.Module`: The quantized model
        """
        model.eval()
        # For Transformer model
        has_config = False
        has_device_map = False
        origin_dtype = model.dtype
        if hasattr(model, "config"):
            has_config = True
            use_cache = model.config.use_cache
            model.config.use_cache = False

        if hasattr(model, "hf_device_map"):
            devices = list(model.hf_device_map.values())
            if "disk" in devices:
                raise ValueError(
                    "disk offload is not supported with QUiP quantization")
            if "cpu" in devices and len(model.hf_device_map) > 1:
                logger.info(
                    "Cpu offload is not recommended. There might be some issues with the memory"
                )
                hook = None
                for name, device in model.hf_device_map.items():
                    if device == "cpu":
                        module = recurse_getattr(model, name)
                        remove_hook_from_module(module, recurse=True)
                        module, hook = cpu_offload_with_hook(
                            module, prev_module_hook=hook)
            # If the model has a device_map, we don't move to model. We have already dispatched the hook that will do the work
            has_device_map = True

        if self.model_seqlen is None:
            self.model_seqlen = get_seqlen(model)

        device = get_device(model)

        # Step 1: Prepare the data
        if isinstance(tokenizer, str):
            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            except Exception:
                raise ValueError(
                    f"""We were not able to get the tokenizer using `AutoTokenizer.from_pretrained`
                    with the string that you have passed {tokenizer}. If you have a custom tokenizer, you can pass it as input.
                    For now, we only support quantization for text model. Support for vision, speech and multimodel will come later."""
                )

        dataset = get_dataset(self.dataset,
                              tokenizer,
                              nsamples=self.all_samples,
                              seqlen=self.model_seqlen,
                              split="train")

        dataset = prepare_dataset(dataset,
                                  batch_size=self.batch_size)

        # Step 2: get the input of the 1st block
        # To do that, we need to put the modules preceding the first block on the same device as the first bloc.
        # Then we run the model and it will stop at the first bloc as we added a prehook that raise an Exception after storing the inputs.

        layer_inputs = []
        layer_outputs = []
        layer_input_kwargs = []

        if self.block_name_to_quantize is None:
            self.block_name_to_quantize = get_block_name_with_pattern(model)

        if self.module_name_preceding_first_block is None:
            self.module_name_preceding_first_block = get_preceding_modules(
                model, self.block_name_to_quantize)

        blocks = recurse_getattr(model, self.block_name_to_quantize)

        # put modules from module_name_preceding_first_block on cuda
        for module_name in self.module_name_preceding_first_block:
            module = recurse_getattr(model, module_name)
            if module is None:
                raise ValueError(
                    f"Module {module_name} was not found in model")
            module = module.float()
            if not has_device_map:
                module = module.to(0)

        blocks[0] = blocks[0].float()
        if not has_device_map:
            blocks[0] = blocks[0].to(0)

        def store_input_hook(_, input, *args):
            kwargs = args[0]
            input = input[0]
            if input is None:
                if "hidden_states" in kwargs:
                    input = kwargs["hidden_states"]
                else:
                    raise ValueError("No input value found in the foward pass")
            layer_inputs.append(
                input.to("cpu") if not self.cache_on_gpu else input)
            other_kwargs = {}
            for k, v in kwargs.items(
            ):  # make sure other arguments also be captured
                if k not in ["hidden_states"]:
                    other_kwargs[k] = v.to("cpu") if not self.cache_on_gpu and isinstance(v, torch.Tensor) else v
            layer_input_kwargs.append(other_kwargs)
            raise ValueError

        handle = blocks[0].register_forward_pre_hook(store_input_hook,
                                                     with_kwargs=True)
        for data in dataset:
            for k, v in data.items():
                data[k] = v.to(0)
            try:
                model(**data)
            except ValueError:
                pass
            for k, v in data.items():
                data[k] = v.cpu()

        handle.remove()
        if not has_device_map:
            for module_name in self.module_name_preceding_first_block:
                module = recurse_getattr(model, module_name)
                if module is None:
                    raise ValueError(
                        f"Module {module_name} was not found in model")
                module = module.to(origin_dtype).to(device)

        torch.cuda.empty_cache()

        # Step 3: Quantize the blocks
        quantizers = {}
        for i, block in enumerate(
                tqdm(
                    blocks,
                    desc=f"Quantizing {self.block_name_to_quantize} blocks ")):
            logger.info(
                f"Start quantizing block {self.block_name_to_quantize} {i + 1}/{len(blocks)}"
            )
            # move block to cuda if needed
            if not has_device_map or get_device(block) == torch.device("cpu"):
                block = block.to(0)
            block = block.float()
            layers = get_layers(block, skip=self.modules_to_not_convert)
            layers_name_list = list(layers.keys())
            logger.info(f"Module to quantize {layers_name_list}")
            if self.merge_suv:
                scale_list = get_layers_for_scaling(model)
                for prev_name, current_names in scale_list:
                    prev_op = recurse_getattr(block, prev_name)
                    if not hasattr(prev_op, "SV"):
                        prev_op.register_buffer("SV",
                            (torch.randn(prev_op.weight.shape[0],
                                         device=get_device(prev_op)).sign() +1e-5).sign()
                        )
                    for current_name in current_names:
                        current_op = recurse_getattr(block, current_name)
                        current_op.SU = prev_op.SV.clone()

            quant_method = {}
            handles = []
            for name in layers_name_list:
                quant_method[name] = QUIP(layers[name], self.codebook)

                def add_batch(name):
                    def tmp(_, input, output):
                        quant_method[name].add_batch(
                            input[0].data, output.data)

                    return tmp

                # because it adding a hook will replace the old one.
                handles.append(layers[name].register_forward_hook(
                    add_batch(name)))
            # update Hessian for each layer in subset_layers thanks to the hook
            block_dev = get_device(block)
            for j in range(self.nsamples // self.batch_size):
                layer_input = layer_inputs[j].to(block_dev
                    ) if not self.cache_on_gpu else layer_inputs[j]
                layer_input_kwarg = {k: v.to(block_dev
                    ) if not self.cache_on_gpu and isinstance(v, torch.Tensor) else v for k, v in layer_input_kwargs[j].items()}
                layer_output = block(layer_input,
                                     **layer_input_kwarg)[0]
                layer_outputs.append(layer_output.cpu(
                    ) if not self.cache_on_gpu else layer_output)
            # remove hook
            for h in handles:
                h.remove()
            # add sample for finetune
            if self.ft_epochs > 0:
                for j in range(self.nsamples // self.batch_size, len(dataset)):
                    layer_input = layer_inputs[j].to(block_dev
                        ) if not self.cache_on_gpu else layer_inputs[j]
                    layer_input_kwarg = {k: v.to(block_dev
                        ) if not self.cache_on_gpu and isinstance(v, torch.Tensor) else v for k, v in layer_input_kwargs[j].items()}
                    layer_output = block(layer_input,
                                         **layer_input_kwarg)[0]
                    layer_outputs.append(layer_output.cpu(
                        ) if not self.cache_on_gpu else layer_output)

            subset_name_lists = split_block_to_sublayers(layers_name_list)
            for j, subset_name_list in tqdm(
                    enumerate(subset_name_lists),
                    leave=False,
                    desc="Quantizing layers inside the block"):
                subset_layers = {
                    name: layers[name]
                    for name in subset_name_list
                }

                for name in subset_name_list:
                    logger.info(
                        f"Quantizing {name} in block {i + 1}/{len(blocks)}...")
                    attr = quant_method[name].quant(
                        rescale_WH=self.rescale_WH,
                        sigma_reg=self.sigma_reg,
                        quip_tune_iters=self.quip_tune_iters,
                        scale_override=self.scale_override,
                        use_rand=self.use_rand,
                        per_channel=self.per_channel)
                    quantizers[
                        f"{self.block_name_to_quantize}.{i}.{name}"] = attr
                    quant_method[name].free()
                del subset_layers
                # replace to quant layer
                self._replace_by_quant_layers(block, subset_name_list, train=True)
                for name in subset_name_list:
                    quant_layer = recurse_getattr(block, name)
                    quant_layer.pack(layers[name],
                                     quantizers[f"{self.block_name_to_quantize}.{i}.{name}"])
                    layers[name].to("cpu")
                    quant_layer.to(block_dev)

                # Block-wise finetune
                if self.ft_epochs > 0 and j < len(subset_name_lists) - 1:
                    torch.set_grad_enabled(True)
                    block.train()
                    # cache the weight for faster finetune
                    for name in subset_name_list:
                        quant_layer = recurse_getattr(block, name)
                        quant_layer.calc_weight()
                    susv_params, params = extract_susv_params(block)
                    optim = get_susv_adam(susv_params, params, self.ft_susv_lr, self.ft_lr)
                    train_size = self.ft_train_size // self.batch_size
                    valid_size = self.ft_valid_size // self.batch_size
                    train_dataset = list(zip(
                        layer_inputs[-train_size-valid_size:-valid_size],
                        layer_input_kwargs[-train_size-valid_size:-valid_size],
                        layer_outputs[-train_size-valid_size:-valid_size]))
                    valid_dataset = list(zip(
                        layer_inputs[-valid_size:],
                        layer_input_kwargs[-valid_size:],
                        layer_outputs[-valid_size:],
                    ))
                    best_loss = calculate_mse_loss(block, valid_dataset)
                    best_sd = {k: v.cpu() for k, v in block.state_dict().items()}
                    logger.info(f"Block {i + 1} initial loss {best_loss}")
                    scaler = torch.cuda.amp.GradScaler(enabled=True)
                    worse_ct = 0
                    for epoch in range(self.ft_epochs):
                        for bidx, (layer_input, layer_input_kwarg, target_output
                                   ) in enumerate(train_dataset):
                            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                                layer_input_kwarg = {k: v.to(block_dev
                                    ) if not self.cache_on_gpu and isinstance(v, torch.Tensor) else v for k, v in layer_input_kwarg.items()}
                                output = block(layer_input.to(block_dev),
                                               **layer_input_kwarg)[0]
                                loss = nn.MSELoss()(output, target_output.to(block_dev))
                                loss = loss / self.ft_update_freq
                            scaler.scale(loss).backward()
                            if bidx % self.ft_update_freq == self.ft_update_freq - 1 or (
                                    bidx == len(train_dataset) - 1):
                                scaler.step(optim)
                                scaler.update()
                                optim.zero_grad()

                        if epoch % self.ft_valid_freq == self.ft_valid_freq - 1:
                            test_loss = calculate_mse_loss(block, valid_dataset)
                            if test_loss < best_loss:
                                logger.info(
                                    f"Block {i + 1} @ epoch {epoch} new loss {test_loss} old loss {best_loss} BETTER"
                                )
                                best_loss = test_loss
                                best_sd = {k: v.cpu() for k, v in block.state_dict().items()}
                                worse_ct = 0
                            else:
                                worse_ct += 1
                                if worse_ct >= self.ft_early_stop:
                                    break
                    block.eval()
                    block.load_state_dict(best_sd)
                    for name in subset_name_list:
                        quant_layer = recurse_getattr(block, name)
                        del quant_layer.W
                    del optim, train_dataset, valid_dataset
                    torch.cuda.empty_cache()
                    torch.set_grad_enabled(False)

            # put back to device
            block = block.to(origin_dtype)
            if not has_device_map:
                blocks[i] = block.to(device)
            del layers, layer_inputs, quant_method
            layer_inputs, layer_outputs = layer_outputs, []
            gc.collect()
            torch.cuda.empty_cache()

        del layer_input_kwargs, quantizers
        gc.collect()
        torch.cuda.empty_cache()

        if self.merge_suv:
            for _, module in model.named_modules():
                if (isinstance(module, nn.LayerNorm) or "rmsnorm" in str(module.__class__).lower()
                    ) and hasattr(module, "SV"):
                    module.weight.div_(module.SV)
                    if hasattr(module, "bias") and module.bias is not None:
                        module.bias.div_(module.SV)
                    module.SV = None
                if isinstance(module, nn.Linear):
                    if hasattr(module, "SV"):
                        module.weight.div_(module.SV.unsqueeze(-1))
                        if hasattr(module, "bias") and module.bias is not None:
                            module.bias.div_(module.SV)
                        module.SV = None
                    if hasattr(module, "SU"):
                        module.weight.div_(module.SU)
                        module.SU = None

        # Step 4: End2end finetune
        if self.ft_epochs > 0:
            module_names_after_last_block = get_preceding_modules(
                model, self.block_name_to_quantize, reverse=True)
            module = nn.Sequential(*[
                recurse_getattr(model, name)
                for name in reversed(module_names_after_last_block)
            ])
            module = module.float()
            if not has_device_map:
                module = module.to(0)

            train_size = self.ft_train_size // self.batch_size
            valid_size = self.ft_valid_size // self.batch_size
            layer_inputs = layer_inputs[-train_size - valid_size:]
            gc.collect()
            for layer_input in layer_inputs:
                layer_input = layer_input.to(get_device(
                    module)) if not self.cache_on_gpu else layer_input
                layer_output = module(layer_input).softmax(dim=-1).float()
                layer_outputs.append(layer_output.cpu(
                    ) if not self.cache_on_gpu else layer_output)
            del layer_inputs

            module = module.to(origin_dtype)
            if not has_device_map:
                module = module.to(device)

            model = model.float()
            model.train()
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            if not has_device_map:
                model = model.to(0)

            torch.set_grad_enabled(True)
            if not self.ft_embedding:
                model.get_input_embeddings().weight.requires_grad = False
                model.get_output_embeddings().weight.requires_grad = False
            susv_params, params = extract_susv_params(model)
            optim = get_susv_adam(susv_params, params, self.ft_susv_lr, self.ft_lr)
            train_dataset = list(zip(
                dataset[-train_size - valid_size: -valid_size],
                layer_outputs[-train_size - valid_size: -valid_size]))
            valid_dataset = list(zip(
                dataset[-valid_size:],
                layer_outputs[-valid_size:],
            ))

            best_loss = calculate_ce_loss(model, valid_dataset)
            scaler = torch.cuda.amp.GradScaler(enabled=True)

            # best_sd = copy.deepcopy(model.state_dict())
            best_sd = {k: v.cpu() for k, v in model.state_dict().items()}
            logger.info(f"End2end initial loss {best_loss}")
            worse_ct = 0
            for epoch in range(self.ft_epochs):
                for bidx, (layer_input, target_output) in enumerate(train_dataset):
                    with torch.autocast(device_type="cuda",
                                        dtype=torch.float16,
                                        enabled=True):
                        layer_input = {k : v.to(0) for k, v in layer_input.items()}
                        output = model(**layer_input)[0]
                        loss = nn.CrossEntropyLoss()(
                            output.view(-1, output.shape[-1]),
                            target_output.to(output.device).view(-1, output.shape[-1]))
                        loss = loss / self.ft_update_freq
                    scaler.scale(loss).backward()
                    if bidx % self.ft_update_freq == self.ft_update_freq - 1 or bidx == len(
                        train_dataset) - 1:
                        scaler.step(optim)
                        scaler.update()
                        optim.zero_grad()

                if epoch % self.ft_valid_freq == (self.ft_valid_freq - 1):
                    test_loss = calculate_ce_loss(model, valid_dataset)
                    if test_loss < best_loss:
                        logger.info(
                            f"epoch {epoch} new loss {test_loss} old loss {best_loss} BETTER"
                        )
                        best_loss = test_loss
                        best_sd = {k: v.cpu() for k, v in model.state_dict().items()}
                        # best_sd = copy.deepcopy(model.state_dict())
                        worse_ct = 0
                    else:
                        worse_ct += 1
                        if worse_ct >= self.ft_early_stop:
                            break

            model.eval()
            with torch.no_grad():
                model.load_state_dict(best_sd)
            model = model.to(origin_dtype)
            torch.set_grad_enabled(False)

        model.is_quantized = True
        if has_config:
            model.config.use_cache = use_cache
            model.config.quantization_config = self.to_dict()

        torch.cuda.empty_cache()
        return model


    def save(self,
             model: nn.Module,
             save_dir: str,
             max_shard_size: str = "10GB",
             safe_serialization: bool = False):
        """
        Save model state dict and configs

        Args:
            model (`nn.Module`):
                Model to be saved. The model can be wrapped or unwraped.
            save_dir (`str`):
                Directory to which to save. Will be created if it doesn't exist.
            max_shard_size (`str`, defaults to `"10GB"`):
                The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size
                lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5MB"`).
                <Tip warning={true}>

                If a single weight of the model is bigger than `max_shard_size`, it will be in its own checkpoint shard
                which will be bigger than `max_shard_size`.

                </Tip>
            safe_serialization (`bool`, defaults to `False`):
                Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).

        """

        os.makedirs(save_dir, exist_ok=True)
        # save model and config
        accelerator = Accelerator()
        accelerator.save_model(model,
                               save_dir,
                               max_shard_size=max_shard_size,
                               safe_serialization=safe_serialization)
        if hasattr(model, "config"):
            model.config.save_pretrained(save_dir)
        with open(os.path.join(save_dir, QUIP_CONFIG), "w",
                  encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


# Copied from https://github.com/casper-hansen/AutoAWQ/blob/main/awq/models/base.py
def load_config(model_path, safetensors=True, trust_remote_code=True, revision=None):
    # [STEP 1] Download model if path is not a directory
    if not os.path.isdir(model_path):
        ignore_patterns = ["*msgpack*", "*h5*", "optimizer.pt"]
        if safetensors:
            ignore_patterns.extend(["*.pt*", "*.bin*", "consolidated*"])
        else:
            ignore_patterns.append("*.safetensors*")

        model_path = snapshot_download(model_path, ignore_patterns=ignore_patterns, revision=revision)

    model_weights_path = model_path

    # [STEP 2] Load config and set sequence length
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code, revision=revision)

    return model_weights_path, config


def load_quantized_model(
    save_folder: str,
    revision: Optional[str] = None,
    torch_dtype: Optional[Union[str, torch.dtype]] = torch.float16,
    trust_remote_code: bool = True,
    use_safetensors: bool = False,
    device_map: Optional[str] = None,
):
    """
    Load quantized weights from the save_folder into the converted model and dispatch the weights according to the device_map.

    Args:
        save_folder (`str`):
            Directory to which to load the weights.
        quant_config_name (`str`, defaults to `QUIP_CONFIG`):
            Name of the quantization config file

    Returns:
        `nn.Module`: The quantized model
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU found. A GPU is needed to run quantized model.")

    model_weights_path, config = load_config(save_folder,
                                             trust_remote_code=trust_remote_code,
                                             safetensors=use_safetensors,
                                             revision=revision)
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype)

    if hasattr(config, "quantization_config"):
        quantize_config_dict = config.quantization_config
    else:
        with open(os.path.join(model_weights_path, QUIP_CONFIG)) as f:
            quantize_config_dict = json.load(f)
    quantize_config_dict["inference"] = True
    quantize_config_dict["ft_epochs"] = 0
    quantizer = QuipQuantizer.from_dict(quantize_config_dict)
    quantizer.codebook = quantizer.codebook.to(torch_dtype)

    model = quantizer.convert_model(model)

    if device_map is None:
        device_map = {"": "cpu"}
    load_checkpoint_and_dispatch(
        model,
        checkpoint=model_weights_path,
        device_map=device_map,
        no_split_module_classes=quantizer.get_no_split_module_classes(model),
        dtype=torch_dtype,
    )

    # Trick for better performance
    for layer in get_layers(model, [QuantLinear]).values():
        layer.wscale_float = layer.Wscale.mean().float().item()
        if layer.per_channel:
            layer.Wscale = layer.Wscale / layer.Wscale.mean()
        if quantizer.merge_suv:
            if torch.all(layer.SU > 0):
                layer.SU = None
            if torch.all(layer.SV > 0):
                layer.SV = None

    model.is_quantized = True
    model.eval()
    return model
