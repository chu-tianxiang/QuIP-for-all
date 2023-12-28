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
import json
import os
from pathlib import Path
from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers.pytorch_utils import Conv1D
from accelerate import (
    Accelerator,
    cpu_offload_with_hook,
    init_empty_weights,
)
from accelerate.hooks import remove_hook_from_module
from safetensors.torch import load_file

from constants import QUIP_CONFIG
from data import get_dataset, prepare_dataset
from utils import (get_block_name_with_pattern, get_device, get_layers,
                   get_preceding_modules, get_seqlen, recurse_getattr)
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
        dataset: Optional[Union[List[str], str]] = None,
        nsamples: int = 4096,
        model_seqlen: int = 2048,
        quip_tune_iters: int = 10,
        sigma_reg: float = 0.01,
        rescale_WH: bool = False,
        use_rand: bool = True,
        scale_override: float = -1,
        sequential: bool = False,
        block_name_to_quantize: Optional[str] = None,
        module_name_preceding_first_block: Optional[List[str]] = None,
        batch_size: int = 1,
        pad_token_id: Optional[int] = None,
        inference: bool = False,
        cache_on_gpu: bool = False,
        modules_to_not_convert: Optional[List] = None,
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
        self.sequential = sequential
        self.block_name_to_quantize = block_name_to_quantize
        self.module_name_preceding_first_block = module_name_preceding_first_block
        self.batch_size = batch_size
        self.pad_token_id = pad_token_id
        self.cache_on_gpu = cache_on_gpu
        self.modules_to_not_convert = modules_to_not_convert
        self.quant_method = 'QUiP'

        if codebook not in ["D4", "E8P12", "HI"]:
            raise ValueError("Invalid codebook, has to be D4 or E8P12 or HI")
        self.codebook = codebook_id[codebook](inference=inference)

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

    def convert_model(self, model: nn.Module):
        """
        Convert the model to a Quip model by getting and replacing the layers.

        Args:
            model (`nn.Module`):
                Model to be converted

        """
        if self.block_name_to_quantize is None:
            self.block_name_to_quantize = get_block_name_with_pattern(model)
        block_name = self.block_name_to_quantize
        layers_to_be_replaced = get_layers(model, prefix=block_name, skip=self.modules_to_not_convert)
        self._replace_by_quant_layers(model, layers_to_be_replaced)

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
                                 name: str = ""):
        """
        Replaces linear layers in `module` by `QuantLinear`

        Args:
            module (`nn.Module`):
                Module to quantize
            names (`List[str]`):
                List of names of the module to quantize
            name (`str`, defaults to `""`):
                To keep track of the name of the current module
        """
        if isinstance(module, QuantLinear):
            return
        for attr in dir(module):
            layer = getattr(module, attr)
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
                new_layer = QuantLinear(in_features,
                                        out_features,
                                        self.codebook,
                                        bias=(layer.bias is not None),
                                        use_rand=self.use_rand,
                                        weight_dtype=layer.weight.dtype,)
                new_layer.device = device
                #setattr(module, attr, new_layer.to(device))
                setattr(module, attr, new_layer)
        for name1, child in module.named_children():
            self._replace_by_quant_layers(
                child, names, name + "." + name1 if name != "" else name1)

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
        if self.dataset is None:
            raise ValueError(
                "You need to pass `dataset` in order to quantize your model")
        elif isinstance(self.dataset, str):
            dataset = get_dataset(self.dataset,
                                  tokenizer,
                                  nsamples=self.nsamples,
                                  seqlen=self.model_seqlen,
                                  split="train")
        elif isinstance(self.dataset, list):
            dataset = [
                tokenizer(data, return_tensors="pt") for data in self.dataset
            ]
        else:
            raise ValueError(
                "You need to pass a list of string or a string for `dataset`")

        dataset = prepare_dataset(dataset,
                                  pad_token_id=self.pad_token_id,
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

        if not has_device_map:
            # put modules from module_name_preceding_first_block on cuda
            for module_name in self.module_name_preceding_first_block:
                module = recurse_getattr(model, module_name)
                if module is None:
                    raise ValueError(
                        f"Module {module_name} was not found in model")
                module = module.to(0)
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
                input.to('cpu') if not self.cache_on_gpu else input)
            other_kwargs = {}
            for k, v in kwargs.items(
            ):  # make sure other arguments also be captured
                if k not in ["hidden_states"]:
                    other_kwargs[k] = v
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
            blocks[0].to(device)
            for module_name in self.module_name_preceding_first_block:
                module = recurse_getattr(model, module_name)
                if module is None:
                    raise ValueError(
                        f"Module {module_name} was not found in model")

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
            layers = get_layers(block, skip=self.modules_to_not_convert)
            layers_name_list = [list(layers.keys())]
            logger.info(f"Module to quantize {layers_name_list}")
            print(f"Module to quantize {layers_name_list}")
            for subset_name_list in tqdm(
                    layers_name_list,
                    leave=False,
                    desc="Quantizing layers inside the block"):
                subset_layers = {
                    name: layers[name]
                    for name in subset_name_list
                }
                quant_method = {}
                handles = []
                # add hook for each layer in subset_layers
                for name in subset_layers:
                    quant_method[name] = QUIP(subset_layers[name],
                                              self.codebook)

                    def add_batch(name):

                        def tmp(_, input, output):
                            quant_method[name].add_batch(
                                input[0].data, output.data)

                        return tmp

                    # because it adding a hook will replace the old one.
                    handles.append(subset_layers[name].register_forward_hook(
                        add_batch(name)))
                # update Hessian for each layer in subset_layers thanks to the hook
                for j in range(len(dataset)):
                    layer_input = layer_inputs[j].to(get_device(
                        block)) if not self.cache_on_gpu else layer_inputs[j]
                    layer_output = block(layer_input,
                                         **layer_input_kwargs[j])[0]
                    layer_outputs.append(layer_output.cpu(
                    ) if not self.cache_on_gpu else layer_output)
                # remove hook
                for h in handles:
                    h.remove()
                for name in subset_name_list:
                    logger.info(
                        f"Quantizing {name} in block {i + 1}/{len(blocks)}...")
                    old_weight = quant_method[name].layer.weight.data.clone()
                    attr = quant_method[name].quant(
                        rescale_WH=self.rescale_WH,
                        sigma_reg=self.sigma_reg,
                        quip_tune_iters=self.quip_tune_iters,
                        scale_override=self.scale_override,
                        use_rand=self.use_rand)
                    logger.info("mse: ", (quant_method[name].layer.weight.data - old_weight).pow(2).mean().sqrt())
                    quantizers[
                        f"{self.block_name_to_quantize}.{i}.{name}"] = attr
                    quant_method[name].free()
                del subset_layers
            # we get the new output from the partial quantized block
            if self.sequential:
                layer_outputs = []
                for j in range(len(dataset)):
                    layer_input = layer_inputs[j].to(get_device(
                        block)) if not self.cache_on_gpu else layer_inputs[j]
                    layer_output = block(layer_input,
                                         **layer_input_kwargs[j])[0]
                    layer_outputs.append(layer_output.cpu(
                    ) if not self.cache_on_gpu else layer_output)

            # put back to device
            if not has_device_map:
                blocks[i] = block.to(device)
            del layers
            del layer_inputs
            layer_inputs, layer_outputs = layer_outputs, []
            torch.cuda.empty_cache()

        # Step 4: Pack the model at the end (Replacing the layers)
        self.pack_model(model=model, quantizers=quantizers)

        model.is_quantized = True
        if has_config:
            model.config.use_cache = use_cache
            model.config.quantization_config = self.to_dict()

        torch.cuda.empty_cache()
        return model

    def pack_model(
        self,
        model: nn.Module,
        quantizers: Dict[str, Tuple],
    ):
        """
        Pack the model by replacing the layers by quantized layers

        Args:
            model (`nn.Module`):
                The model to pack
            quantizers (`Dict[str,Tuple]`):
                A mapping of the layer name and the data needed to pack the layer
        """
        logger.info("Packing model...")
        layers = get_layers(model, skip=self.modules_to_not_convert)
        layers = {n: layers[n].to('cpu') for n in quantizers}
        self._replace_by_quant_layers(model, quantizers)
        qlayers = get_layers(model, [QuantLinear])
        for name in qlayers:
            logger.info(name)
            attr = quantizers[name]
            layer_device = qlayers[name].device
            qlayers[name] = qlayers[name].to("cpu")
            qlayers[name].pack(layers[name], attr)
            qlayers[name].to(layer_device)

        logger.info("Model packed.")

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


def load_quantized_model(
    save_folder: str,
    revision: Optional[str] = None,
    torch_dtype: Optional[Union[str, torch.dtype]] = torch.float16,
    trust_remote_code: bool = True,
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

    config = AutoConfig.from_pretrained(save_folder,
                                        trust_remote_code=trust_remote_code,
                                        revision=revision)
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype)

    if hasattr(config, "quantization_config"):
        quantize_config_dict = config.quantization_config
    else:
        with open(os.path.join(save_folder, QUIP_CONFIG)) as f:
            quantize_config_dict = json.load(f)
    quantize_config_dict["inference"] = True
    quantizer = QuipQuantizer.from_dict(quantize_config_dict)
    quantizer.codebook = quantizer.codebook.to(torch_dtype)

    model = quantizer.convert_model(model)
    model.codebook = quantizer.codebook

    # move model to cpu
    model = model._apply(lambda t: torch.zeros_like(t, device="cpu")
                         if t.device == torch.device("meta") else t)

    checkpoint_dir = Path(save_folder)
    pt_model_map_json = checkpoint_dir / "pytorch_model.bin.index.json"
    st_model_map_json = checkpoint_dir / "model.safetensors.index.json"
    pt_model = checkpoint_dir / "pytorch_model.bin"
    st_model = checkpoint_dir / "model.safetensors"
    if pt_model_map_json.is_file() or st_model_map_json.is_file():
        model_map_json = pt_model_map_json if pt_model_map_json.is_file(
        ) else st_model_map_json
        with open(model_map_json) as json_map:
            bin_index = json.load(json_map)
        bin_files = {
            checkpoint_dir / bin
            for bin in bin_index["weight_map"].values()
        }
    elif pt_model.is_file():
        bin_files = {pt_model}
    elif st_model.is_file():
        bin_files = {st_model}
    else:
        return None

    for file in sorted(bin_files):
        if str(file).endswith(".bin"):
            state_dict = torch.load(str(file),
                                    map_location="cpu",
                                    mmap=True,
                                    weights_only=True)
        else:
            state_dict = load_file(str(file), device="cpu")
        model.load_state_dict(state_dict, assign=False, strict=False)

    model.is_quantized = True
    model.eval()
    return model
