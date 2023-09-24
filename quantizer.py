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
import copy
import json
import os
from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers.pytorch_utils import Conv1D
from accelerate import (
    Accelerator,
    cpu_offload_with_hook,
    load_checkpoint_and_dispatch,
)
from accelerate.hooks import remove_hook_from_module

from constants import QUIP_CONFIG
from data import get_dataset, prepare_dataset
from utils import (
    get_block_name_with_pattern,
    get_device,
    get_layers,
    get_preceding_modules,
    get_seqlen,
    recurse_getattr
)
from quip import Balance
from qlinear import QuantLinear

logger = getLogger(__name__)


class QuipQuantizer(object):
    r"""
    A simple API for QUIP Quantization
    """

    def __init__(
        self,
        bits: int,
        dataset: Optional[Union[List[str], str]] = None,
        nsamples: int = 1024,
        model_seqlen: int = 2048,
        quant: str = 'ldlq',
        npasses: int = 0,
        per_channel: bool = True,
        damp_percent: float = 0.01,
        true_sequential: bool = True,
        block_name_to_quantize: Optional[str] = None,
        module_name_preceding_first_block: Optional[List[str]] = None,
        batch_size: int = 1,
        pad_token_id: Optional[int] = None,
        *args,
        **kwargs,
    ):
        self.bits = bits
        self.dataset = dataset
        self.nsamples = nsamples
        self.quant = quant
        self.npasses = npasses
        self.per_channel = per_channel
        self.damp_percent = damp_percent
        self.model_seqlen = model_seqlen
        self.true_sequential = true_sequential
        self.block_name_to_quantize = block_name_to_quantize
        self.module_name_preceding_first_block = module_name_preceding_first_block
        self.batch_size = batch_size
        self.pad_token_id = pad_token_id
        self.quant_method = 'QUiP'

        if self.bits not in [2, 3, 4, 8]:
            raise ValueError("only support quantize to [2,3,4,8] bits.")
        if not (0 < self.damp_percent < 1):
            raise ValueError("damp_percent must between 0 and 1.")

    def to_dict(self):
        """
        Returns the args in dict format.
        """
        return copy.deepcopy(self.__dict__)

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
        layers_to_be_replaced = get_layers(model, prefix=block_name)
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
                new_layer = QuantLinear(self.bits, in_features, out_features)
                new_layer.device = device
                setattr(module, attr, new_layer.to(device))
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

        if hasattr(model, "dtype"):
            self.use_cuda_fp16 = model.dtype == torch.float16

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
            layer_inputs.append(input)
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
                # put the data on gpu, we won't put them back to cpu
                data[k] = v.to(0)
            try:
                model(**data)
            except ValueError:
                pass

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
            layers = get_layers(block)
            if self.true_sequential:
                # lazy sequential but works well
                layers_name_list = [[key] for key in layers.keys()]
            else:
                layers_name_list = [list(layers.keys())]
            logger.info(f"Module to quantize {layers_name_list}")
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
                    quant_method[name] = Balance(subset_layers[name],
                                                 self.quant, self.bits,
                                                 self.npasses)
                    quant_method[name].quantizer.configure(
                        bits=self.bits,
                        sym=False,
                        perchannel=self.per_channel,
                        qfn="b")

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
                    # the args are already on the gpu
                    # don't need to store the output
                    #print(j, layer_inputs[j].dtype)
                    block(layer_inputs[j], **layer_input_kwargs[j])
                # remove hook
                for h in handles:
                    h.remove()
                for name in subset_name_list:
                    logger.info(
                        f"Quantizing {name} in block {i + 1}/{len(blocks)}...")
                    quant_method[name].preproc(preproc_gptqH=True,
                                               percdamp=True,
                                               preproc_rescale=True,
                                               preproc_proj=True)
                    qweight, scale, scaleWH, subU, subV = quant_method[
                        name].fasterquant()
                    quantizers[f"{self.block_name_to_quantize}.{i}.{name}"] = (
                        quant_method[name].quantizer.to("cpu"),
                        qweight.to('cpu'), scale.to("cpu"), scaleWH.to("cpu"),
                        subU, subV)
                    quant_method[name].free()
                del subset_layers
            # we get the new output from the partial quantized block
            for j in range(len(dataset)):
                layer_output = block(layer_inputs[j],
                                     **layer_input_kwargs[j])[0]
                layer_outputs.append(layer_output)

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
        layers = get_layers(model)
        layers = {n: layers[n] for n in quantizers}
        self._replace_by_quant_layers(model, quantizers)
        qlayers = get_layers(model, [QuantLinear])
        for name in qlayers:
            logger.info(name)
            quantizers[
                name], qweight, scale, scale_wh, sub_u, sub_v = quantizers[
                    name]
            # so far can only pack layer on CPU
            layer_device = qlayers[name].device
            qlayers[name].to("cpu")
            layers[name], scale, scale_wh = layers[name].to('cpu'), scale.to(
                'cpu'), scale_wh.to('cpu')
            qlayers[name].pack(layers[name], qweight, scale, scale_wh, sub_u,
                               sub_v)
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
        with open(os.path.join(save_dir, QUIP_CONFIG), "w",
                  encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def load_quantized_model(
    model: nn.Module,
    save_folder: str,
    quant_config_name: str = QUIP_CONFIG,
    state_dict_name: Optional[str] = None,
    device_map: Optional[str] = None,
    max_memory: Optional[Dict] = None,
    no_split_module_classes: Optional[Dict] = None,
    offload_folder: Optional[str] = None,
    offload_buffers: Optional[str] = None,
    offload_state_dict: bool = False,
    max_input_length: Optional[int] = None,
):
    """
    Load quantized weights from the save_folder into the converted model and dispatch the weights according to the device_map.

    Args:
        model (`nn.Module`):
            The model can be enpty or not.
        save_folder (`str`):
            Directory to which to load the weights.
        quant_config_name (`str`, defaults to `QUIP_CONFIG`):
            Name of the quantization config file
        state_dict_name (`Optional[str]`, defaults to `None`):
            Name of the state dict file
        device_map (`Optional[str]`, defaults to `None`):
            A map that specifies where each submodule should go. It doesn't need to be refined to each parameter/buffer
            name, once a given module name is inside, every submodule of it will be sent to the same device.
            To have Accelerate compute the most optimized `device_map` automatically, set `device_map="auto"`.
        max_memory (`Optional[Dict]`, defaults to `None`):
            A dictionary device identifier to maximum memory. Will default to the maximum memory available for each GPU
            and the available CPU RAM if unset.
        no_split_module_classes (`Optional[Dict]`, defaults to `None`):
            A list of layer class names that should never be split across device (for instance any layer that has a
            residual connection).
        offload_folder (`Optional[str]`, defaults to `None`):
            If the `device_map` contains any value `"disk"`, the folder where we will offload weights.
        offload_buffers (`Optional[str]`, defaults to `None`):
            In the layers that are offloaded on the CPU or the hard drive, whether or not to offload the buffers as
            well as the parameters.
        offload_state_dict (`bool`, defaults to `False`):
            If `True`, will temporarily offload the CPU state dict on the hard drive to avoid getting out of CPU RAM if
            the weight of the CPU state dict + the biggest shard does not fit. Will default to `True` if the device map
            picked contains `"disk"` values.
        max_input_length (`Optional[int]`, defaults to `None`):
            The maximum input length. This is needed to initialize a buffer that depends on the maximum expected input length.
            It is specific to the exllama backend with act-order.

    Returns:
        `nn.Module`: The quantized model
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU found. A GPU is needed to run quantized model.")
    if device_map is None:
        device_map = {"": torch.cuda.current_device()}
        logger.info(
            "The device_map was not initialized."
            "Setting device_map to `{'':torch.cuda.current_device()}`.")

    with open(os.path.join(save_folder, quant_config_name),
              "r",
              encoding="utf-8") as f:
        quantize_config_dict = json.load(f)
    quantizer = QuipQuantizer.from_dict(quantize_config_dict)
    quantizer.max_input_length = max_input_length

    model = quantizer.convert_model(model)

    if no_split_module_classes is None:
        no_split_module_classes = quantizer.get_no_split_module_classes(model)

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.path.join(save_folder, state_dict_name)
        if state_dict_name is not None else save_folder,
        device_map=device_map,
        max_memory=max_memory,
        no_split_module_classes=no_split_module_classes,
        offload_folder=offload_folder,
        offload_buffers=offload_buffers,
        offload_state_dict=offload_state_dict,
    )

    model.is_quantized = True
    model.eval()
    return model
