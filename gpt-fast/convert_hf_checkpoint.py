# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import sys
from pathlib import Path
from typing import Optional

import torch
import re
from safetensors.torch import load_file

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from model import ModelArgs

@torch.inference_mode()
def convert_hf_checkpoint(
    *,
    checkpoint_dir: Path = Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf"),
    model_name: Optional[str] = None,
) -> None:
    if model_name is None:
        model_name = checkpoint_dir.name

    config = ModelArgs.from_name(model_name)
    print(f"Model config {config.__dict__}")

    # Load the json file containing weight mapping
    pt_model_map_json = checkpoint_dir / "pytorch_model.bin.index.json"
    st_model_map_json = checkpoint_dir / "model.safetensors.index.json"
    pt_model = checkpoint_dir / "pytorch_model.bin"
    st_model = checkpoint_dir / "model.safetensors"
    if pt_model_map_json.is_file() or st_model_map_json.is_file():
        model_map_json = pt_model_map_json if pt_model_map_json.is_file() else st_model_map_json
        with open(model_map_json) as json_map:
            bin_index = json.load(json_map)
        bin_files = {checkpoint_dir / bin for bin in bin_index["weight_map"].values()}
    elif pt_model.is_file():
        bin_files = {pt_model}
    elif st_model.is_file():
        bin_files = {st_model}
    else:
        print("cannot find model weights")

    weight_map = {
        # llama
        "model.embed_tokens": "tok_embeddings",
        "model.layers.{}.self_attn.q_proj": "layers.{}.attention.wq",
        "model.layers.{}.self_attn.k_proj": "layers.{}.attention.wk",
        "model.layers.{}.self_attn.v_proj": "layers.{}.attention.wv",
        "model.layers.{}.self_attn.o_proj": "layers.{}.attention.wo",
        'model.layers.{}.self_attn.rotary_emb': None,
        'model.layers.{}.mlp.gate_proj': 'layers.{}.feed_forward.w1',
        "model.layers.{}.mlp.up_proj": "layers.{}.feed_forward.w3",
        "model.layers.{}.mlp.down_proj": "layers.{}.feed_forward.w2",
        "model.layers.{}.input_layernorm": "layers.{}.attention_norm",
        "model.layers.{}.post_attention_layernorm": "layers.{}.ffn_norm",
        "model.norm": "norm",
        "lm_head": "output",
        # qwen
        "transformer.wte": "tok_embeddings",
        "transformer.h.{}.attn.c_attn": "layers.{}.attention.wqkv",
        "transformer.h.{}.attn.c_proj": "layers.{}.attention.wo",
        'transformer.h.{}.attn.rotary_emb': None,
        'transformer.h.{}.mlp.w1': 'layers.{}.feed_forward.w3',
        "transformer.h.{}.mlp.w2": "layers.{}.feed_forward.w1",
        "transformer.h.{}.mlp.c_proj": "layers.{}.feed_forward.w2",
        "transformer.h.{}.ln_1": "layers.{}.attention_norm",
        "transformer.h.{}.ln_2": "layers.{}.ffn_norm",
        "transformer.ln_f": "norm",
    }

    merged_result = {}
    for file in sorted(bin_files):
        if str(file).endswith(".bin"):
            state_dict = torch.load(str(file), map_location="cpu", mmap=True, weights_only=True)
        else:
            state_dict = load_file(str(file), device="cpu")
        merged_result.update(state_dict)

    final_result = {}
    for key, value in merged_result.items():
        if "layers" in key or ".h." in key:
            abstract_key = re.sub(r'\.(\d+)', '.{}', key)
            abstract_layer, suffix = abstract_key.rsplit(".", 1)
            layer_num = re.search(r'\d+', key).group(0)
            new_key = weight_map[abstract_layer]
            if new_key is None:
                continue
            new_key = new_key.format(layer_num) + f".{suffix}"
        else:
            layer, suffix = key.rsplit(".", 1)
            new_key = weight_map[layer] + f".{suffix}"

        final_result[new_key] = value

    hf_config = checkpoint_dir / "config.json"
    assert hf_config.is_file()
    hf_config = json.load(open(hf_config))
    if "quantization_config" in hf_config and hf_config["quantization_config"]["quant_method"].lower() == "quip":
        conf = hf_config["quantization_config"]
        if conf["codebook"] == "E8P12":
            bits = 2
        elif conf["codebook"] == "E8P12RVQ3B":
            bits = 3
        else:
            bits = 4
        if conf["use_rand"]:
            model_name = checkpoint_dir / f"model_int{bits}_rand.pth"
        else:
            model_name = checkpoint_dir / f"model_int{bits}.pth"
    else:
        model_name = checkpoint_dir / "model.pth"
    print(f"Saving checkpoint to {model_name}")
    torch.save(final_result, model_name)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert HuggingFace checkpoint.')
    parser.add_argument('--checkpoint_dir', type=Path, default=Path("checkpoints/meta-llama/llama-2-7b-chat-hf"))
    parser.add_argument('--model_name', type=str, default=None)

    args = parser.parse_args()
    convert_hf_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        model_name=args.model_name,
    )
