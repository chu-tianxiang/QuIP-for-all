# https://github.com/huggingface/optimum/blob/main/optimum/gptq/constants.py
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

SEQLEN_KEYS_TRANFORMERS = [
    "max_position_embeddings", "seq_length", "n_positions"
]
BLOCK_PATTERNS = [
    "transformer.h",
    "model.decoder.layers",
    "gpt_neox.layers",
    "model.layers",
]

QUIP_CONFIG = "quantization_config.json"

ATTN_QKV_PATTERNS = [
    "self_attention.query_key_value",
    "attention.query_key_value",
    "attn.c_attn",
    "attn.qkv_proj",
    "self_attn.W_pack",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.q_proj",
    "attn.k_proj",
    "attn.v_proj",
    "attn.q_proj"
]

ATTN_OUT_PATTENRS = [
    "self_attention.dense",
    "self_attn.out_proj",
    "self_attn.o_proj",
    "attn.c_proj",
    "attn.out_proj",
    "attention.dense",
]

FC1_PATTERN = [
    "mlp.dense_h_to_4h",
    "mlp.up_proj",
    "mlp.gate_proj",
    "mlp.c_fc",
    "mlp.fc_in",
    "fc1",
    "mlp.w1",
    "mlp.w2",
    # mixtral moe
    "block_sparse_moe.experts.0.w1",
    "block_sparse_moe.experts.1.w1",
    "block_sparse_moe.experts.2.w1",
    "block_sparse_moe.experts.3.w1",
    "block_sparse_moe.experts.4.w1",
    "block_sparse_moe.experts.5.w1",
    "block_sparse_moe.experts.6.w1",
    "block_sparse_moe.experts.7.w1",
    "block_sparse_moe.experts.0.w3",
    "block_sparse_moe.experts.1.w3",
    "block_sparse_moe.experts.2.w3",
    "block_sparse_moe.experts.3.w3",
    "block_sparse_moe.experts.4.w3",
    "block_sparse_moe.experts.5.w3",
    "block_sparse_moe.experts.6.w3",
    "block_sparse_moe.experts.7.w3",
]

FC2_PATTERN = [
    "mlp.dense_4h_to_h",
    "mlp.down_proj",
    "mlp.c_proj",
    "mlp.fc_out",
    "mlp.c_proj",
    "fc2",
    # mixtral moe
    "block_sparse_moe.experts.0.w2",
    "block_sparse_moe.experts.1.w2",
    "block_sparse_moe.experts.2.w2",
    "block_sparse_moe.experts.3.w2",
    "block_sparse_moe.experts.4.w2",
    "block_sparse_moe.experts.5.w2",
    "block_sparse_moe.experts.6.w2",
    "block_sparse_moe.experts.7.w2",
]
