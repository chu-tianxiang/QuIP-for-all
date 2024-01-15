# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn

import register_lib
from utils import (get_hadK, matmul_hadUt_cuda, matmul_hadU_cuda, CODEBOOK, E81B_CODEBOOK)

def replace_linear_weight_only_quip(module, codebook, use_rand):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and "out" not in name:
            setattr(module, name, WeightOnlyQuipLinear(child.in_features, child.out_features,
                                                       use_rand=use_rand, codebook=codebook,
                                                       bias=child.bias is not None))
        else:
            replace_linear_weight_only_quip(child, codebook, use_rand)

class WeightOnlyQuipQuantHandler:
    def __init__(self, mod, codebook, use_rand):
        self.mod = mod
        self.codebook = codebook
        self.use_rand = use_rand

    def convert_for_runtime(self):
        replace_linear_weight_only_quip(self.mod, self.codebook, self.use_rand)
        return self.mod


class WeightOnlyQuipLinear(torch.nn.Module):
    __constants__ = ['in_features', 'out_features']

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 use_rand=False, codebook="E8P", resid_scale=-1) -> None:
        super().__init__()
        self.codebook = codebook
        self.in_features = in_features
        self.out_features = out_features
        had_left, self.K_left, self.q_in_features = get_hadK(in_features, use_rand)
        had_right, self.K_right, self.q_out_features = get_hadK(out_features, use_rand)
        if had_left is not None:
            self.register_buffer('had_left',
                                 had_left.to(torch.float16),
                                 persistent=use_rand)
        else:
            self.had_left = None
        if had_right is not None:
            self.register_buffer('had_right',
                                 had_right.to(torch.float16),
                                 persistent=use_rand)
        else:
            self.had_right = None
        if codebook == "E8P12":
            self.register_buffer(
                "Qidxs",
                torch.zeros((self.q_out_features,
                            self.q_in_features // 8),
                            dtype=torch.int16))
        elif codebook == "E8P12RVQ3B":
            self.register_buffer(
                "Qidxs",
                torch.zeros((self.q_out_features,
                            self.q_in_features // 32 * 3),
                            dtype=torch.int32))
            self.opt_resid_scale = 1 / 2.04 if resid_scale < 0 else resid_scale
        else:
            self.register_buffer(
                "Qidxs",
                torch.zeros((self.q_out_features,
                            self.q_in_features // 8),
                            dtype=torch.int32))
            self.opt_resid_scale = 1 / 3.45 if resid_scale < 0 else resid_scale
        self.register_buffer('grid_packed_abs', CODEBOOK, persistent=False)
        if codebook == "E8P12RVQ3B":
            self.register_buffer('e81b_grid', E81B_CODEBOOK, persistent=False)
        self.register_buffer("SU", torch.ones(in_features, dtype=torch.float16))
        self.register_buffer("SV", torch.ones(out_features, dtype=torch.float16))
        self.register_buffer("Wscale", torch.ones((), dtype=torch.float))
        if bias:
            self.register_buffer(
                'bias', torch.zeros((out_features), dtype=torch.float16))
        else:
            self.bias = None
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "SU" not in state_dict:
            self.SU = None
        if prefix + "SV" not in state_dict:
            self.SV = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input.view(-1, input.shape[-1])
        x_dtype = x.dtype
        if self.SU is not None:
            x = x * self.SU
        x = matmul_hadUt_cuda(x, self.had_left, self.K_left,
                              self.q_in_features)
        x = x * self.Wscale
        if x_dtype != torch.float16:
            x = x.to(torch.float16)

        if self.codebook == "E8P12":
            out = torch.ops.quip_lib.e8p_mm_origorder(
                x, self.Qidxs, self.grid_packed_abs
            )
        elif self.codebook == "E8P12RVQ3B":
            out = torch.ops.quip_lib.e8prvq3_mm_origorder(
                x, self.Qidxs, self.grid_packed_abs, self.e81b_grid,
                self.opt_resid_scale,
            )
        else:
            out = torch.ops.quip_lib.e8prvq4_mm_origorder(
                x, self.Qidxs, self.grid_packed_abs,
                self.opt_resid_scale,
            )

        if x_dtype != torch.float16:
            out = out.to(dtype=x_dtype)

        out = matmul_hadU_cuda(out, self.had_right, self.K_right,
                               self.q_out_features)[..., :self.out_features]

        if self.SV is not None:
            out = out * self.SV
        out = out.view(*input.shape[:-1], out.shape[-1])
        out = out + self.bias if self.bias is not None else out
        return out
