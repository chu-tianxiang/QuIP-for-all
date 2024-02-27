# https://github.com/Cornell-RelaxML/quip-sharp/blob/main/lib/codebook/latticee8_padded12_rvq3bit.py
"""
E8 3 bit.
Made from 2 bit E8P + 1 bit E8 with RVQ.
"""
from fractions import Fraction

import numpy as np
import torch
from torch import nn

import quiptools_cuda
from .e8p12 import get_full_grid, get_packed_abs_grid, _E8P_CODESZ

def get_e81bgrid():
    intr = torch.arange(-4, 4)
    hintr = intr + 1 / 2

    gintr = torch.cartesian_prod(*[intr] * 8)
    ghintr = torch.cartesian_prod(*[hintr] * 8)

    ge8 = torch.concat([gintr, ghintr], dim=0)
    ge8m2 = (ge8.sum(dim=-1) % 2 == 0)
    ge8n = ge8.norm(dim=-1)**2 <= 2

    e8 = ge8[torch.where(ge8m2 * ge8n)[0]]

    norm4 = torch.tensor([
        [2, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 2],
        [-2, 0, 0, 0, 0, 0, 0, 0],
        [0, -2, 0, 0, 0, 0, 0, 0],
        [0, 0, -2, 0, 0, 0, 0, 0],
        [0, 0, 0, -2, 0, 0, 0, 0],
        [0, 0, 0, 0, -2, 0, 0, 0],
        [0, 0, 0, 0, 0, -2, 0, 0],
        [0, 0, 0, 0, 0, 0, -2, 0],
        #[0, 0, 0, 0, 0, 0, 0, -2],
    ])

    e8 = torch.concat([e8, norm4], dim=0)
    return e8.to(torch.float16)

def pack_e81b(cba):
    cba = cba[:, [0, 2, 4, 6, 1, 3, 5, 7]]
    cba = cba * 2
    cba = cba.to(torch.int32)
    cba = cba & 0xf
    acc = cba[:,0]
    for i in range(7):
        acc = acc | (cba[:,(i+1)] << ((i+1)*4))
    return acc

class E8P12RVQ3B_codebook(nn.Module):

    def __init__(self, inference=False, opt_resid_scale=None, **kwargs):
        super(E8P12RVQ3B_codebook, self).__init__()
        self.id = "E8P12RVQ3B"
        self.opt_scale = 0.98
        self.codesz = _E8P_CODESZ
        self.idx_dtype = torch.int32
        self.packsz = Fraction(4, 3)
        self.pack_out = False
        self.version = 0
        self.opt_resid_scale = 1 / 2.04 if opt_resid_scale is None else opt_resid_scale

        self.register_buffer("grid_packed_abs", get_packed_abs_grid(), persistent=False)
        self.register_buffer("e81b_grid", get_e81bgrid(), persistent=False)
        self.register_buffer("e81b_grid_packed", pack_e81b(self.e81b_grid), persistent=False)

        if not inference:
            _E8P_GRID, _ = get_full_grid(self.grid_packed_abs)
            self.register_buffer("grid", _E8P_GRID, persistent=False)
            self.register_buffer("grid_norm", self.grid.norm(dim=-1)**2, persistent=False)
            self.register_buffer("e81b_grid_norm", self.e81b_grid.norm(dim=-1)**2, persistent=False)

    def round(self, X, grid, grid_norm):
        assert X.shape[-1] == self.codesz
        Xqidx = (2 * X @ grid.T - grid_norm).argmax(-1)
        return grid[Xqidx], Xqidx

    def quantize(self, X, return_idx=True):
        init_vals, init_idxs = self.round(X, self.grid, self.grid_norm)
        resid = (X - init_vals) / self.opt_resid_scale
        resid_vals, resid_idxs = self.round(resid, self.e81b_grid, self.e81b_grid_norm)
        final_vals = init_vals + resid_vals * self.opt_resid_scale
        final_idxs = (init_idxs << 8) + resid_idxs
        if return_idx:
            return final_vals, final_idxs
        return final_vals

    def maybe_pack_idxs(self, idxs):
        # remove the first 8 bit assuming little end
        # Todo: better packing for memory access
        idxs_int8 = idxs.view(dtype=torch.int8).view(idxs.shape[0], idxs.shape[1], -1)
        idxs = idxs_int8[..., :3].reshape(idxs.shape[0], -1).view(torch.int32)
        return idxs

    def decompress_weight(self, Qidxs):
        W_decompressed = torch.empty(
            Qidxs.shape[0], Qidxs.shape[1] * 32 // 3,
            dtype=torch.float16, device=Qidxs.device
        )
        quiptools_cuda.decompress_e8prvq3_origorder(
            Qidxs, self.grid_packed_abs, self.e81b_grid_packed,
            W_decompressed, self.opt_resid_scale
        )
        return W_decompressed

    def forward(self,
                input,
                Qidxs):
        if input.size(0) < 32:
            output = quiptools_cuda.e8prvq3_mm_origorder(
                input,
                Qidxs,
                self.grid_packed_abs,
                self.e81b_grid_packed,
                self.opt_resid_scale
            )
        else:
            W_decompressed = self.decompress_weight(Qidxs)
            output = input @ W_decompressed.T
        return output