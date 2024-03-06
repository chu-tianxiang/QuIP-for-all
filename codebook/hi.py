# https://github.com/Cornell-RelaxML/quip-sharp/blob/main/lib/codebook/half_integer_4bit_1col.py
from functools import cache

import torch
from torch import nn


@cache
def get_grid():
    hintr = torch.arange(-8, 8) + 1 / 2
    return hintr.unsqueeze(-1)


class HI4B1C_codebook(nn.Module):

    def __init__(self, inference=False, **kwargs):
        super(HI4B1C_codebook, self).__init__()
        self.id = "HI"
        self.opt_scale = 2.97
        self.codesz = 1
        self.idx_dtype = torch.int32
        self.packsz = 8
        self.pack_out = False
        self.version = 0

        if not inference:
            self.register_buffer("grid", get_grid(), persistent=False)
            self.register_buffer("grid_norm", torch.diag(self.grid @ self.grid.T), persistent=False)

    def round(self, X, grid, grid_norm):
        assert X.shape[-1] == self.codesz
        Xqidx = (2 * X @ grid.T - grid_norm).argmax(-1)
        return grid[Xqidx], Xqidx

    def quantize(self, X, return_idx=True):
        vals, idx = self.round(X, self.grid, self.grid_norm)
        if not return_idx:
            return vals
        return vals, idx.to(self.idx_dtype)

    def maybe_pack_idxs(self, idxs):
        return \
            (idxs[:, 0::self.packsz] << 4*0) + \
            (idxs[:, 2::self.packsz] << 4*1) + \
            (idxs[:, 4::self.packsz] << 4*2) + \
            (idxs[:, 6::self.packsz] << 4*3) + \
            (idxs[:, 1::self.packsz] << 4*4) + \
            (idxs[:, 3::self.packsz] << 4*5) + \
            (idxs[:, 5::self.packsz] << 4*6) + \
            (idxs[:, 7::self.packsz] << 4*7)

    def decompress_weight(self, Qidxs):
        return torch.ops.quip_lib.decompress_hi_origorder(Qidxs)

    def forward(self,
                input,
                Qidxs):
        if input.shape[0] < 32:
            output = torch.ops.quip_lib.hi_mm_origorder(input, Qidxs)
        else:
            W_decompressed = self.decompress_weight(Qidxs)
            output = input @ W_decompressed.t()
        return output
