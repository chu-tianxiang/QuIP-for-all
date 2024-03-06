# https://github.com/Cornell-RelaxML/quip-sharp/blob/main/lib/codebook/latticee8_padded12_rvq4bit.py
"""
E8 4 bit.
2 2 bit E8P codebooks with RVQ.
"""
import torch
from torch import nn

from .e8p12 import get_full_grid, get_packed_abs_grid, _E8P_CODESZ


class E8P12RVQ4B_codebook(nn.Module):

    def __init__(self, inference=False, opt_resid_scale=None, **kwargs):
        super(E8P12RVQ4B_codebook, self).__init__()
        self.id = "E8P12RVQ4B"
        self.opt_scale = 1.03
        self.codesz = _E8P_CODESZ
        self.idx_dtype = torch.int32
        self.packsz = 1
        self.pack_out = False
        self.version = 0
        self.opt_resid_scale = 1 / 3.45 if opt_resid_scale is None else opt_resid_scale

        self.register_buffer("grid_packed_abs", get_packed_abs_grid(), persistent=False)

        if not inference:
            _E8P_GRID, _ = get_full_grid(self.grid_packed_abs)
            self.register_buffer("grid", _E8P_GRID, persistent=False)
            self.register_buffer("grid_norm", self.grid.norm(dim=-1)**2, persistent=False)

    def round(self, X, grid, grid_norm):
        assert X.shape[-1] == self.codesz
        Xqidx = (2 * X @ grid.T - grid_norm).argmax(-1)
        return grid[Xqidx], Xqidx

    def quantize(self, X, return_idx=True):
        init_vals, init_idxs = self.round(X, self.grid, self.grid_norm)
        resid = (X - init_vals) / self.opt_resid_scale
        resid_vals, resid_idxs = self.round(resid, self.grid, self.grid_norm)
        final_vals = init_vals + resid_vals * self.opt_resid_scale
        final_idxs = (init_idxs << 16) + resid_idxs
        if return_idx:
            return final_vals, final_idxs
        return final_vals

    def maybe_pack_idxs(self, idxs):
        return idxs

    def decompress_weight(self, Qidxs):
        return torch.ops.quip_lib.decompress_e8prvq4_origorder(
            Qidxs, self.grid_packed_abs, self.opt_resid_scale
        )

    def forward(self,
                input,
                Qidxs):
        if input.size(0) < 32:
            output = torch.ops.quip_lib.e8prvq4_mm_origorder(
                input,
                Qidxs,
                self.grid_packed_abs,
                self.opt_resid_scale,
            )
        else:
            W_decompressed = self.decompress_weight(Qidxs)
            output = input @ W_decompressed.T
        return output