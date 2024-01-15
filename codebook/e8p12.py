# https://github.com/Cornell-RelaxML/quip-sharp/blob/main/lib/codebook/latticee8_padded12.py
"""
D8^ = D8 + 1/2 intersected with ball of radius sqrt(10)
|D8^| has 227 entries
We then add 29 entries from the set of vectors with 5 3/2 and 3 1/2
The total codebook is all 2^7 flips of these 256 entries (2^15) +- 1/4
which makes 2^16 entries.
This corresponds to a subset of E8 + 1/4
"""
import numpy as np
import torch
from torch import nn

import quiptools_cuda


_E8P_CODESZ = 8
_INT_MAP = 2**(torch.arange(_E8P_CODESZ).flip(0))


def int2mask(i, int_map):
    return ((i & int_map) > 0).int()


def mask2int(mask, int_map):
    return (int_map.unsqueeze(0) * mask.int()).sum(dim=-1)


def get_norm12():
    # 29 elements of norm 12 in E8 + 1/4
    return torch.tensor([
        [3, 1, 1, 1, 3, 3, 3, 3],
        [1, 3, 1, 1, 3, 3, 3, 3],
        [1, 1, 3, 1, 3, 3, 3, 3],
        [1, 1, 1, 3, 3, 3, 3, 3],
        [3, 3, 3, 1, 3, 3, 1, 1],
        [3, 3, 3, 1, 3, 1, 3, 1],
        [3, 3, 3, 1, 1, 3, 3, 1],
        [3, 3, 3, 1, 3, 1, 1, 3],
        [3, 3, 3, 1, 1, 3, 1, 3],
        [3, 3, 3, 1, 1, 1, 3, 3],
        [3, 3, 1, 3, 3, 3, 1, 1],
        [3, 3, 1, 3, 3, 1, 3, 1],
        [3, 3, 1, 3, 1, 3, 3, 1],
        [3, 3, 1, 3, 3, 1, 1, 3],
        [3, 3, 1, 3, 1, 3, 1, 3],
        [3, 3, 1, 3, 1, 1, 3, 3],
        [3, 1, 3, 3, 3, 3, 1, 1],
        [3, 1, 3, 3, 3, 1, 3, 1],
        [3, 1, 3, 3, 1, 3, 3, 1],
        [3, 1, 3, 3, 3, 1, 1, 3],
        [3, 1, 3, 3, 1, 3, 1, 3],
        [1, 3, 3, 3, 1, 1, 3, 3],
        [1, 3, 3, 3, 3, 3, 1, 1],
        [1, 3, 3, 3, 3, 1, 3, 1],
        [1, 3, 3, 3, 1, 3, 3, 1],
        [1, 3, 3, 3, 3, 1, 1, 3],
        [1, 3, 3, 3, 1, 3, 1, 3],
        [1, 1, 3, 3, 1, 3, 3, 3],
        [3, 3, 1, 1, 3, 3, 3, 1],
    ]) / 2


def get_packed_abs_grid():
    intr = torch.arange(-4, 4)
    d8 = torch.cartesian_prod(*[intr] * 8).float() + 1 / 2
    d8m2 = (d8.sum(dim=-1) % 2 == 0)
    d8n = d8.norm(dim=-1)**2 <= 10
    d8abs = torch.unique(d8[sorted(torch.where(d8m2 * d8n)[0])].abs(), dim=0)
    norm12 = get_norm12()
    cba = torch.concat([d8abs, norm12], dim=0)
    cba = cba[:, [0, 2, 1, 3, 4, 6, 5, 7]]
    cba[:,7] *= (1 - 2 * (cba.sum(1) % 2))
    cba = cba * 4
    cba = cba.to(torch.int64)
    acc = cba[:,0]
    for i in range(7):
        acc = acc | (cba[:,(i+1)] << ((i+1)*8))
    return acc


def get_abs_grid():
    intr = torch.arange(-4, 4)
    d8 = torch.cartesian_prod(*[intr] * _E8P_CODESZ).float() + 1 / 2
    d8m2 = (d8.sum(dim=-1) % 2 == 0)
    d8n = d8.norm(dim=-1)**2 <= 10
    d8abs = torch.unique(d8[sorted(torch.where(d8m2 * d8n)[0])].abs(), dim=0)
    norm12 = get_norm12()
    return torch.concat([d8abs, norm12], dim=0)


def get_full_grid(packed_abs_grid):
    synth_codebook = torch.zeros(1 << 16, 8)
    shuffle_map = [0, 2, 1, 3, 4, 6, 5, 7]
    for c in range(1 << 16):
        signs = c & 255
        abs = c >> 8
        parity = 0
        for i in range(8):
            parity = parity ^ ((signs >> i) & 1)
        signs = signs ^ parity
        abs_code = packed_abs_grid[abs].item()
        for i in range(8):
            ii = shuffle_map[i]
            synth_codebook[c,i] = np.int8((abs_code >> (8 * ii)) & 255) / 4
            if ((signs >> (7 - ii)) & 1):
                synth_codebook[c,i] *= -1
        if parity:
            synth_codebook[c,:] -= 0.25
        else:
            synth_codebook[c,:] += 0.25
    return synth_codebook, torch.arange(1 << 16)


class E8P12_codebook(nn.Module):

    def __init__(self, inference=False, **kwargs):
        super(E8P12_codebook, self).__init__()
        self.id = "E8P12"
        self.opt_scale = 1.03
        self.codesz = _E8P_CODESZ
        self.idx_dtype = torch.int16
        self.packsz = 1
        self.pack_out = False
        self.version = 1

        self.register_buffer('grid_packed_abs', get_packed_abs_grid())

        if not inference:
            _E8P_GRID, _ = get_full_grid(self.grid_packed_abs)
            self.register_buffer('grid', _E8P_GRID)
            self.register_buffer('grid_norm', _E8P_GRID.norm(dim=-1)**2)

    def round(self, X, grid, grid_norm):
        assert X.shape[-1] == self.codesz
        Xqidx = (2 * X @ grid.T - grid_norm).argmax(-1)
        return grid[Xqidx], Xqidx

    def quantize(self, X, return_idx=True):
        final_vals, final_idxs = self.round(X, self.grid, self.grid_norm)
        if return_idx:
            return final_vals, final_idxs
        return final_vals

    def maybe_pack_idxs(self, idxs):
        return idxs

    def forward(self,
                input,
                Qidxs):
        m, n = Qidxs.shape
        if input.size(0) < 32:
            output = quiptools_cuda.e8p_mm_origorder(
                input,
                Qidxs,
                self.grid_packed_abs
            )
        else:
            W_decompressed = torch.empty(
                Qidxs.shape[0], Qidxs.shape[1] * _E8P_CODESZ,
                dtype=torch.float16, device=input.device
            )
            quiptools_cuda.decompress_e8p_origorder(
                Qidxs, self.grid_packed_abs, W_decompressed
            )
            output = input @ W_decompressed.T
        return output
