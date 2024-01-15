# https://github.com/Cornell-RelaxML/quip-sharp/blob/main/lib/codebook/latticed4.py
"""
builds a deep-hole-centered D4 codebook
this is a codebook consisting of points on the lattice in R4
    where each component is a half-integer
    and the components sum to an even number
from this lattice, we select the points that have a norm-squared of at most 9
this results in a codebook of 256 points distributed as follows
    8 with sorted abs of [1/2, 1/2, 1/2, 1/2]
    8                    [3/2, 3/2, 3/2, 3/2]
    4c2 * 8 = 48         [1/2, 1/2. 3/2, 3/2]
    4 * 8 = 32           [1/2, 1/2, 1/2, 3/2]
    4 * 8 = 32           [1/2, 3/2, 3/2, 3/2]
    4 * 8 = 32           [1/2, 1/2, 1/2, 5/2]
    4 * 3 * 8 = 96       [1/2, 1/2, 3/2, 5/2]
"""

import torch
from torch import nn

import quiptools_cuda


_D4_CODESZ = 4


def code3_signs(i3, x):
    if (i3 & (1 << 5)):
        x[2] *= -1
    if (i3 & (1 << 6)):
        x[1] *= -1
    if (sum(x) % 2 != 0):
        x[3] *= -1
    if (i3 & (1 << 7)):
        for j in range(_D4_CODESZ):
            x[j] *= -1
    assert (sum(x) % 2 == 0)
    return x


def code8_to_d4(i8):
    assert ((i8 >= 0) and (i8 < 256))
    i3 = i8 & (7 << 5)
    i8 = i8 & 31
    if i8 < 16:
        if i8 < 8:
            if i8 < 2:
                if i8 < 1:
                    return code3_signs(i3, [0.5] * _D4_CODESZ)
                else:
                    return code3_signs(i3, [1.5] * _D4_CODESZ)
            else:
                ibx = i8 >> 1
                if i8 & 1:
                    x = [0.5] * _D4_CODESZ
                    x[0] = 1.5
                    x[ibx] = 1.5
                else:
                    x = [1.5] * _D4_CODESZ
                    x[0] = 0.5
                    x[ibx] = 0.5
                return code3_signs(i3, x)
        else:
            ibx = (i8 & 3)
            if i8 < 8 + 4:
                x = [0.5] * _D4_CODESZ
                x[ibx] = 1.5
            else:
                x = [1.5] * _D4_CODESZ
                x[ibx] = 0.5
            return code3_signs(i3, x)
    else:
        if i8 < 16 + 4:
            ibx = (i8 & 3)
            x = [0.5] * _D4_CODESZ
            x[ibx] = 2.5
            return code3_signs(i3, x)
        else:
            ibx = i8 - 20
            ib4 = ibx & 3
            ib3 = ibx >> 2
            x = [0.5] * _D4_CODESZ
            x[ib4] = 1.5
            if (ib3 >= ib4):
                ib3 += 1
            x[ib3] = 2.5
            return code3_signs(i3, x)


def build_D4_CB():
    CB = torch.zeros(256, _D4_CODESZ)
    for i in range(256):
        x = code8_to_d4(i)
        for j in range(_D4_CODESZ):
            CB[i, j] = x[j]
    return CB


class D4_codebook(nn.Module):

    def __init__(self, inference=False, **kwargs):
        super(D4_codebook, self).__init__()
        self.id = "D4"
        self.register_buffer("grid", build_D4_CB())
        if not inference:
            self.register_buffer('grid_norm', (self.grid @ self.grid.T).diag())
        self.codesz = _D4_CODESZ
        self.opt_scale = 1.21
        self.idx_dtype = torch.uint8
        self.packsz = 1
        self.pack_out = False
        self.version = 0

    def _quantize_noscale(self, X, return_idx=True):
        Xqidx = (2 * X @ self.grid.T - self.grid_norm).argmax(1)
        if return_idx:
            return self.grid[Xqidx, :], Xqidx.to(self.idx_dtype)
        return self.grid[Xqidx, :]

    def quantize(self, X, return_idx=True):
        assert X.shape[-1] == self.codesz
        return self._quantize_noscale(X, return_idx=return_idx)

    def maybe_pack_idxs(self, idxs):
        return idxs

    def forward(self,
                input,
                Qidxs):
        if input.shape[0] < 24:
            output = quiptools_cuda.d4_mm_origorder(input, Qidxs, self.grid)
        else:
            W_decompressed = torch.zeros(
                Qidxs.shape[0], Qidxs.shape[1] * _D4_CODESZ,
                dtype=torch.float16, device=input.device
            )
            quiptools_cuda.decompress_d4_origorder(Qidxs, self.grid, W_decompressed)
            output = input @ W_decompressed.t()
        return output
