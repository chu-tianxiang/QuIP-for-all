# https://github.com/Cornell-RelaxML/quip-sharp/blob/main/lib/codebook/latticee8_padded12.py
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
"""
D8^ = D8 + 1/2 intersected with ball of radius sqrt(10)
|D8^| has 227 entries
We then add 29 entries from the set of vectors with 5 3/2 and 3 1/2
The total codebook is all 2^7 flips of these 256 entries (2^15) +- 1/4
which makes 2^16 entries.
This corresponds to a subset of E8 + 1/4
"""

import torch
from torch import nn

import quiptools_cuda


_E8P_CODESZ = 8
_INT_MAP = 2**(torch.arange(_E8P_CODESZ).flip(0))


def int2mask(i, int_map):
    return ((i & int_map) > 0).int()


def mask2int(mask, int_map):
    return (int_map.unsqueeze(0) * mask.int()).sum(dim=-1)


def get_abs_grid():
    intr = torch.arange(-4, 4)
    d8 = torch.cartesian_prod(*[intr] * _E8P_CODESZ).to(torch.float) + 1 / 2
    d8m2 = (d8.sum(dim=-1) % 2 == 0)
    d8n = d8.norm(dim=-1) ** 2 <= 10
    d8abs = torch.unique(d8[sorted(torch.where(d8m2 * d8n)[0])].abs(), dim=0)

    norm12 = torch.tensor([
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
        [1, 3, 3, 3, 1, 1, 3, 3],
        [3, 3, 1, 1, 3, 3, 3, 1],
    ]) / 2
    return torch.concat([d8abs, norm12], dim=0)


def get_full_grid(abs_grid):
    """
    idx format:
        - first 8 bits = which of the 256 entries in the abs grid
        - next 7 bits = which of the right 7 dims to negate (8th can be inferred)
        - last bit = +1/4 if true else -1/4
    """
    is_even_flips = abs_grid.sum(dim=-1) % 2 == 0
    abs_idxs = torch.arange(len(abs_grid)) << _E8P_CODESZ
    entries = [[], []]
    idxs = [[], []]
    for i in range(2**(_E8P_CODESZ - 1)):
        mask = int2mask(i, _INT_MAP)
        mask_even = (mask.sum(dim=-1) % 2 == 0)
        mask = mask.unsqueeze(0).repeat(len(abs_grid), 1)
        mask[:, 0] = mask_even != is_even_flips
        mask = 1 - 2 * mask
        entries[0].append(abs_grid * mask + 1 / 4)
        idxs[0].append(abs_idxs + (i << 1) + 1)
        entries[1].append(abs_grid * mask - 1 / 4)
        idxs[1].append(abs_idxs + (i << 1))

    for i in range(2):
        entries[i] = torch.concat(entries[i], dim=0)
        idxs[i] = torch.concat(idxs[i], dim=0)
    entries = torch.concat(entries, dim=0)
    idxs = torch.concat(idxs, dim=0)
    return entries, idxs


_E8P_ABS_CACHED = get_abs_grid()
_E8P_GRID, _E8P_GRID_IDX = get_full_grid(_E8P_ABS_CACHED)


class E8P12_codebook(nn.Module):

    def __init__(self, inference=False):
        super(E8P12_codebook, self).__init__()
        self.id = "E8P12"
        self.opt_scale = 1  #.03#/1.09
        self.codesz = _E8P_CODESZ
        self.idx_dtype = torch.int16
        self.idx_offset = -2**15
        self.packsz = 1
        self.pack_out = False
        self.version = 0

        self.register_buffer('grid_abs', _E8P_ABS_CACHED)
        self.register_buffer('grid_abs_even', self.grid_abs.sum(dim=-1) % 2 == 0)

        if not inference:
            self.register_buffer('int_map', _INT_MAP)
            self.register_buffer('grid', _E8P_GRID)
            self.register_buffer('grid_idx_map',
                                 (_E8P_GRID_IDX + self.idx_offset).to(self.idx_dtype))
            idx_lut = torch.zeros(_E8P_GRID_IDX.shape).int()
            idx_lut[_E8P_GRID_IDX] = torch.arange(len(_E8P_GRID_IDX)).int()
            self.register_buffer('grid_idx_inv', idx_lut)

            self.register_buffer('grid_norm', torch.diag(self.grid @ self.grid.T))
            grid_part = self.grid[:len(self.grid) // 2] - 1 / 4
            idxs = torch.where(
                ((grid_part[:, 1:] < 0).sum(dim=-1) <= 1) * \
                (grid_part[:, 1:].min(dim=-1).values >= -0.5)
            )[0]
            grid_part = grid_part[idxs]
            self.register_buffer('grid_part', grid_part)
            self.register_buffer('grid_part_norm', torch.diag(grid_part @ grid_part.T))
            allcombo_idx, idx_map = self.iterate_mask()
            self.register_buffer('allcombo_idx', allcombo_idx)
            self.register_buffer('idx_map', idx_map)
        else:
            self.register_buffer('codebook_matvec', torch.zeros((256, ), dtype=torch.int64))
            for i in range(8):
                chunk = (self.grid_abs[:, i] * 4).to(torch.int64)
                self.codebook_matvec |= chunk << (i * 8)

    def iterate_mask(self, device=0):
        flips = torch.stack([((torch.tensor([i]) & self.int_map) > 0).int()
                             for i in range(2**_E8P_CODESZ)]).to(device)
        raw_idx = torch.where(flips.sum(dim=-1) % 2 == 0)[0]
        flips = 1 - 2 * flips[raw_idx]
        idx_map = torch.zeros(2**_E8P_CODESZ, dtype=torch.int32)
        for i in range(len(raw_idx)):
            idx_map[raw_idx[i]] = i
        allcombo = flips.unsqueeze(1) * self.grid_part.unsqueeze(0).to(device)
        allcombo_idx = torch.zeros(allcombo.shape[0:2]).int()
        for i in range(len(allcombo)):
            allcombo_idx[i] = self.round(allcombo[i], self.grid.to(device),
                                         self.grid_norm.to(device))[1]
        return allcombo_idx.cpu(), idx_map.cpu()

    def round(self, X, grid, grid_norm):
        assert X.shape[-1] == self.codesz
        Xqidx = (2 * X @ grid.T - grid_norm).argmax(-1)
        return grid[Xqidx], Xqidx

    def fast_quantize_part(self, X):
        X_part = torch.abs(X)
        X_odd = torch.where((X < 0).sum(dim=-1) % 2 != 0)[0]
        X_part[X_odd, 0] = -X_part[X_odd, 0]
        mask = 1 - 2 * (X < 0).to(torch.float)
        mask[X_odd, 0] = -mask[X_odd, 0]
        roundout, Xqidx = self.round(X_part, self.grid_part, self.grid_part_norm)
        vals = roundout * mask
        real_idx = self.allcombo_idx[self.idx_map[mask2int((1 - mask) / 2, self.int_map)], Xqidx]
        err = (X - vals).norm(dim=-1)
        return vals, real_idx, err

    def quantize(self, X, return_idx=True):
        X_plus = X + 1 / 4  # quantize X to D8^ - 1/4
        X_minus = X - 1 / 4  # quantize X to D8^ + 1/4

        plus_vals, plus_idx, plus_err = self.fast_quantize_part(X_plus)
        minus_vals, minus_idx, minus_err = self.fast_quantize_part(X_minus)
        plus_idx = plus_idx + 2 ** 15

        which = plus_err < minus_err
        final_vals = torch.where(which.unsqueeze(-1), plus_vals - 1 / 4, minus_vals + 1 / 4)

        if return_idx:
            final_idxs = self.grid_idx_map[torch.where(which, plus_idx, minus_idx)]
            return final_vals, final_idxs

        return final_vals

    def by_idxs(self, idxs, **kwargs):
        return self.grid[self.grid_idx_inv[idxs.int() - self.idx_offset]]

    def forward(self,
                input,
                Qidxs,
                Wscale):
        if input.size(0) < 8:
            output = quiptools_cuda.e8p_mm_cuda(input, Qidxs - 0x8000,
                                                self.codebook_matvec)
        else:
            W_decompressed = torch.zeros(Qidxs.shape[0],
                                         Qidxs.shape[1] * _E8P_CODESZ,
                                         device=Qidxs.device,
                                         dtype=torch.float16)
            quiptools_cuda.decompress_e8p(Qidxs, self.grid_abs,
                                          self.grid_abs_even, W_decompressed)
            output = input @ W_decompressed.T
        output *= Wscale
        return output
