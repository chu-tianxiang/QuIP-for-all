# Modified from https://github.com/Cornell-RelaxML/quip-sharp/blob/main/lib/algo/quip.py
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
import math
import os

import torch
import torch.nn as nn
import transformers

from quant import (
    LDLQ,
    LDLQ_buffered,
    get_hadK,
    matmul_hadU,
    matmul_hadUt,
)


class QUIP:
    '''
    Base class for quantization methods
    '''

    def __init__(self, layer, cb):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns),
                             dtype=torch.float64,
                             device=self.dev)
        self.mu = torch.zeros((self.columns, ),
                              dtype=torch.float64,
                              device=self.dev)
        self.nsamples = 0
        self.preproc_done = False
        self.cb = cb.to(self.dev)

    def add_batch(self, inp, out):
        if os.environ.get("DEBUG"):
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(
                self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(self.layer.kernel_size,
                               dilation=self.layer.dilation,
                               padding=self.layer.padding,
                               stride=self.layer.stride)
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.mu *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        self.mu += inp.sum(dim=1).to(torch.float64) / self.nsamples
        inp = math.sqrt(2 / self.nsamples) * inp.to(torch.float64)
        self.H += inp.matmul(inp.t())

    def quant(self,
              rescale_WH=False,
              use_fp64=False,
              sigma_reg=0.01,
              scale_override=0,
              use_buffered=True,
              use_rand=True,
              quip_tune_iters=0):
        self.rescale_WH = rescale_WH
        if not use_fp64:
            self.H = self.H.to(torch.float32)

        w = self.layer.weight.data.clone().to(self.H.dtype)
        if isinstance(self.layer, nn.Conv2d):
            w = w.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            w = w.t()
        H = self.H.clone()
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        w[:, dead] = 0

        H.div_(torch.diag(H).mean())
        # diag = torch.arange(H.shape[0], device=H.device)
        # H[diag, diag] += sigma_reg

        if rescale_WH:
            H /= H.abs().max()
            diagH = torch.diag(H)
            diagW2 = torch.diag(w.T @ w)
            diagH = torch.clamp(diagH, min=1e-8)
            diagW2 = torch.clamp(diagW2, min=1e-8)
            scaleWH = (diagH / diagW2).sqrt().sqrt().to(torch.float32)
            scaleWH = scaleWH.clamp(min=1e-8)
            w *= scaleWH[None, :]
            H /= scaleWH[None, :]
            H /= scaleWH[:, None]

        if hasattr(self.layer, "SU"):
            merge_su = True
            SU = self.layer.SU.to(self.H.dtype)
        else:
            merge_su = False
            SU = (torch.randn(self.columns, device=self.dev).sign() +
                  1e-5).sign().to(self.H.dtype)
        if hasattr(self.layer, "SV"):
            merge_sv = True
            SV = self.layer.SV.to(self.H.dtype)
        else:
            merge_sv = False
            SV = (torch.randn(self.rows, device=self.dev).sign() + 1e-5).sign().to(
                self.H.dtype)
        left_hadK, left_K, left_N = get_hadK(self.columns, use_rand=use_rand)
        right_hadK, right_K, right_N = get_hadK(self.rows, use_rand=use_rand)
        H = matmul_hadUt(matmul_hadUt(H * SU, left_hadK, left_K, left_N).T * SU, left_hadK, left_K, left_N)
        w = matmul_hadUt(matmul_hadUt(w.T * SV, right_hadK, right_K, right_N).T * SU, left_hadK, left_K, left_N)

        attempts = 0
        while True:
            try:
                diag = torch.arange(H.shape[0], device=H.device)
                H[diag, diag] += sigma_reg
                L = torch.linalg.cholesky(H)
                if torch.any(torch.isnan(L)):
                    raise RuntimeError
                break
            except RuntimeError:
                attempts += 1
                if attempts == 10:
                    raise ValueError("Hessian is not invertible")

        w_scale = w.square().mean().sqrt()
        if scale_override > 0:
            w_scale /= scale_override
        else:
            w_scale /= self.cb.opt_scale
        w = w / w_scale
        if not use_buffered:
            hat_w, Qidxs = LDLQ(w, H, L, self.cb, quip_tune_iters)
        else:
            hat_w, Qidxs = LDLQ_buffered(w,
                                         H,
                                         L,
                                         self.cb,
                                         quip_tune_iters,
                                         buf_cols=128)
        hat_w = hat_w * w_scale

        w = (matmul_hadU((matmul_hadU(hat_w, left_hadK, left_K, left_N)[..., :self.columns] *
                          SU.to(self.dev)).T, right_hadK, right_K, right_N)[..., :self.rows] *
             SV.to(self.dev)).T
        if self.rescale_WH:
            w = w / scaleWH[None, :]
        if isinstance(self.layer, transformers.Conv1D):
            w = w.t()
        self.layer.weight.data = w.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype)
        Qidxs = self.cb.maybe_pack_idxs(Qidxs)
        attr = {
            'left_hadK': left_hadK.to('cpu') if use_rand and left_hadK is not None else None,
            'right_hadK': right_hadK.to('cpu') if use_rand and right_hadK is not None else None,
            'Qidxs': Qidxs.to('cpu'),
            'w_scale': w_scale.to('cpu'),
            'SU': SU.to('cpu'),
            'SV': SV.to('cpu'),
            'merge_su': merge_su,
            'merge_sv': merge_sv,
            'scaleWH': scaleWH.to('cpu') if self.rescale_WH else None,
        }
        return attr

    def free(self):
        if os.environ.get("DEBUG"):
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.layer.SU = None
        self.layer.SV = None
        torch.cuda.empty_cache()
