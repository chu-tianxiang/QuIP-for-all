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
            self.scaleWH = scaleWH.to(torch.float32).cpu()

        SU = (torch.randn(self.columns, device=self.dev).sign() +
              1e-5).sign().to(self.H.dtype)
        SV = (torch.randn(self.rows, device=self.dev).sign() + 1e-5).sign().to(
            self.H.dtype)
        H = matmul_hadUt(matmul_hadUt(H * SU).T * SU)
        w = matmul_hadUt(matmul_hadUt(w.T * SV).T * SU)
        self.SU = SU.cpu()
        self.SV = SV.cpu()

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

        w = (matmul_hadU((matmul_hadU(hat_w)[..., :self.columns] *
                          self.SU.to(self.dev)).T)[..., :self.rows] *
             self.SV.to(self.dev)).T
        if self.rescale_WH:
            scaleWH = self.scaleWH.to(w.device)
            w = w / scaleWH[None, :]
        if isinstance(self.layer, transformers.Conv1D):
            w = w.t()
        self.layer.weight.data = w.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype)
        attr = {
            'Qidxs': Qidxs.to('cpu'),
            'w_scale': w_scale.to('cpu'),
            'SU': self.SU.clone().to(torch.int8),
            'SV': self.SV.clone().to(torch.int8),
            'scaleWH': self.scaleWH.clone() if self.rescale_WH else None,
        }
        return attr

    def free(self):
        if os.environ.get("DEBUG"):
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.scaleWH = None
        self.SU = None
        self.SV = None
        torch.cuda.empty_cache()
