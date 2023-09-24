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
import torch
import torch.nn as nn


def quantize_qfna(x, scale, zero, maxq):
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)


def quantize_qfnb(x, scale, maxq):
    q = x / scale
    q = torch.clamp(torch.round(((q + 1) / 2) * maxq), 0, maxq)
    q = (q / maxq) * 2 - 1
    q = q * scale
    return q


class Quantizer(nn.Module):

    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(self, bits, perchannel=False, sym=True, qfn='a'):
        self.maxq = torch.tensor(2**bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.qfn = qfn

    def find_params(self, x):
        if self.qfn == 'a':
            self.find_params_qfna(x)
        elif self.qfn == 'b':
            self.find_params_qfnb(x)

    def find_params_qfna(self, x):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            x = x.flatten(1)
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        self.scale = (xmax - xmin) / self.maxq
        if self.sym:
            self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
        else:
            self.zero = torch.round(-xmin / self.scale)

        if not self.perchannel:
            tmp = shape[0]
            self.scale = self.scale.repeat(tmp)

        shape = [-1] + [1] * (len(shape) - 1)
        self.scale = self.scale.reshape(shape)

    def find_params_qfnb(self, x):
        dev = x.device
        shape = x.shape
        self.maxq = self.maxq.to(dev)
        if self.perchannel:
            x = x.flatten(1)
        else:
            x = x.flatten().unsqueeze(0)
        self.scale = 2.4 * x.square().mean(1).sqrt() + 1e-16
        if not self.perchannel:
            tmp = shape[0]
            self.scale = self.scale.repeat(tmp)
        shape = [-1] + [1] * (len(shape) - 1)
        self.scale = self.scale.reshape(shape)
        self.zero = None

    def quantize(self, x):
        if self.qfn == 'a':
            assert self.ready()
            return quantize_qfna(x, self.scale, self.zero, self.maxq)
        elif self.qfn == 'b':
            assert torch.all(self.maxq != 0)
            return quantize_qfnb(x, self.scale, self.maxq)
        else:
            return NotImplementedError()
