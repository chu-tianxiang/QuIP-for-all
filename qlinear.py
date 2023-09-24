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
import numpy as np
import torch
import torch.nn as nn

from method import butterfly_factors


class QuantLinear(nn.Module):

    def __init__(
        self,
        bits,
        infeatures,
        outfeatures,
        bias=True,
    ):
        super().__init__()
        if bits not in [2, 3, 4, 8]:
            raise NotImplementedError("Only 2,3,4,8 bits are supported.")
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.maxq = 2**self.bits - 1
        h1, w1 = butterfly_factors(infeatures)
        h2, w2 = butterfly_factors(outfeatures)

        self.register_buffer(
            'qweight',
            torch.zeros((infeatures // 32 * self.bits, outfeatures),
                        dtype=torch.int32))
        self.register_buffer(
            'scale', torch.zeros((1, outfeatures), dtype=torch.float16))
        self.register_buffer('u1', torch.zeros((h2, h2), dtype=torch.float16))
        self.register_buffer('u2', torch.zeros((w2, w2), dtype=torch.float16))
        self.register_buffer('u_inp',
                             torch.zeros(outfeatures, dtype=torch.int32))
        self.register_buffer('u_outp',
                             torch.zeros(outfeatures, dtype=torch.int32))
        self.register_buffer('v1', torch.zeros((h1, h1), dtype=torch.float16))
        self.register_buffer('v2', torch.zeros((w1, w1), dtype=torch.float16))
        self.register_buffer('v_inp', torch.zeros(infeatures,
                                                  dtype=torch.int32))
        self.register_buffer('v_outp',
                             torch.zeros(infeatures, dtype=torch.int32))
        self.register_buffer('scale_hw',
                             torch.zeros(infeatures, dtype=torch.float16))

        if bias:
            self.register_buffer(
                'bias', torch.zeros((outfeatures), dtype=torch.float16))
        else:
            self.bias = None
        self.half_indim = self.infeatures // 2

        # is performed by unpacking the weights and using torch.matmul
        if self.bits in [2, 4, 8]:
            self.wf = torch.tensor(list(range(0, 32, self.bits)),
                                   dtype=torch.int32).unsqueeze(0)
        elif self.bits == 3:
            self.wf = torch.tensor([
                [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0],
                [0, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31],
                [0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0],
            ],
            dtype=torch.int32).reshape(1, 3, 12)

    def pack(self, linear, qweight, scale, scale_hw, sub_u, sub_v):
        scale_t = scale.t().contiguous()
        self.scale = scale_t.clone().half()
        self.scale_hw = scale_hw.clone().half()
        (u1, u2), u_inp, u_outp = sub_u
        self.u_inp = u_inp.argsort()
        self.u_outp = u_outp.argsort()
        self.u1 = u1.clone().half()
        self.u2 = u2.clone().half()
        (v1, v2), v_inp, v_outp = sub_v
        self.v_inp = v_inp
        self.v_outp = v_outp
        self.v1 = v1.clone().half()
        self.v2 = v2.clone().half()

        if linear.bias is not None:
            self.bias = linear.bias.clone().half()

        intweight = []
        for idx in range(self.infeatures):
            intweight.append(
                torch.round(qweight[:, idx]).to(torch.int)[:, None])
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)

        i = 0
        row = 0
        qweight = np.zeros(
            (intweight.shape[0] // 32 * self.bits, intweight.shape[1]),
            dtype=np.uint32)
        while row < qweight.shape[0]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (32 // self.bits)):
                    qweight[row] |= intweight[j] << (self.bits * (j - i))
                i += 32 // self.bits
                row += 1
            elif self.bits == 3:
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i))
                i += 10
                qweight[row] |= intweight[i] << 30
                row += 1
                qweight[row] |= (intweight[i] >> 2) & 1
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i) + 1)
                i += 10
                qweight[row] |= intweight[i] << 31
                row += 1
                qweight[row] |= (intweight[i] >> 1) & 0x3
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i) + 2)
                i += 10
                row += 1
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight)

    def forward(self, x):
        out_shape = x.shape[:-1] + (self.outfeatures, )
        x = x.reshape(-1, x.shape[-1]).half()
        if self.wf.device != self.qweight.device:
            self.wf = self.wf.to(self.qweight.device)
        if self.bits in [2, 4, 8]:
            weight = torch.bitwise_right_shift(
                torch.unsqueeze(self.qweight,
                                1).expand(-1, 32 // self.bits, -1),
                self.wf.unsqueeze(-1)).to(torch.int16 if self.bits ==
                                          8 else torch.int8)
            torch.bitwise_and(weight, (2**self.bits) - 1, out=weight)
            weight = weight.reshape(-1, weight.shape[-1])

        elif self.bits == 3:
            weight = self.qweight.reshape(self.qweight.shape[0] // 3, 3, 1,
                                          self.qweight.shape[1]).expand(
                                              -1, -1, 12, -1)
            weight = (weight >> self.wf.unsqueeze(-1)) & 0x7
            weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | (
                (weight[:, 1, 0] << 2) & 0x4)
            weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | (
                (weight[:, 2, 0] << 1) & 0x6)
            weight = weight & 0x7
            weight = torch.cat(
                [weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]],
                dim=1)
            weight = weight.reshape(-1, weight.shape[-1])
        else:
            raise NotImplementedError("Only 2,3,4,8 bits are supported.")
        weight = self.scale * ((weight / self.maxq) * 2 - 1)
        batch_size = x.shape[0]
        x = x / self.scale_hw[None, :]
        x = x[:, self.v_inp].view(batch_size, self.v1.shape[0],
                                  self.v2.shape[0])
        x = torch.matmul(torch.matmul(self.v1, x),
                         self.v2.T).view(batch_size, -1)[:, self.v_outp]
        out = torch.matmul(x, weight.half())
        out = out[:, self.u_outp].view(batch_size, self.u1.shape[0],
                                       self.u2.shape[0])
        out = torch.matmul(torch.matmul(self.u1.T, out),
                           self.u2).view(batch_size, -1)[:, self.u_inp]

        out = out.half().reshape(out_shape)
        out = out + self.bias if self.bias is not None else out
        return out
