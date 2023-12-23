# Modified from https://github.com/Cornell-RelaxML/quip-sharp/blob/main/lib/linear/quantized_linear.py
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

from quant import (get_hadK, matmul_hadUt_cuda, matmul_hadU_cuda)


class QuantLinear(nn.Module):

    def __init__(self,
                 in_features,
                 out_features,
                 codebook,
                 bias=True,
                 weight_dtype=torch.float16):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.codebook = codebook
        self.weight_dtype = weight_dtype

        had_left, self.K_left, self.q_in_features = get_hadK(in_features)
        had_right, self.K_right, self.q_out_features = get_hadK(out_features)
        if had_left is not None:
            self.register_buffer('had_left',
                                 had_left.to(weight_dtype),
                                 persistent=False)
        else:
            self.had_left = None
        if had_right is not None:
            self.register_buffer('had_right',
                                 had_right.to(weight_dtype),
                                 persistent=False)
        else:
            self.had_right = None

        # direction we pack in, the code dimension is always in the in dimension
        if codebook.pack_out:
            self.register_buffer(
                "Qidxs",
                torch.zeros(self.q_out_features // codebook.packsz,
                            self.q_in_features // codebook.codesz,
                            dtype=codebook.idx_dtype))
        else:
            self.register_buffer(
                "Qidxs",
                torch.zeros(self.q_out_features,
                            self.q_in_features //
                            (codebook.codesz * codebook.packsz),
                            dtype=codebook.idx_dtype))

        self.register_buffer("SU", torch.ones(in_features, dtype=weight_dtype))
        self.register_buffer("SV", torch.ones(out_features, dtype=weight_dtype))

        if bias:
            self.register_buffer(
                'bias', torch.zeros((out_features), dtype=weight_dtype))
        else:
            self.bias = None

    def forward(self, input):
        x = input.view(-1, input.shape[-1])
        x_dtype = x.dtype
        x = x * self.SU
        x = matmul_hadUt_cuda(x, self.had_left, self.K_left,
                              self.q_in_features)
        if x_dtype != torch.float16:
            x = x.to(torch.float16)
        out = self.codebook(x, self.Qidxs)
        if x_dtype != torch.float16:
            out = out.to(dtype=x_dtype)

        out = matmul_hadU_cuda(out, self.had_right, self.K_right,
                               self.q_out_features)[..., :self.out_features]

        out = out * self.SV
        out = out.view(*input.shape[:-1], out.shape[-1])
        out = out + self.bias if self.bias is not None else out

        return out

    def pack(self, linear, attr):
        if attr["scaleWH"] is not None:
            self.SU = (attr["SU"] * attr["scaleWH"] * attr["w_scale"]).to(self.weight_dtype)
        else:
            self.SU = (attr["SU"] * attr["w_scale"]).to(self.weight_dtype)
        self.Qidxs = attr["Qidxs"].clone()
        self.SV = attr["SV"].to(self.weight_dtype)
        self.codebook = None

        if linear.bias is not None:
            self.bias = linear.bias.clone()
