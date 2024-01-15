# Modified from https://github.com/Cornell-RelaxML/quip-sharp/blob/main/lib/linear/quantized_linear.py
import torch
import torch.nn as nn

from quant import (get_hadK, matmul_hadUt_cuda, matmul_hadU_cuda)


class QuantLinear(nn.Module):

    def __init__(self,
                 in_features,
                 out_features,
                 codebook,
                 bias=True,
                 use_rand=True,
                 per_channel=False,
                 weight_dtype=torch.float16):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.codebook = codebook
        self.use_rand = use_rand
        self.per_channel = per_channel
        self.weight_dtype = weight_dtype

        had_left, self.K_left, self.q_in_features = get_hadK(in_features, use_rand)
        had_right, self.K_right, self.q_out_features = get_hadK(out_features, use_rand)
        if had_left is not None:
            self.register_buffer('had_left',
                                 had_left.to(weight_dtype),
                                 persistent=use_rand)
        else:
            self.had_left = None
        if had_right is not None:
            self.register_buffer('had_right',
                                 had_right.to(weight_dtype),
                                 persistent=use_rand)
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
        if self.per_channel:
            self.register_buffer("Wscale", torch.ones((self.q_out_features), dtype=weight_dtype))
        else:
            self.register_buffer("Wscale", torch.ones((), dtype=torch.float))
        self.wscale_float = 1.0

        if bias:
            self.register_buffer(
                'bias', torch.zeros((out_features), dtype=weight_dtype))
        else:
            self.bias = None


    def forward(self, input):
        x = input.view(-1, input.shape[-1])
        x_dtype = x.dtype
        if self.SU is not None:
            x = x * self.SU
        x = matmul_hadUt_cuda(x, self.had_left, self.K_left,
                              self.q_in_features, self.wscale_float)
        if x_dtype != torch.float16:
            x = x.to(torch.float16)
        out = self.codebook(x, self.Qidxs)
        if x_dtype != torch.float16:
            out = out.to(dtype=x_dtype)
        if self.per_channel:
            out = out * self.Wscale
        out = matmul_hadU_cuda(out, self.had_right, self.K_right,
                               self.q_out_features)[..., :self.out_features]

        if self.SV is not None:
            out = out * self.SV
        out = out.view(*input.shape[:-1], out.shape[-1])
        out = out + self.bias if self.bias is not None else out
        return out

    def pack(self, linear, attr):
        if attr["scaleWH"] is not None and not attr["merge_su"]:
            self.SU = (attr["SU"] * attr["scaleWH"]).to(self.weight_dtype)
        elif attr["scaleWH"] is not None:
            self.SU = attr["scaleWH"].to(self.weight_dtype)
        elif not attr["merge_su"]:
            self.SU = attr["SU"].to(self.weight_dtype)
        else:
            self.SU = None
        self.Qidxs = attr["Qidxs"].clone()
        if self.per_channel:
            self.Wscale = attr["w_scale"].squeeze().to(self.weight_dtype)
        else:
            self.Wscale = attr["w_scale"].to(torch.float32)
        if not attr["merge_sv"]:
            self.SV = attr["SV"].to(self.weight_dtype)
        else:
            self.SV = None
        self.codebook = None
        if attr["left_hadK"] is not None:
            self.had_left = attr["left_hadK"]
        if attr["right_hadK"] is not None:
            self.had_right = attr["right_hadK"]

        if linear.bias is not None:
            if attr["merge_sv"]:
                self.bias = (linear.bias / attr["SV"]).to(self.weight_dtype)
            else:
                self.bias = linear.bias.to(self.weight_dtype)
