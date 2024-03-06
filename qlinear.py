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
                 weight_dtype=torch.float16,
                 train=False):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.codebook = codebook
        self.use_rand = use_rand
        self.per_channel = per_channel
        self.weight_dtype = weight_dtype
        self.train_mode = train

        had_left, self.K_left, self.q_in_features = get_hadK(in_features, use_rand)
        had_right, self.K_right, self.q_out_features = get_hadK(out_features, use_rand)
        if had_left is not None:
            self.register_buffer("had_left",
                                 had_left.to(weight_dtype),
                                 persistent=use_rand)
        else:
            self.had_left = None
        if had_right is not None:
            self.register_buffer("had_right",
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

        self.register_parameter(
            "SU",
            torch.nn.Parameter(
                torch.ones(in_features, dtype=weight_dtype),
                requires_grad=True
            ))
        self.register_parameter(
            "SV",
            torch.nn.Parameter(
                torch.ones(out_features, dtype=weight_dtype),
                requires_grad=True
            ))
        if self.per_channel:
            self.register_buffer("Wscale", torch.ones((self.q_out_features), dtype=weight_dtype))
        else:
            self.register_buffer("Wscale", torch.ones((), dtype=torch.float))
        self.wscale_float = 1.0
        # transformers use weight to decide device
        # so we register a fake weight here
        self.register_buffer('weight', torch.zeros((),  dtype=weight_dtype))

        if bias:
            self.register_buffer(
                "bias", torch.zeros((out_features), dtype=weight_dtype))
        else:
            self.bias = None


    def forward(self, input):
        x = input.view(-1, input.shape[-1])
        x_dtype = x.dtype
        if self.SU is not None:
            x = x * self.SU

        if self.train_mode:
            if x.shape[-1] != self.q_in_features:
                x = torch.nn.functional.pad(x, (0, self.q_in_features - x.shape[-1]))
            W = self.W if hasattr(self, "W") else self.calc_weight(cache=False).to(x.dtype)
            out = (x @ W)[..., :self.out_features]
        else:
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
            self.SU.data.copy_(attr["SU"] * attr["scaleWH"])
        elif attr["scaleWH"] is not None:
            self.SU.data.copy_(attr["scaleWH"])
        elif not attr["merge_su"]:
            self.SU.data.copy_(attr["SU"])
        else:
            self.SU = None

        if not attr["merge_sv"]:
            self.SV.data.copy_(attr["SV"])
        else:
            self.SV = None

        self.Qidxs.copy_(attr["Qidxs"])
        self.Wscale.copy_(attr["w_scale"].squeeze(
            ) if self.per_channel else attr["w_scale"])

        if attr["left_hadK"] is not None:
            self.had_left.copy_(attr["left_hadK"])
        if attr["right_hadK"] is not None:
            self.had_right.copy_(attr["right_hadK"])

        if linear.bias is not None:
            self.bias.copy_(linear.bias / attr["SV"] if attr["merge_sv"] else linear.bias)

    @torch.no_grad()
    def calc_weight(self, cache=True):
        weight = self.codebook.decompress_weight(self.Qidxs)
        wscale_float = self.Wscale.mean().float().item()
        had_left = self.had_left.to(weight.dtype) if self.had_left is not None else None
        had_right = self.had_right.to(weight.dtype) if self.had_right is not None else None
        W = matmul_hadU_cuda(
                matmul_hadU_cuda(weight, had_left, self.K_left,
                                 self.q_in_features, wscale_float).T,
                had_right, self.K_right, self.q_out_features
            ).to(self.weight_dtype)
        if self.per_channel:
            W = W * self.Wscale / self.Wscale.mean()
        if cache:
            self.register_buffer("W", W, persistent=False)
        return W
