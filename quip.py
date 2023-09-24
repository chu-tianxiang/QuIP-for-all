# Copyright 2023 HuggingFace Inc. team and GPTQ and AutoGPTQ authors.
#
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
import torch.nn as nn
import transformers

from method import QuantMethod
from vector_balance import quantize_weight_vecbal


class Balance(QuantMethod):

    def __init__(self, layer, qmethod, nbits, npasses):
        super().__init__(layer)
        self.qmethod = qmethod
        self.nbits = nbits
        self.npasses = npasses

    def fasterquant(self, lazy_batch=False):
        w = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            raise NotImplementedError()
        if isinstance(self.layer, transformers.Conv1D):
            raise NotImplementedError()

        H = self.H.data.clone()
        self.quantizer.find_params(w)
        dequant_w, quant_w = quantize_weight_vecbal(w=w,
                                                    H=H,
                                                    nbits=self.nbits,
                                                    npasses=self.npasses,
                                                    scale=self.quantizer.scale,
                                                    zero=self.quantizer.zero,
                                                    maxq=self.quantizer.maxq,
                                                    qfn=self.quantizer.qfn,
                                                    qmethod=self.qmethod,
                                                    lazy_batch=lazy_batch)
        self.layer.weight.data = dequant_w
        self.postproc()
        return quant_w, self.quantizer.scale, self.scaleWH, self.subU, self.subV
