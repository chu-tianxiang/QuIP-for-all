# QuIP

This is an adaptation of [quip-sharp](https://github.com/Cornell-RelaxML/quip-sharp) to support a wider range of  model architectures.

There're a few changes making it incompatable with the checkpoints provided by quip team.
* Every linear layer is quantized seperated without fusions like concatenating QKV layers.
* The packing format is slightly different, weights are simply packed as `(outdim, indim / codesz)` shape without complex permuting.

## Usage

Please install the cuda kernel in the `quip_cuda` folder.
```
python setup_cuda.py install
```

* Quantize

By default 4096 samples of calibration data will be used as [suggested by the author](https://github.com/Cornell-RelaxML/quip-sharp/issues/13#issuecomment-1848867522), which is very time consuming.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from quantizer import QuipQuantizer

model_name = "meta-llama/Llama-2-70b-hf"
quant_dir = "llama-70b_2bit_quip"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

quant = QuipQuantizer(codebook="E8P12", dataset="c4", nsamples=4096)
quant_model = quant.quantize_model(model, tokenizer)
quant.save(quant_model, quant_dir)
tokenizer.save_pretrained(quant_dir)
```

* Inference
```python
from transformers import AutoTokenizer
from quantizer import load_quantized_model

quant_dir = "llama-70b_2bit_quip"

quant_model = load_quantized_model(quant_dir).cuda()
tokenizer = AutoTokenizer.from_pretrained(quant_dir)

input_ids = tokenizer.encode("The capital of France is", return_tensors="pt").cuda()
print(tokenizer.decode(quant_model.generate(input_ids, do_sample=True)[0]))
```

## Speedup

### GPT-Fast
In the `gpt-fast` folder I include simply codes to use `torch.compile` for speedup. The code are adapted from [GPT-Fast](https://github.com/pytorch-labs/gpt-fast). Currently only Llama and Qwen are supported.

Testes at A100:
* Qwen-72B: 14 tokens/s
* Llama-7b: 93 tokens/s

### vLLM

Working on the a [custom branch](https://github.com/chu-tianxiang/vllm-gptq/tree/quip_gemv) of vLLM now.
Unfunately tensor-parallel is not supported because Hadamard transform cannot be done for sharded input. Currently the generation speed is about 86 tokens/s for Llama-7b at batchsize=1 in single A100.