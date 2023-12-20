# QuIP

This is an adaptation of [quip-sharp](https://github.com/Cornell-RelaxML/quip-sharp) to support a wider range of  model architectures.

There're a few changes making it incompatable with the checkpoints provided by quip team.
* Every linear layer is quantized seperated without fusions like concatenating QKV layers.
* D4 weights are not permuted. New kernels are implemented to better support batch-size=1.

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
