# QuIP

This repo is a adaptation of [jerry-chee/QuIP](https://github.com/jerry-chee/QuIP).
* more model architectures
* model save and load
* channel-wise quantization

Please install the cuda kernel first.
```
pip install -r requirements.txt
python setup_cuda.py install
```

The following are perplexity scores of LLaMA-2-70b on Wikitext dataset with 512 stride and 2048  max length. models are quantized with random samples from C4.

* fp16: 4.062
* 3bit: 4.508 ([Huggingface Link](https://huggingface.co/keyfan/llama-2-70b-3bit))
* 2bit: 7.150 ([Huggingface Link](https://huggingface.co/keyfan/llama-2-70b-2bit))

## Usage

* Quantize
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from quantizer import QuipQuantizer

model_name = "meta-llama/Llama-2-70b-hf"
quant_dir = "llama-70b_2bit_quip"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

quant = QuipQuantizer(bits=2, dataset="c4")
quant_model = quant.quantize_model(model, tokenizer)
quant.save(quant_model, quant_dir)
```

* Inference
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights
from quantizer import load_quantized_model

model_name = "meta-llama/Llama-2-70b-hf"
quant_dir = "llama-70b_2bit_quip"

with init_empty_weights():
    empty_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
empty_model.tie_weights()
quant_model = load_quantized_model(empty_model, save_folder=quant_dir, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

input_ids = tokenizer.encode("The capital of France is", return_tensors="pt").cuda()
print(tokenizer.decode(quant_model.generate(input_ids, do_sample=True)[0]))
```
