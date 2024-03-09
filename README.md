# QuIP

This is an adaptation of [official quip-sharp repo](https://github.com/Cornell-RelaxML/quip-sharp) to support a wider range of  model architectures.

There're a few changes making it incompatable with the checkpoints provided by quip team.
* Every linear layer is quantized seperated without fusions like concatenating QKV layers.
* The packing format is slightly different, weights are simply packed as `(outdim, indim / codesz)` shape without complex permuting.

## Usage

Please install the cuda kernel in the `quip_cuda` folder.
```
python setup_cuda.py install
```

### Quantize

Please refer to the [author's blog](https://cornell-relaxml.github.io/quip-sharp/) for introduction of quip# algorithm.


```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from quantizer import QuipQuantizer

model_name = "meta-llama/Llama-2-70b-hf"
quant_dir = "llama-70b_2bit_quip"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

quant = QuipQuantizer(codebook="E8P12", dataset="redpajama")
quant.quantize_model(model, tokenizer, quant_dir)
```

Arguments of `QuipQuantizer` includes:
* codebook: the algorithm for quantization, including `E8P12`(2-bit), `E8P12RVQ3B`(3-bit), `E8P12RVQ4B`(4-bit), `D4`(2-bit), `HI`(4-bit). `D4` and `HI` are relatively inferior to E8P12-based methods.
* dataset: the data used for calibration, supports `c4`, `ptb`, `wikitext2`, `redpajama`.
* nsamples: number of samples used for calibration, larger is slower. By default 4096 samples of calibration data will be used as [suggested by the author](https://github.com/Cornell-RelaxML/quip-sharp/issues/13#issuecomment-1848867522), which will consume 500 - 750 GB CPU memory. Please reduce `nsamples` and `ft_train_size` if you don't have enough CPU memory.
* quip_tune_iters: Greedy update passes of the algorithm, higher is slower but yields slightly better quanlity. Default to 10.
* use_rand: A boolean flag that determines the decomposition strategy for dimensionality that is not a power of two. Say `dim = 2^n * base`, `True` will decompose it into `2^n` Hadamard matrix and `base x base` random orthogonal matrices, `False` will decompose into two Hadamard matrix following the original implementation and use padding when such decomposition cannot be found. Default to true.
* modules_to_not_convert: the name of layers not to quantize, useful for MOE models where gate layer is often unquantized.
* merge_suv: An optional optimization that can cancel out certain vectors to reduce computation. It is currently only supported for llama, mixtral, and qwen models. The default setting is false.
* ft_lr: The learning rate for the fine-tuning stage, as described in the paper's appendix. The default value is 5e-5.
* ft_susv_lr: The learning rate for the SU/SV parameters during fine-tuning. The default value is 5e-4.
* ft_epochs: The number of epochs for fine-tuning. The default is set to 5.
* ft_train_size: The size of the training dataset used for fine-tuning. Larger sizes require **significantly** more CPU memory as it needs to calculate a larger logit matrix. The default is 384.
* ft_valid_size: The size of the validation dataset for fine-tuning. The default is 128.
* ft_batch_size: Batch size for the fine-tuning process. The default is 8.
* ft_valid_freq: The frequency, in epochs, at which the validation is run. The default is every epoch.
* ft_early_stop: The number of epochs to wait for an improvement in validation loss before early stopping. The default is 3 epochs.
* ft_embedding: Whether finetune input and output embedding layer during end2end finetune stage. The default is false.

### Inference
```python
from transformers import AutoTokenizer
from quantizer import load_quantized_model

quant_dir = "llama-70b_2bit_quip"

quant_model = load_quantized_model(quant_dir).cuda()
tokenizer = AutoTokenizer.from_pretrained(quant_dir)

input_ids = tokenizer.encode("The capital of France is", return_tensors="pt").cuda()
print(tokenizer.decode(quant_model.generate(input_ids, do_sample=True)[0]))
```

### Finetune
Finetune with Lora adapter is supported. Please check `example_finetune.py` as a minial example.

## Speedup

### torch.compile
For simplicity I removed the old `gpt-fast` examples and added a simple script based on the `huggingface/transformers` newly introduced static cache.

It's slower than `gpt-fast` though. If you wanna use `gpt-fast` example, please refer to the [previous commit](https://github.com/chu-tianxiang/QuIP-for-all/blob/412dd470918f5312a9ed055c2cddf9e2d1d838f5/gpt-fast/generate.py).

```
python example_generate.py --model_path llama-70b_2bit_quip --streaming --compile
```

### vLLM

Working on the a [custom branch](https://github.com/chu-tianxiang/vllm-gptq/tree/gptq_hf) of vLLM now.
Unfunately tensor-parallel is not supported because Hadamard transform cannot be done for sharded input. Currently the generation speed is about 86 tokens/s for Llama-7b at batchsize=1 in single A100.
