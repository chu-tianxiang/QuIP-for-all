# GPT-Fast

[GPT-Fast](https://github.com/pytorch-labs/gpt-fast) adapted for quip-sharp.

Currently only LLaMA and QWen models are supported.

## Usage

Please make sure you have installed the cuda kernel in the `quip_cuda` folder.
```
python setup_cuda.py install
```

* Convert QUIP checkpoint to GPT-Fast

```bash
python convert_hf_checkpoint.py --checkpoint_dir Qwen-72B-Chat-2bit
```

* Inference

The `compile` process will takes several minutes.

```bash
python generate.py --checkpoint_path Qwen-72B-Chat-2bit/model_int2_rand.pth --interactive --compile
```