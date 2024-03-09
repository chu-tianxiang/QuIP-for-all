# Midified from https://github.com/casper-hansen/AutoAWQ/blob/main/examples/train.py
import types

import datasets
import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.tuners.lora import (
    LoraConfig,
    LoraModel,
    QuantLinear as PeftQuantLinear
)
from quantizer import load_quantized_model, QuantLinear

# Monkey patch PEFT
def create_new_module(lora_config, adapter_name, target, **kwargs):
    new_module = None
    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, QuantLinear):
        new_module = PeftQuantLinear(target, adapter_name, lora_config=lora_config, **kwargs)
        target.weight = target_base_layer.weight
    return new_module

LoraModel._create_new_module = staticmethod(create_new_module)

def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            names = name.split('.')
            lora_module_names.add(names[-1])
    return list(lora_module_names)


def prepare_split(tokenizer):
    data = datasets.load_dataset("mhenrichsen/alpaca_2k_test", split="train")
    prompt_template = "<s>[INST] {prompt} [/INST] {output}</s>"

    def format_prompt(x):
        return prompt_template.format(
            prompt=x["instruction"],
            output=x["output"]
        )

    data = data.map(
        lambda x: {"text": format_prompt(x)},
    ).select_columns(["text"])
    data = data.map(lambda x: tokenizer(x["text"]), batched=True)

    return data

model_path = "llama-70b_2bit_quip"

# Load model
model = load_quantized_model(model_path, torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

# Prepare data
data_train = prepare_split(tokenizer)

# Config Lora
# transformers trainer will try to read hf_quantizer.is_trainable
# so we hack it by adding a fake hf_quantizer
model.is_quantized = True
model.hf_quantizer = types.SimpleNamespace(is_trainable=True)
modules = find_all_linear_names(model)
lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=modules,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    inference_mode=False,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_arguments = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=1,
    optim="adamw_torch",
    num_train_epochs=1,
    learning_rate=1e-4,
    evaluation_strategy="no",
    save_strategy="epoch",
    save_steps=100,
    logging_steps=50,
    eval_steps=None,
    load_best_model_at_end=False
)

trainer = Trainer(
    model=model,
    train_dataset=data_train,
    args=training_arguments,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()
trainer.save_model("output")
