# https://github.com/huggingface/optimum/blob/main/optimum/gptq/data.py
#  Copyright 2023 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import load_dataset
"""
Set of utilities for loading most used datasets (original dataset from GPTQ paper) and be able to easily use them during quantization
"""


def prepare_dataset(examples: List[Dict[str, torch.LongTensor]],
                    batch_size: int = 1):
    """
    Prepare the dataset by making sure that we have the right format and `batch_size`
    Args:
        examples (`List[Dict[str, torch.LongTensor]]`):
            List of data to prepare
        batch_size (`int`, defaults to `1`):
            Batch size of the data
    Returns:
        ` List[Dict[str, torch.LongTensor]]`: Batched dataset
    """
    new_examples = []
    for example in examples:
        input_ids = example["input_ids"]
        attention_mask = example["attention_mask"]
        new_examples.append({
            "input_ids": torch.LongTensor(input_ids),
            "attention_mask": torch.LongTensor(attention_mask)
        })
    new_examples = [
        collate_data(new_examples[start:start + batch_size])
        for start in range(0, len(new_examples), batch_size)
    ]
    return new_examples


def collate_data(
    blocks: List[Dict[str, torch.LongTensor]],
    contain_labels: bool = False,
) -> Dict[str, torch.LongTensor]:
    """
        Collate data in `blocks`
    Args:
        blocks (`List[Dict[str, torch.LongTensor]]`):
            List of tensors that we need to batch together
        contain_labels (`bool`, defaults to `False`):
           Set True to also process the labels

    Returns:
        `Dict[str, torch.LongTensor]`: Batched data
    """

    input_ids_blocks = [block["input_ids"] for block in blocks]
    attention_mask_blocks = [block["attention_mask"] for block in blocks]
    if contain_labels:
        label_blocks = [block["labels"] for block in blocks]

    data = {
        "input_ids": torch.cat(input_ids_blocks, dim=0).long(),
        "attention_mask": torch.cat(attention_mask_blocks, dim=0).long(),
    }
    if contain_labels:
        data["labels"] = torch.cat(label_blocks, dim=0).long()

    return data


def get_wikitext2(tokenizer: Any,
                  seqlen: int,
                  nsamples: int,
                  split: str = "train"):
    if split == "train":
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    elif split == "validation":
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "".join([" \n" if s == "" else s for s in data["text"]])

    enc = tokenizer(text, return_tensors="pt")
    dataset = []
    for k in range(nsamples):
        i = random.randint(0, enc.input_ids.shape[1] - seqlen)
        j = i + seqlen
        inp = enc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        dataset.append({"input_ids": inp, "attention_mask": attention_mask})
    return dataset


def get_c4(tokenizer: Any, seqlen: int, nsamples: int, split: str = "train"):
    if split == "train":
        data = load_dataset(
            "allenai/c4",
            "allenai--c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train")
    elif split == "validation":
        data = load_dataset(
            "allenai/c4",
            "allenai--c4",
            data_files={
                "validation": "en/c4-validation.00000-of-00008.json.gz"
            },
            split="validation",
        )
    dataset = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(data) - 1)
            enc = tokenizer(data[i]["text"], return_tensors="pt")
            if enc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, enc.input_ids.shape[1] - seqlen)
        j = i + seqlen
        inp = enc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        dataset.append({"input_ids": inp, "attention_mask": attention_mask})

    return dataset


def get_c4_new(tokenizer: Any,
               seqlen: int,
               nsamples: int,
               split: str = "train"):
    if split == "train":
        data = load_dataset(
            "allenai/c4",
            "allenai--c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train")
    elif split == "validation":
        data = load_dataset(
            "allenai/c4",
            "allenai--c4",
            data_files={
                "validation": "en/c4-validation.00000-of-00008.json.gz"
            },
            split="validation",
        )
    dataset = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(data) - 1)
            enc = tokenizer(data[i]["text"], return_tensors="pt")
            if enc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, enc.input_ids.shape[1] - seqlen)
        j = i + seqlen
        inp = enc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        dataset.append({"input_ids": inp, "attention_mask": attention_mask})

    return dataset


def get_ptb(tokenizer: Any, seqlen: int, nsamples: int, split: str = "train"):
    if split == "train":
        data = load_dataset("ptb_text_only", "penn_treebank", split="train")
    elif split == "validation":
        data = load_dataset("ptb_text_only",
                            "penn_treebank",
                            split="validation")

    enc = tokenizer(" ".join(data["sentence"]), return_tensors="pt")

    dataset = []
    for _ in range(nsamples):
        i = random.randint(0, enc.input_ids.shape[1] - seqlen)
        j = i + seqlen
        inp = enc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        dataset.append({"input_ids": inp, "attention_mask": attention_mask})

    return dataset


def get_ptb_new(tokenizer: Any,
                seqlen: int,
                nsamples: int,
                split: str = "train"):
    if split == "train":
        data = load_dataset("ptb_text_only", "penn_treebank", split="train")
    elif split == "validation":
        data = load_dataset("ptb_text_only", "penn_treebank", split="test")

    enc = tokenizer(" ".join(data["sentence"]), return_tensors="pt")

    dataset = []
    for _ in range(nsamples):
        i = random.randint(0, enc.input_ids.shape[1] - seqlen)
        j = i + seqlen
        inp = enc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        dataset.append({"input_ids": inp, "attention_mask": attention_mask})
    return dataset


def get_redpajama(tokenizer: Any, seqlen: int, nsamples: int, split: str = "train"):
    assert split == "train"
    data = load_dataset(
        "togethercomputer/RedPajama-Data-1T-Sample",
        split="train")

    dataset = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(data) - 1)
            enc = tokenizer(data[i]["text"], return_tensors="pt")
            if enc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, enc.input_ids.shape[1] - seqlen)
        j = i + seqlen
        inp = enc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        dataset.append({"input_ids": inp, "attention_mask": attention_mask})
    return dataset


def get_dataset(dataset_name: str,
                tokenizer: Any,
                nsamples: int = 128,
                seqlen: int = 2048,
                seed: int = 0,
                split: str = "train"):
    """
    Get the dataset from the original paper of GPTQ

    Args:
        dataset_name (`str`):
            Dataset name. Available options are `['wikitext2', 'c4', 'ptb', 'c4-new', 'ptb_new']`.
        tokenizer (`Any`):
            Tokenizer of the model
        nsamples (`int`, defaults to `128`):
            Number of samples
        seqlen (`int`, defaults to `2048`):
            The sequence length of the model
        seed (`int`, defaults to `0`):
            Seed
        split (`str`, defaults to `train`):
            Split of the dataset. Can be either "train" or "validation"
    Returns:
        `List[Dict[str,torch.LongTensor]]`: The tokenized dataset.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    get_dataset_map = {
        "wikitext2": get_wikitext2,
        "c4": get_c4,
        "c4-new": get_c4_new,
        "ptb": get_ptb,
        "ptb-new": get_ptb_new,
        "redpajama": get_redpajama,
    }
    if split not in ["train", "validation"]:
        raise ValueError(
            f"The split need to be 'train' or 'validation' but found {split}")
    if dataset_name not in get_dataset_map:
        raise ValueError(
            f"Expected a value in {list(get_dataset_map.keys())} but found {dataset_name}"
        )
    get_dataset_fn = get_dataset_map[dataset_name]
    return get_dataset_fn(tokenizer=tokenizer,
                          nsamples=nsamples,
                          seqlen=seqlen,
                          split=split)
