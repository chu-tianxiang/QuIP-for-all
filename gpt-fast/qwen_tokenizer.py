import base64
import unicodedata
from typing import Dict

import tiktoken

PAT_STR = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
ENDOFTEXT = "<|endoftext|>"
IMSTART = "<|im_start|>"
IMEND = "<|im_end|>"
# as the default behavior is changed to allow special tokens in
# regular texts, the surface forms of special tokens need to be
# as different as possible to minimize the impact
EXTRAS = tuple((f"<|extra_{i}|>" for i in range(205)))
# changed to use actual index to avoid misconfiguration with vocabulary expansion
SPECIAL_START_ID = 151643
SPECIAL_TOKENS = tuple(
    enumerate(
        (
            (
                ENDOFTEXT,
                IMSTART,
                IMEND,
            )
            + EXTRAS
        ),
        start=SPECIAL_START_ID,
    )
)
SPECIAL_TOKENS_SET = set(t for i, t in SPECIAL_TOKENS)


def _load_tiktoken_bpe(tiktoken_bpe_file: str) -> Dict[bytes, int]:
    with open(tiktoken_bpe_file, "rb") as f:
        contents = f.read()
    return {
        base64.b64decode(token): int(rank)
        for token, rank in (line.split() for line in contents.splitlines() if line)
    }


class QwenTokenizer:
    def __init__(self, vocab_file):
        self.mergeable_ranks = _load_tiktoken_bpe(vocab_file)  # type: Dict[bytes, int]
        self.special_tokens = {
            token: index
            for index, token in SPECIAL_TOKENS
        }
        self.tokenizer = tiktoken.Encoding(
            "Qwen",
            pat_str=PAT_STR,
            mergeable_ranks=self.mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        self.decoder = {
            v: k for k, v in self.mergeable_ranks.items()
        }  # type: dict[int, bytes|str]
        self.decoder.update({v: k for k, v in self.special_tokens.items()})


    def encode(self, text):
        text = unicodedata.normalize("NFC", text)
        return self.tokenizer.encode(text, allowed_special="all")

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

    def eos_id(self):
        # <|im_end|>
        return 151645
