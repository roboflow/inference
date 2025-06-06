""" CLIP tokenizer

Copied from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""

import gzip
import html
import os
import random
import string
from functools import lru_cache, partial
from typing import Callable, List, Optional, Union

import ftfy
import numpy as np
import regex as re
import torch

# https://stackoverflow.com/q/62691279
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_CONTEXT_LENGTH = 77  # default context length for OpenAI CLIP


@lru_cache()
def default_bpe():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz"
    )


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def _clean_canonicalize(x):
    # basic, remove whitespace, remove punctuation, lower case
    return canonicalize_text(basic_clean(x))


def _clean_lower(x):
    # basic, remove whitespace, lower case
    return whitespace_clean(basic_clean(x)).lower()


def _clean_whitespace(x):
    # basic, remove whitespace
    return whitespace_clean(basic_clean(x))


def get_clean_fn(type: str):
    if type == "canonicalize":
        return _clean_canonicalize
    elif type == "lower":
        return _clean_lower
    elif type == "whitespace":
        return _clean_whitespace
    else:
        assert False, f"Invalid clean function ({type})."


def canonicalize_text(text, *, keep_punctuation_exact_string=None):
    """Returns canonicalized `text` (lowercase and punctuation removed).

    From: https://github.com/google-research/big_vision/blob/53f18caf27a9419231bbf08d3388b07671616d3d/big_vision/evaluators/proj/image_text/prompt_engineering.py#L94

    Args:
      text: string to be canonicalized.
      keep_punctuation_exact_string: If provided, then this exact string kept.
        For example providing '{}' will keep any occurrences of '{}' (but will
        still remove '{' and '}' that appear separately).
    """
    text = text.replace("_", " ")
    if keep_punctuation_exact_string:
        text = keep_punctuation_exact_string.join(
            part.translate(str.maketrans("", "", string.punctuation))
            for part in text.split(keep_punctuation_exact_string)
        )
    else:
        text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class SimpleTokenizer(object):
    def __init__(
        self,
        bpe_path: str = default_bpe(),
        additional_special_tokens: Optional[List[str]] = None,
        context_length: Optional[int] = DEFAULT_CONTEXT_LENGTH,
        clean: str = "lower",
        reduction_mask: str = "",
    ):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split("\n")
        merges = merges[1 : 49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        special_tokens = ["<start_of_text>", "<end_of_text>"]
        if additional_special_tokens:
            special_tokens += additional_special_tokens
        vocab.extend(special_tokens)
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {t: t for t in special_tokens}
        special = "|".join(special_tokens)
        self.pat = re.compile(
            special + r"""|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )
        self.vocab_size = len(self.encoder)
        self.all_special_ids = [self.encoder[t] for t in special_tokens]
        self.sot_token_id = self.all_special_ids[0]
        self.eot_token_id = self.all_special_ids[1]
        self.context_length = context_length
        self.clean_fn = get_clean_fn(clean)
        self.reduction_fn = (
            get_reduction_mask_fn(reduction_mask) if reduction_mask else None
        )

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = self.clean_fn(text)
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(
                self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" ")
            )
        return bpe_tokens

    def decode(self, tokens):
        text = "".join([self.decoder[token] for token in tokens])
        text = (
            bytearray([self.byte_decoder[c] for c in text])
            .decode("utf-8", errors="replace")
            .replace("</w>", " ")
        )
        return text

    def __call__(
        self, texts: Union[str, List[str]], context_length: Optional[int] = None
    ) -> torch.LongTensor:
        """Returns the tokenized representation of given input string(s)

        Parameters
        ----------
        texts : Union[str, List[str]]
            An input string or a list of input strings to tokenize
        context_length : int
            The context length to use; all CLIP models use 77 as the context length

        Returns
        -------
        A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
        """
        if isinstance(texts, str):
            texts = [texts]

        context_length = context_length or self.context_length
        assert context_length, "Please set a valid context length"

        if self.reduction_fn is not None:
            # use reduction strategy for tokenize if set, otherwise default to truncation below
            return self.reduction_fn(
                texts,
                context_length=context_length,
                sot_token_id=self.sot_token_id,
                eot_token_id=self.eot_token_id,
                encode_fn=self.encode,
            )

        all_tokens = [
            [self.sot_token_id] + self.encode(text) + [self.eot_token_id]
            for text in texts
        ]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                tokens = tokens[:context_length]  # Truncate
                tokens[-1] = self.eot_token_id
            result[i, : len(tokens)] = torch.tensor(tokens)

        return result


def random_mask_tokenize(
    texts: Union[str, List[str]],
    context_length: int,
    sot_token_id: int,
    eot_token_id: int,
    encode_fn: Callable,
    shuffle: bool = False,
):
    all_tokens = [encode_fn(text) for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        tokens = torch.tensor(tokens)
        num_tokens = len(tokens)
        if num_tokens > context_length - 2:  # 2 for sot and eot token
            num_keep = context_length - 2
            indices = torch.randperm(len(tokens))
            indices = indices[:num_keep]
            if not shuffle:
                indices = indices.msort()
            tokens = tokens[indices]
            num_tokens = num_keep
        result[i, 0] = sot_token_id
        result[i, 1 : num_tokens + 1] = tokens
        result[i, num_tokens + 1] = eot_token_id

    return result


def simple_mask_tokenize(
    texts: Union[str, List[str]],
    context_length: int,
    sot_token_id: int,
    eot_token_id: int,
    encode_fn: Callable,
):
    all_tokens = [encode_fn(text) for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        num_tokens = len(tokens)
        if num_tokens > context_length - 2:  # 2 for sot and eot token
            num_keep = context_length - 2
            start_index = random.randint(0, num_tokens - num_keep)  # high is incl
            tokens = tokens[start_index : start_index + num_keep]
        tokens = [sot_token_id] + tokens + [eot_token_id]
        result[i, : len(tokens)] = torch.tensor(tokens)

    return result



def get_reduction_mask_fn(type: str):
    """Choose strategy for dropping (masking) tokens to achieve target context length"""
    assert type in ("simple", "random", "shuffle")
    if type == "simple":
        return simple_mask_tokenize  # randomly select block [start:end]
    elif type == "random":
        return random_mask_tokenize  # randomly drop tokens (keep order)
    elif type == "shuffle":
        return partial(
            random_mask_tokenize, shuffle=True
        )  # randomly drop tokens (shuffle order)