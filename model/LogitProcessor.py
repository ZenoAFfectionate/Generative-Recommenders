from transformers.generation import LogitsProcessor
from transformers import AutoTokenizer
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import math
import numpy as np
import torch
import warnings

from transformers.utils import add_start_docstrings

LOGITS_PROCESSOR_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
        scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
            search or log softmax for each vocabulary token when using beam search

    Return:
        `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`: The processed prediction scores.

"""


class SIDTrie:
    """Trie for constrained SID decoding.

    Stores valid token sequences and provides prefix-based lookup
    for valid next tokens via tree traversal instead of hash lookup.
    """
    __slots__ = ('children',)

    def __init__(self):
        self.children = {}  # token_id -> SIDTrie

    def insert(self, token_ids):
        """Insert a valid token sequence (SID tokens + EOS)."""
        node = self
        for tid in token_ids:
            if tid not in node.children:
                node.children[tid] = SIDTrie()
            node = node.children[tid]

    def get_valid_tokens(self, prefix):
        """Return valid next tokens given a prefix of generated tokens.

        For empty prefix, returns all valid first tokens (root children).
        Returns empty list if prefix leads to an invalid path.
        """
        node = self
        for tid in prefix:
            if tid not in node.children:
                return []
            node = node.children[tid]
        return list(node.children.keys())

    @classmethod
    def build(cls, token_sequences):
        """Build a trie from an iterable of token ID sequences."""
        trie = cls()
        for seq in token_sequences:
            trie.insert(seq)
        return trie


class ConstrainedLogitsProcessor(LogitsProcessor):

    def __init__(
        self,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
        num_beams: int,
        prefix_index: int = 3,
        eos_token_id: int = None
    ):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = num_beams
        self.count = 0
        self.prefix_index = prefix_index
        self.eos_token_id = eos_token_id
        self._cached_mask = None
        self._warned = False


    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = torch.nn.functional.log_softmax(scores, dim=-1)

        # Reuse cached mask tensor to avoid repeated allocation
        if self._cached_mask is None or self._cached_mask.shape != scores.shape:
            self._cached_mask = torch.full_like(scores, float('-inf'))
        else:
            self._cached_mask.fill_(float('-inf'))
        mask = self._cached_mask

        # Pass only the generated SID tokens (not the response prefix)
        if self.count == 0:
            all_keys = [[] for _ in range(input_ids.shape[0])]
        else:
            all_keys = input_ids[:, -self.count:].cpu().tolist()

        total_beams = input_ids.shape[0]
        for i in range(total_beams):
            batch_id = i // self._num_beams
            prefix_allowed_tokens = self._prefix_allowed_tokens_fn(batch_id, all_keys[i])

            if len(prefix_allowed_tokens) == 0:
                if not self._warned:
                    warnings.warn(
                        f"No valid tokens found at step {self.count}. "
                        f"This indicates the model generated an unexpected token. "
                        f"Subsequent warnings will be suppressed."
                    )
                    self._warned = True
                if self.eos_token_id is not None:
                    mask[i, self.eos_token_id] = 0
                continue

            mask[i, prefix_allowed_tokens] = 0

        self.count += 1

        scores = scores + mask
        return scores
