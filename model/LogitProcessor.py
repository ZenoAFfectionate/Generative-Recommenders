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


# ---------------------------------------------------------------------------
# STATIC: Vectorized CSR-based constrained decoding (GPU-accelerated)
#
# Based on: "Vectorizing the Trie: Efficient Constrained Decoding for
# LLM-based Generative Retrieval on Accelerators" (Su et al., 2026)
# Reference impl: github.com/youtube/static-constraint-decoding
# ---------------------------------------------------------------------------

def _build_static_index(
    sids: np.ndarray,
    vocab_size: int,
    dense_lookup_layers: int = 2,
):
    """Build a STATIC index (CSR + dense tables) from sorted SID sequences.

    Adapted from google/static-constraint-decoding (Apache-2.0).

    Args:
        sids: Sorted array of SID token sequences. Shape: (N, L).
        vocab_size: Number of distinct tokens (V).
        dense_lookup_layers: Number of initial layers for dense O(1) lookup.

    Returns:
        packed_csr, indptr, layer_max_branches, start_mask, dense_mask, dense_states
    """
    import gc

    N, L = sids.shape
    if dense_lookup_layers >= L:
        raise ValueError(
            f"dense_lookup_layers ({dense_lookup_layers}) must be < SID length ({L})."
        )

    # Level-0 mask
    start_mask = np.zeros(vocab_size, dtype=bool)
    start_mask[np.unique(sids[:, 0])] = True

    # Identify unique trie nodes via prefix diffs
    diff = sids[1:] != sids[:-1]
    first_diff = np.full(N - 1, L, dtype=np.int8)
    has_diff = diff.any(axis=1)
    first_diff[has_diff] = diff[has_diff].argmax(axis=1)

    is_new = np.zeros((N, L), dtype=bool)
    is_new[0, :] = True
    for depth in range(L):
        is_new[1:, depth] = first_diff <= depth

    # State ID assignment
    state_ids = np.zeros((N, L - 1), dtype=np.int32)
    state_ids[:, 0] = sids[:, 0].astype(np.int32) + 1
    depth_id_ranges = []
    current_offset = vocab_size + 1
    for depth in range(1, L - 1):
        mask = is_new[:, depth]
        num_new = np.sum(mask)
        depth_id_ranges.append((current_offset, current_offset + num_new))
        state_ids[mask, depth] = np.arange(current_offset, current_offset + num_new, dtype=np.int32)
        state_ids[:, depth] = np.maximum.accumulate(state_ids[:, depth])
        current_offset += num_new
    num_states = current_offset

    # Collect edges
    all_parents, all_tokens, all_children = [], [], []
    for depth in range(1, L):
        mask = is_new[:, depth]
        parent_ids = state_ids[mask, depth - 1]
        token_ids = sids[mask, depth].astype(np.int32)
        child_ids = (
            state_ids[mask, depth] if depth < L - 1
            else np.zeros_like(parent_ids, dtype=np.int32)
        )
        all_parents.append(parent_ids)
        all_tokens.append(token_ids)
        all_children.append(child_ids)

    # Dense specialization
    dense_shape = tuple([vocab_size] * dense_lookup_layers)
    dense_mask = np.zeros(dense_shape, dtype=bool)
    dense_states = np.zeros(dense_shape, dtype=np.int32)
    indices = tuple(sids[:, i].astype(np.int32) for i in range(dense_lookup_layers))
    final_dense_ids = state_ids[:, dense_lookup_layers - 1]
    dense_mask[indices] = True
    dense_states[indices] = final_dense_ids

    # CSR construction
    parents = np.concatenate(all_parents)
    tokens = np.concatenate(all_tokens)
    children = np.concatenate(all_children)
    del state_ids, is_new; gc.collect()

    counts = np.bincount(parents, minlength=num_states)
    indptr = np.zeros(num_states + 1, dtype=np.int32)
    indptr[1:] = np.cumsum(counts)

    # Layer max branches
    layer_max_branches = [int(np.sum(start_mask))]
    l0_counts = counts[1:vocab_size + 1]
    layer_max_branches.append(int(l0_counts.max()) if len(l0_counts) > 0 else 0)
    for start_id, end_id in depth_id_ranges:
        if start_id < len(counts):
            layer_max_branches.append(int(counts[start_id:end_id].max()) if end_id > start_id else 0)
        else:
            layer_max_branches.append(0)
    while len(layer_max_branches) < L:
        layer_max_branches.append(1)

    # Pack [token, next_state]
    raw_indices = np.concatenate([tokens, np.full(vocab_size, vocab_size, dtype=np.int32)])
    raw_data = np.concatenate([children, np.zeros(vocab_size, dtype=np.int32)])
    indptr = np.append(indptr, indptr[-1] + vocab_size)
    packed_csr = np.ascontiguousarray(np.vstack([raw_indices, raw_data]).T)

    return packed_csr, indptr, tuple(layer_max_branches), start_mask, dense_mask, dense_states


class STATICIndex:
    """GPU-resident STATIC index for vectorized constrained decoding.

    Remaps sparse LLM token IDs to a contiguous space, builds CSR + dense
    tables, and provides vectorized masking over the full LLM vocabulary.
    """

    def __init__(self, sid_token_sequences: List[List[int]], eos_token_id: int,
                 device: torch.device = None):
        """Build the STATIC index from SID token sequences.

        Args:
            sid_token_sequences: List of SID token ID lists (LLM vocab IDs,
                WITHOUT EOS — EOS is appended internally).
            eos_token_id: The EOS token ID in the LLM vocabulary.
            device: Target GPU device.
        """
        self.eos_token_id = eos_token_id
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Collect all unique LLM token IDs used in SIDs
        all_llm_ids = set()
        for seq in sid_token_sequences:
            all_llm_ids.update(seq)
        all_llm_ids.add(eos_token_id)
        sorted_llm_ids = sorted(all_llm_ids)

        # Build bidirectional mapping: LLM ID <-> contiguous ID
        self._llm_to_compact = {}
        self._compact_to_llm = {}
        for compact_id, llm_id in enumerate(sorted_llm_ids):
            self._llm_to_compact[llm_id] = compact_id
            self._compact_to_llm[compact_id] = llm_id

        compact_eos = self._llm_to_compact[eos_token_id]
        self.compact_vocab_size = len(sorted_llm_ids)

        # Convert sequences to compact IDs and append EOS
        compact_seqs = []
        for seq in sid_token_sequences:
            compact_seq = [self._llm_to_compact[t] for t in seq]
            compact_seq.append(compact_eos)
            compact_seqs.append(compact_seq)

        # Pad variable-length sequences to max length with EOS
        max_len = max(len(s) for s in compact_seqs)
        for seq in compact_seqs:
            while len(seq) < max_len:
                seq.append(compact_eos)

        sid_array = np.array(compact_seqs, dtype=np.int32)
        self.sid_len = sid_array.shape[1]  # includes EOS

        # Sort for CSR builder
        sort_keys = [sid_array[:, i] for i in range(self.sid_len - 1, -1, -1)]
        sorted_sids = sid_array[np.lexsort(sort_keys)]

        # Determine dense_lookup_layers (use 2 if SID length allows, else 1)
        d_dense = min(2, self.sid_len - 1)

        # Build STATIC index
        packed_csr, indptr, layer_max_branches, start_mask, dense_mask, dense_states = (
            _build_static_index(sorted_sids, self.compact_vocab_size, dense_lookup_layers=d_dense)
        )

        self.d_dense = d_dense
        self.layer_max_branches = layer_max_branches

        # Move to GPU
        self.packed_csr = torch.tensor(packed_csr, dtype=torch.long, device=self.device)
        self.indptr = torch.tensor(indptr, dtype=torch.long, device=self.device)
        self.start_mask_compact = torch.tensor(start_mask, dtype=torch.bool, device=self.device)
        self.dense_mask = torch.tensor(dense_mask, dtype=torch.bool, device=self.device)
        self.dense_states = torch.tensor(dense_states, dtype=torch.long, device=self.device)

        # LLM token ID mapping tensors on GPU
        self._compact_to_llm_t = torch.tensor(
            [self._compact_to_llm[i] for i in range(self.compact_vocab_size)],
            dtype=torch.long, device=self.device
        )
        # Reverse mapping: LLM ID -> compact ID (sparse, use a large tensor)
        max_llm_id = max(sorted_llm_ids) + 1
        self._llm_to_compact_t = torch.full(
            (max_llm_id,), -1, dtype=torch.long, device=self.device
        )
        for llm_id, compact_id in self._llm_to_compact.items():
            self._llm_to_compact_t[llm_id] = compact_id

        # Precompute full-LLM-vocab start mask (reusable across steps)
        self._precomputed_start_mask = None  # lazily built on first use

    def get_start_mask(self, llm_vocab_size: int) -> torch.Tensor:
        """Get the level-0 mask in LLM vocab space. Shape: (llm_vocab_size,)."""
        if self._precomputed_start_mask is None or self._precomputed_start_mask.shape[0] != llm_vocab_size:
            mask = torch.full((llm_vocab_size,), float("-inf"), device=self.device)
            valid_compact = self.start_mask_compact.nonzero(as_tuple=True)[0]
            valid_llm = self._compact_to_llm_t[valid_compact]
            mask[valid_llm] = 0.0
            self._precomputed_start_mask = mask
        return self._precomputed_start_mask


class VectorizedConstrainedLogitsProcessor(LogitsProcessor):
    """STATIC-accelerated constrained logits processor.

    Replaces the per-beam Python for-loop with fully vectorized GPU
    operations using a CSR sparse matrix representation of the SID trie.
    """

    def __init__(
        self,
        static_index: STATICIndex,
        num_beams: int,
        eos_token_id: int,
    ):
        self.index = static_index
        self._num_beams = num_beams
        self.eos_token_id = eos_token_id
        self.count = 0

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        device = scores.device
        total_beams, llm_vocab_size = scores.shape
        scores = torch.nn.functional.log_softmax(scores, dim=-1)

        step = self.count
        self.count += 1

        # Step 0: broadcast start mask (identical for all beams)
        if step == 0:
            start_mask = self.index.get_start_mask(llm_vocab_size)
            return scores + start_mask.unsqueeze(0)

        # Last step (EOS only): all SID tokens generated, force EOS
        if step >= self.index.sid_len:
            mask = torch.full_like(scores, float("-inf"))
            mask[:, self.eos_token_id] = 0.0
            return scores + mask

        # Extract generated SID tokens so far and map to compact IDs
        prefix_llm = input_ids[:, -step:]  # (total_beams, step)
        prefix_compact = self.index._llm_to_compact_t[prefix_llm]  # (total_beams, step)

        # Step 1..d_dense: use dense lookup
        # dense_mask is indexed by compact TOKEN IDs, not state IDs
        if step <= self.index.d_dense:
            if self.index.d_dense == 2 and step == 1:
                # Level-0 compact tokens → dense_mask[l0] gives valid level-1 tokens
                l0 = prefix_compact[:, 0].long()
                valid_compact_mask = self.index.dense_mask[l0]  # (total_beams, V_compact)
            elif self.index.d_dense == 2 and step == 2:
                # (level-0, level-1) → use dense_states to get CSR state IDs
                l0 = prefix_compact[:, 0].long()
                l1 = prefix_compact[:, 1].long()
                flat_states = self.index.dense_states[l0, l1]
                return self._apply_csr_mask(scores, flat_states, step, device, llm_vocab_size)
            elif self.index.d_dense == 1 and step == 1:
                # d_dense=1: level-0 state = compact_token + 1, use CSR
                flat_states = (prefix_compact[:, 0] + 1).long()
                return self._apply_csr_mask(scores, flat_states, step, device, llm_vocab_size)
            else:
                l0 = prefix_compact[:, 0].long()
                valid_compact_mask = self.index.dense_mask[l0]

            # Convert compact boolean mask → LLM vocab mask
            mask = torch.full_like(scores, float("-inf"))
            valid_positions = valid_compact_mask.nonzero(as_tuple=False)  # (K, 2): [beam_idx, compact_id]
            if valid_positions.shape[0] > 0:
                beam_idx = valid_positions[:, 0]
                compact_ids = valid_positions[:, 1]
                llm_ids = self.index._compact_to_llm_t[compact_ids]
                mask[beam_idx, llm_ids] = 0.0
            return scores + mask

        # Step d_dense+1..sid_len-1: CSR sparse lookup
        # Compute state from prefix using dense_states
        if self.index.d_dense == 2:
            l0 = prefix_compact[:, 0].long()
            l1 = prefix_compact[:, 1].long()
            flat_states = self.index.dense_states[l0, l1]
        else:
            flat_states = (prefix_compact[:, 0] + 1).long()

        # Walk CSR for remaining prefix levels
        for d in range(self.index.d_dense, step):
            token_at_d = prefix_compact[:, d].long()
            flat_states = self._csr_transition(flat_states, token_at_d)

        return self._apply_csr_mask(scores, flat_states, step, device, llm_vocab_size)

    def _csr_transition(self, states, tokens):
        """Transition states through CSR given tokens. All on GPU."""
        starts = self.index.indptr[states]
        ends = self.index.indptr[states + 1]
        max_width = (ends - starts).max().item()
        if max_width == 0:
            return torch.zeros_like(states)

        offsets = torch.arange(max_width, device=states.device)
        gather_idx = (starts.unsqueeze(1) + offsets.unsqueeze(0)).clamp(
            max=self.index.packed_csr.size(0) - 1
        )
        gathered = self.index.packed_csr[gather_idx]
        child_tokens = gathered[..., 0]
        child_states = gathered[..., 1]
        actual_lens = ends - starts
        valid = offsets.unsqueeze(0) < actual_lens.unsqueeze(1)

        # Find matching token in each row
        match = (child_tokens == tokens.unsqueeze(1)) & valid
        # Use argmax to get the first match index per beam
        match_idx = match.float().argmax(dim=1)
        beam_indices = torch.arange(states.shape[0], device=states.device)
        new_states = child_states[beam_indices, match_idx]
        # If no match found, state becomes 0 (terminal)
        no_match = ~match.any(dim=1)
        new_states[no_match] = 0
        return new_states

    def _apply_csr_mask(self, scores, flat_states, step, device, llm_vocab_size):
        """Apply CSR-based masking for a given set of trie states."""
        limit = self.index.layer_max_branches[step] if step < len(self.index.layer_max_branches) else 1
        total_beams = scores.shape[0]

        starts = self.index.indptr[flat_states]
        actual_lens = self.index.indptr[flat_states + 1] - starts
        offsets = torch.arange(limit, device=device)
        gather_idx = (starts.unsqueeze(1) + offsets.unsqueeze(0)).clamp(
            max=self.index.packed_csr.size(0) - 1
        )
        gathered = self.index.packed_csr[gather_idx]
        candidate_compact = gathered[..., 0].long()
        valid_mask = offsets.unsqueeze(0) < actual_lens.unsqueeze(1)

        # Map compact IDs → LLM IDs and scatter into full-vocab mask
        candidate_llm = self.index._compact_to_llm_t[
            candidate_compact.clamp(max=self.index.compact_vocab_size - 1)
        ]

        mask = torch.full((total_beams, llm_vocab_size), float("-inf"), device=device)
        beam_indices = torch.arange(total_beams, device=device).unsqueeze(1).expand_as(candidate_llm)
        mask[beam_indices[valid_mask], candidate_llm[valid_mask]] = 0.0

        # Handle beams with no valid tokens → allow EOS
        no_valid = actual_lens == 0
        if no_valid.any():
            mask[no_valid, self.eos_token_id] = 0.0

        return scores + mask
