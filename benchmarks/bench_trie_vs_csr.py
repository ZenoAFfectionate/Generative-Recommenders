"""Benchmark: Dict-Trie (current) vs CSR-Vectorized (STATIC) constrained decoding.

Loads MiniOneRec's actual SID data and measures the per-step mask creation time
for both approaches across realistic batch×beam configurations.

Usage:
    cd /home/kemove/LLM_Projects/MiniOneRec
    python benchmarks/bench_trie_vs_csr.py
"""

import json
import sys
import os
import time
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "/tmp/static-constraint-decoding")

from model.LogitProcessor import SIDTrie, ConstrainedLogitsProcessor
from static_decoding.csr_utils import build_static_index

# ─────────────────────────────────────────────────────
# 1. Load actual SID data
# ─────────────────────────────────────────────────────
INDEX_PATH = "data/Amazon/index/Industrial_and_Scientific.index.json"
with open(INDEX_PATH) as f:
    sid_index = json.load(f)

# Simulate tokenizer mapping: SID string tokens → contiguous integer IDs
# In the real model these are LLM vocab IDs, but for benchmarking we
# remap to a small contiguous space to build the CSR index, then map back.
all_tokens = set()
for sid_tokens in sid_index.values():
    all_tokens.update(sid_tokens)
token_to_id = {t: i for i, t in enumerate(sorted(all_tokens))}
VOCAB_SIZE = len(token_to_id)  # ~560 SID tokens
EOS_ID = VOCAB_SIZE  # use next ID as EOS

# Build integer SID sequences (for both trie and CSR)
sid_sequences = []
for sid_tokens in sid_index.values():
    seq = [token_to_id[t] for t in sid_tokens]
    sid_sequences.append(seq)

sid_array = np.array(sid_sequences, dtype=np.int32)  # (N, 3)
N_ITEMS, SID_LEN = sid_array.shape
print(f"Loaded {N_ITEMS} items, SID length={SID_LEN}, vocab={VOCAB_SIZE} (+EOS)")

# ─────────────────────────────────────────────────────
# 2a. Build dict-based trie (current implementation)
# ─────────────────────────────────────────────────────
sid_trie = SIDTrie()
for seq in sid_sequences:
    sid_trie.insert(seq + [EOS_ID])


def trie_lookup_fn(batch_id, prefix):
    return sid_trie.get_valid_tokens(prefix)


# ─────────────────────────────────────────────────────
# 2b. Build CSR index (STATIC)
# ─────────────────────────────────────────────────────
# STATIC requires sorted SIDs
sorted_indices = np.lexsort([sid_array[:, i] for i in range(SID_LEN - 1, -1, -1)])
sorted_sids = sid_array[sorted_indices]

# Add EOS as a fourth column so the CSR handles the full trie depth
eos_col = np.full((N_ITEMS, 1), EOS_ID, dtype=np.int32)
sorted_sids_with_eos = np.hstack([sorted_sids, eos_col])

packed_csr, indptr, layer_max_branches, start_mask, dense_mask, dense_states = (
    build_static_index(sorted_sids_with_eos, vocab_size=VOCAB_SIZE + 1, dense_lookup_layers=2)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Benchmark device: {device}")

# Move CSR tensors to device
csr_packed_t = torch.tensor(packed_csr, dtype=torch.long, device=device)
csr_indptr_t = torch.tensor(indptr, dtype=torch.long, device=device)
start_mask_t = torch.tensor(start_mask, dtype=torch.bool, device=device)
dense_mask_t = torch.tensor(dense_mask, dtype=torch.bool, device=device)
dense_states_t = torch.tensor(dense_states, dtype=torch.long, device=device)

# Full-vocab mask for start (broadcast-ready)
full_start_mask = torch.full((VOCAB_SIZE + 1,), float("-inf"), device=device)
full_start_mask[start_mask_t] = 0.0

# ─────────────────────────────────────────────────────
# 3. Benchmark functions
# ─────────────────────────────────────────────────────
FULL_VOCAB = VOCAB_SIZE + 1  # scores dimension


def bench_dict_trie_step(total_beams, num_beams, step, prefixes):
    """Simulate one ConstrainedLogitsProcessor.__call__ step."""
    scores = torch.randn(total_beams, FULL_VOCAB, device=device)
    scores = torch.nn.functional.log_softmax(scores, dim=-1)

    mask = torch.full_like(scores, float("-inf"))

    # This is the hot loop we want to eliminate
    if step == 0:
        all_keys = [[] for _ in range(total_beams)]
    else:
        all_keys = prefixes[:, -step:].cpu().tolist()

    for i in range(total_beams):
        batch_id = i // num_beams
        valid = trie_lookup_fn(batch_id, all_keys[i])
        if valid:
            mask[i, valid] = 0

    scores = scores + mask
    return scores


def bench_csr_vectorized_step(total_beams, num_beams, step, prefixes):
    """Simulate one CSR-vectorized masking step (STATIC approach)."""
    scores = torch.randn(total_beams, FULL_VOCAB, device=device)
    scores = torch.nn.functional.log_softmax(scores, dim=-1)

    if step == 0:
        # Broadcast start mask to all beams
        scores = scores + full_start_mask.unsqueeze(0)
    elif step == 1:
        # Dense lookup: get valid tokens at level 1 given level-0 tokens
        level0_tokens = prefixes[:, -1].long()
        # Use dense_mask[level0_tokens] to get per-beam masks
        # level0 state IDs: token + 1 (STATIC convention)
        parent_states = level0_tokens + 1
        masks = dense_mask_t[parent_states.cpu().numpy()]
        masks_t = torch.tensor(masks, dtype=torch.bool, device=device)
        scores = torch.where(masks_t, scores, torch.tensor(float("-inf"), device=device))
    elif step == 2:
        # CSR sparse lookup for level 2
        level0_tokens = prefixes[:, -2].long()
        level1_tokens = prefixes[:, -1].long()
        # Get state IDs from dense_states table
        flat_states = dense_states_t[
            level0_tokens.cpu().numpy(), level1_tokens.cpu().numpy()
        ]
        flat_states = torch.tensor(flat_states, dtype=torch.long, device=device)
        # CSR gather
        limit = layer_max_branches[2]
        starts = csr_indptr_t[flat_states]
        offsets = torch.arange(limit, device=device)
        gather_idx = starts.unsqueeze(1) + offsets.unsqueeze(0)
        max_idx = csr_packed_t.size(0) - 1
        gather_idx = gather_idx.clamp(max=max_idx)
        gathered = csr_packed_t[gather_idx]
        candidate_tokens = gathered[..., 0]
        actual_lens = csr_indptr_t[flat_states + 1] - starts
        valid_mask = offsets.unsqueeze(0) < actual_lens.unsqueeze(1)
        # Build full vocab mask
        mask = torch.full_like(scores, float("-inf"))
        for k in range(limit):
            beam_mask = valid_mask[:, k]
            tok_ids = candidate_tokens[:, k].long().clamp(max=FULL_VOCAB - 1)
            mask[beam_mask, tok_ids[beam_mask]] = 0
        scores = scores + mask
    else:
        # Step 3: only EOS valid
        mask = torch.full_like(scores, float("-inf"))
        mask[:, EOS_ID] = 0
        scores = scores + mask

    return scores


def bench_csr_fully_vectorized_step(total_beams, num_beams, step, prefixes):
    """Fully vectorized CSR step — zero Python loops, all on GPU."""
    scores = torch.randn(total_beams, FULL_VOCAB, device=device)
    scores = torch.nn.functional.log_softmax(scores, dim=-1)

    if step == 0:
        scores = scores + full_start_mask.unsqueeze(0)
    elif step == 1:
        level0_tokens = prefixes[:, -1].long()
        parent_states = (level0_tokens + 1).to(device)
        masks = dense_mask_t[parent_states]
        scores = torch.where(masks, scores, torch.tensor(float("-inf"), device=device))
    elif step == 2:
        level0_tokens = prefixes[:, -2].long().to(device)
        level1_tokens = prefixes[:, -1].long().to(device)
        flat_states = dense_states_t[level0_tokens, level1_tokens]
        limit = layer_max_branches[2]
        starts = csr_indptr_t[flat_states]
        offsets = torch.arange(limit, device=device)
        gather_idx = (starts.unsqueeze(1) + offsets.unsqueeze(0)).clamp(
            max=csr_packed_t.size(0) - 1
        )
        gathered = csr_packed_t[gather_idx]
        candidate_tokens = gathered[..., 0].long()
        actual_lens = csr_indptr_t[flat_states + 1] - starts
        valid_mask = offsets.unsqueeze(0) < actual_lens.unsqueeze(1)
        # Scatter valid tokens into full-vocab mask — fully vectorized
        mask = torch.full((total_beams, FULL_VOCAB), float("-inf"), device=device)
        beam_indices = (
            torch.arange(total_beams, device=device)
            .unsqueeze(1)
            .expand_as(candidate_tokens)
        )
        safe_tokens = candidate_tokens.clamp(max=FULL_VOCAB - 1)
        mask[beam_indices[valid_mask], safe_tokens[valid_mask]] = 0.0
        scores = scores + mask
    else:
        mask = torch.full_like(scores, float("-inf"))
        mask[:, EOS_ID] = 0
        scores = scores + mask

    return scores


# ─────────────────────────────────────────────────────
# 4. Run benchmarks
# ─────────────────────────────────────────────────────
SCENARIOS = [
    ("Eval (B=4, beams=50)", 4, 50),
    ("RL (B=8, gen=16)", 8, 16),
    ("Large (B=16, beams=50)", 16, 50),
]

WARMUP = 10
ITERS = 100

print("\n" + "=" * 70)
print("BENCHMARK: Dict-Trie vs CSR-Vectorized Constrained Decoding")
print("=" * 70)

for name, batch_size, num_beams in SCENARIOS:
    total_beams = batch_size * num_beams
    print(f"\n{'─' * 60}")
    print(f"Scenario: {name}  (total_beams={total_beams})")
    print(f"{'─' * 60}")

    for step in range(SID_LEN + 1):  # 0, 1, 2, 3 (3=EOS)
        # Generate random prefixes for this step
        if step == 0:
            prefixes = torch.zeros(total_beams, 1, dtype=torch.long, device=device)
        else:
            # Simulate realistic prefixes from actual SID data
            rand_items = np.random.randint(0, N_ITEMS, size=total_beams)
            prefix_data = sid_array[rand_items, :step]
            prefixes = torch.tensor(prefix_data, dtype=torch.long, device=device)

        # Warm up
        for _ in range(WARMUP):
            bench_dict_trie_step(total_beams, num_beams, step, prefixes)
            bench_csr_fully_vectorized_step(total_beams, num_beams, step, prefixes)
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Benchmark dict-trie
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            bench_dict_trie_step(total_beams, num_beams, step, prefixes)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_dict = (time.perf_counter() - t0) / ITERS * 1000  # ms

        # Benchmark CSR-vectorized
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            bench_csr_fully_vectorized_step(total_beams, num_beams, step, prefixes)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_csr = (time.perf_counter() - t0) / ITERS * 1000  # ms

        speedup = t_dict / t_csr if t_csr > 0 else float("inf")
        print(
            f"  Step {step}: Dict-Trie={t_dict:8.3f}ms | CSR={t_csr:8.3f}ms | Speedup={speedup:6.1f}x"
        )

print("\n" + "=" * 70)
print("BENCHMARK COMPLETE")
print("=" * 70)
