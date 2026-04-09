# -*- coding: utf-8 -*-
"""
MQAR — Multi-Query Associative Recall

Problem: Prior synthetic recall tests used a single query at a fixed position
with tiny vocabularies — unrealistic compared to real language. MQAR requires
multiple key-value lookups per sequence.
"""

import random
import torch


def generate_mqar(
    batch_size=64,
    seq_len=128,
    vocab_size=8192,
    num_kv_pairs=4,
    seed=42
):
    """
    Generates MQAR sequences:
    [k1, v1, k2, v2, ..., padding..., q1, q2, ...] -> [v1, v2, ...]
    Keys and queries come from a reserved range; values from the rest.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    key_vocab_start = vocab_size  # keys live above normal vocab
    inputs = torch.zeros(batch_size, seq_len, dtype=torch.long)
    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)  # -100 = ignore

    for b in range(batch_size):
        # Generate unique key-value pairs
        keys = random.sample(range(key_vocab_start, key_vocab_start + 256), num_kv_pairs)
        values = random.sample(range(1, vocab_size), num_kv_pairs)

        # Place KV pairs at the start
        for i, (k, v) in enumerate(zip(keys, values)):
            inputs[b, 2 * i] = k
            inputs[b, 2 * i + 1] = v

        # Fill middle with random padding tokens
        kv_end = 2 * num_kv_pairs
        query_start = seq_len - num_kv_pairs
        inputs[b, kv_end:query_start] = torch.randint(1, vocab_size, (query_start - kv_end,))

        # Place queries at the end, labels are the expected values
        for i, (k, v) in enumerate(zip(keys, values)):
            inputs[b, query_start + i] = k
            labels[b, query_start + i] = v

    return inputs, labels


if __name__ == "__main__":
    inputs, labels = generate_mqar(batch_size=32, seq_len=64, num_kv_pairs=4)
    print(f"Inputs shape:  {inputs.shape}")
    print(f"Labels shape:  {labels.shape}")
    print(f"Sample input:  {inputs[0, :12]}")  # first 4 KV pairs + some padding
    print(f"Sample labels: {labels[0, -4:]}")  # expected answers for 4 queries
