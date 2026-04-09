# -*- coding: utf-8 -*-
"""
Selective Copying

Problem: Standard copying tasks could be solved with fixed convolution kernels
using positional tricks. Randomizing token positions forces content-aware selection.
"""

import random
import torch


def generate_selective_copying(
    batch_size=64,
    seq_len=128,
    num_tokens_to_copy=8,
    vocab_size=16,
    blank_token=0,
    marker_token=None,
    seed=42
):
    """
    Selective Copying: a sequence has `num_tokens_to_copy` meaningful tokens
    scattered at random positions among blank tokens. The model must output
    only the meaningful tokens in order.

    Unlike fixed-interval copying, positions are random — so the model must
    use content-aware (selective) reasoning, not just fixed offsets.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if marker_token is None:
        marker_token = vocab_size  # special token outside vocab

    inputs = torch.full((batch_size, seq_len), blank_token, dtype=torch.long)
    labels = torch.zeros(batch_size, num_tokens_to_copy, dtype=torch.long)

    for b in range(batch_size):
        # Pick random positions for the meaningful tokens
        positions = sorted(random.sample(range(seq_len - 1), num_tokens_to_copy))
        tokens = torch.randint(1, vocab_size, (num_tokens_to_copy,))

        for i, pos in enumerate(positions):
            inputs[b, pos] = tokens[i]
            labels[b, i] = tokens[i]

        # Append a marker token at the end to signal "now reproduce"
        inputs[b, -1] = marker_token

    return inputs, labels


if __name__ == "__main__":
    inputs, labels = generate_selective_copying(batch_size=32, seq_len=64, num_tokens_to_copy=6)
    print("=== Selective Copying ===")
    print(f"Inputs shape:  {inputs.shape}")
    print(f"Labels shape:  {labels.shape}")
    print(f"Sample input:  {inputs[0]}")
    print(f"Expected copy: {labels[0]}")
