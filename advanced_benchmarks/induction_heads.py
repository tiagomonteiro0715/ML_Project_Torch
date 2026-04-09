# -*- coding: utf-8 -*-
"""
Induction Heads

Problem: Needed to understand the mechanism behind in-context learning.
Found that induction heads ([A][B]...[A]->[B]) are likely the primary circuit
driving ICL.
"""

import random
import torch


def generate_induction_heads(
    batch_size=64,
    seq_len=128,
    vocab_size=64,
    num_triggers=4,
    seed=42
):
    """
    Induction Head task: earlier in the sequence, pattern "A B" appears.
    Later, "A" appears again — the model must predict "B".

    This tests the two-layer attention circuit:
      Layer 1 attends to the previous token of the current query.
      Layer 2 copies the token that followed the previous occurrence.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    inputs = torch.randint(1, vocab_size, (batch_size, seq_len))
    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)  # -100 = ignore

    for b in range(batch_size):
        # Create trigger pairs (A, B) in the first half
        pair_positions = sorted(random.sample(range(0, seq_len // 2 - 1), num_triggers))
        pairs = []
        for pos in pair_positions:
            a = random.randint(1, vocab_size - 1)
            btoken = random.randint(1, vocab_size - 1)
            inputs[b, pos] = a
            inputs[b, pos + 1] = btoken
            pairs.append((a, btoken))

        # Place query "A" tokens in the second half; label is "B"
        query_positions = sorted(random.sample(
            range(seq_len // 2, seq_len - 1), num_triggers
        ))
        for i, qpos in enumerate(query_positions):
            a, btoken = pairs[i]
            inputs[b, qpos] = a
            labels[b, qpos] = btoken  # model should predict B here

    return inputs, labels


if __name__ == "__main__":
    inputs, labels = generate_induction_heads(batch_size=32, seq_len=64, num_triggers=3)
    print("=== Induction Heads ===")
    print(f"Inputs shape:  {inputs.shape}")
    print(f"Labels shape:  {labels.shape}")

    # Show the trigger positions (where label != -100)
    sample_idx = 0
    trigger_mask = labels[sample_idx] != -100
    trigger_pos = trigger_mask.nonzero().squeeze()
    print(f"Query positions: {trigger_pos.tolist()}")
    print(f"Expected tokens: {labels[sample_idx, trigger_mask].tolist()}")
