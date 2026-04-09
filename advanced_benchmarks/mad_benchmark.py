# -*- coding: utf-8 -*-
"""
MAD Benchmark (6 sub-tasks)

Problem: Architecture design was too expensive; needed cheap proxy tasks
predictive of scaling laws.
"""

import random
import torch


def mad_in_context_recall(batch_size=32, seq_len=128, vocab_size=64, num_kv=8, seed=42):
    """Exact key-value recall from context."""
    random.seed(seed); torch.manual_seed(seed)
    inputs = torch.randint(1, vocab_size, (batch_size, seq_len))
    labels = torch.full((batch_size, seq_len), -100)
    for b in range(batch_size):
        keys = random.sample(range(vocab_size, vocab_size + 128), num_kv)
        vals = [random.randint(1, vocab_size - 1) for _ in range(num_kv)]
        for i, (k, v) in enumerate(zip(keys, vals)):
            inputs[b, 2*i] = k
            inputs[b, 2*i+1] = v
        for i, (k, v) in enumerate(zip(keys, vals)):
            qpos = seq_len - num_kv + i
            inputs[b, qpos] = k
            labels[b, qpos] = v
    return inputs, labels


def mad_fuzzy_recall(batch_size=32, seq_len=128, vocab_size=64, num_kv=8, noise_frac=0.3, seed=42):
    """Recall with noisy/corrupted keys at query time."""
    random.seed(seed); torch.manual_seed(seed)
    inputs = torch.randint(1, vocab_size, (batch_size, seq_len))
    labels = torch.full((batch_size, seq_len), -100)
    for b in range(batch_size):
        keys = [random.randint(vocab_size, vocab_size + 127) for _ in range(num_kv)]
        vals = [random.randint(1, vocab_size - 1) for _ in range(num_kv)]
        for i, (k, v) in enumerate(zip(keys, vals)):
            inputs[b, 2*i] = k
            inputs[b, 2*i+1] = v
        for i, (k, v) in enumerate(zip(keys, vals)):
            qpos = seq_len - num_kv + i
            # Corrupt the key slightly
            noisy_key = k + (random.randint(-3, 3) if random.random() < noise_frac else 0)
            inputs[b, qpos] = max(1, noisy_key)
            labels[b, qpos] = v
    return inputs, labels


def mad_noisy_recall(batch_size=32, seq_len=128, vocab_size=64, num_kv=8, seed=42):
    """Recall with extra distractor KV pairs injected."""
    random.seed(seed); torch.manual_seed(seed)
    inputs = torch.randint(1, vocab_size, (batch_size, seq_len))
    labels = torch.full((batch_size, seq_len), -100)
    for b in range(batch_size):
        keys = random.sample(range(vocab_size, vocab_size + 128), num_kv)
        vals = [random.randint(1, vocab_size - 1) for _ in range(num_kv)]
        # Place real pairs
        for i, (k, v) in enumerate(zip(keys, vals)):
            inputs[b, 2*i] = k
            inputs[b, 2*i+1] = v
        # Inject distractors in the middle
        mid = 2 * num_kv
        for j in range(mid, mid + num_kv):
            inputs[b, j] = random.randint(vocab_size + 128, vocab_size + 255)
            inputs[b, j] = random.randint(1, vocab_size - 1)
        # Queries
        for i, (k, v) in enumerate(zip(keys, vals)):
            qpos = seq_len - num_kv + i
            inputs[b, qpos] = k
            labels[b, qpos] = v
    return inputs, labels


def mad_selective_copying(batch_size=32, seq_len=128, vocab_size=16, num_copy=8, seed=42):
    """Copy only marked tokens from random positions."""
    random.seed(seed); torch.manual_seed(seed)
    blank = 0; marker = vocab_size
    inputs = torch.full((batch_size, seq_len), blank)
    labels = torch.zeros(batch_size, num_copy, dtype=torch.long)
    for b in range(batch_size):
        positions = sorted(random.sample(range(seq_len - 1), num_copy))
        tokens = torch.randint(1, vocab_size, (num_copy,))
        for i, pos in enumerate(positions):
            inputs[b, pos] = tokens[i]
            labels[b, i] = tokens[i]
        inputs[b, -1] = marker
    return inputs, labels


def mad_compression(batch_size=32, seq_len=128, vocab_size=32, pattern_len=8, repeats=4, seed=42):
    """Detect and compress a repeated pattern in the sequence."""
    random.seed(seed); torch.manual_seed(seed)
    inputs = torch.randint(1, vocab_size, (batch_size, seq_len))
    labels = torch.zeros(batch_size, pattern_len, dtype=torch.long)
    for b in range(batch_size):
        pattern = torch.randint(1, vocab_size, (pattern_len,))
        for r in range(repeats):
            start = r * pattern_len
            if start + pattern_len <= seq_len:
                inputs[b, start:start+pattern_len] = pattern
        labels[b] = pattern  # model should output the underlying pattern
    return inputs, labels


def mad_memorization(batch_size=32, seq_len=128, vocab_size=64, seed=42):
    """Memorize and reproduce the entire input sequence."""
    torch.manual_seed(seed)
    inputs = torch.randint(1, vocab_size, (batch_size, seq_len))
    labels = inputs.clone()
    return inputs, labels


if __name__ == "__main__":
    tasks = {
        "In-Context Recall": mad_in_context_recall(),
        "Fuzzy Recall":      mad_fuzzy_recall(),
        "Noisy Recall":      mad_noisy_recall(),
        "Selective Copying":  mad_selective_copying(),
        "Compression":       mad_compression(),
        "Memorization":      mad_memorization(),
    }

    for name, (inp, lab) in tasks.items():
        print(f"{name:20s} | inputs: {inp.shape}  labels: {lab.shape}")
