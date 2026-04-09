# -*- coding: utf-8 -*-
"""
State Tracking — Parity / Permutation

Problem: Even the simplest state-tracking task — computing parity of a bit
sequence — cannot be solved by current linear RNNs, exposing fundamental
expressivity limits of SSMs and transformers on sequential computation.
"""

import random


def generate_parity_data(n_samples=1000, seq_len=50):
    data = []
    for _ in range(n_samples):
        bits = [random.randint(0, 1) for _ in range(seq_len)]
        label = sum(bits) % 2
        data.append((bits, label))
    return data


if __name__ == "__main__":
    parity_data = generate_parity_data()
    print(f"Parity samples: {len(parity_data)}, first: {parity_data[0]}")
