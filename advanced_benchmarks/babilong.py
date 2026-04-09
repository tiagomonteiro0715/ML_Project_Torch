# -*- coding: utf-8 -*-
"""
BABILong

Problem: Existing evaluation methods failed to comprehensively assess models
on long contexts. Standard needle-in-a-haystack was too simple and being gamed
by training on similar tasks. BABILong embeds 20 reasoning tasks into texts
up to 10M tokens.

Configs are separate axes:
  By length: 0k, 1k, 2k, 4k, 8k, 16k, 32k, 64k, 128k, 256k, 512k, 1M, 10M
  By task:   qa1, qa2, qa3, qa4, qa5, qa6, qa7, qa8, qa9, qa10
  (130 combinations possible)
"""

from datasets import load_dataset


if __name__ == "__main__":
    # Load by length
    babilong_0k = load_dataset("RMT-team/babilong", "0k")
    print("BABILong (0k):", babilong_0k)

    # Load by task
    babilong_qa1 = load_dataset("RMT-team/babilong", "qa1")
    print("BABILong (qa1):", babilong_qa1)

    # Example: longer context
    babilong_32k = load_dataset("RMT-team/babilong", "32k")
    print("BABILong (32k):", babilong_32k)
