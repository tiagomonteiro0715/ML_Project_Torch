# -*- coding: utf-8 -*-
"""
BASED Recall (SQuAD / SWDE / FDA)

Problem: Common zero-shot benchmarks use extremely short text and don't
stress-test recall capabilities. BASED curated real-world recall-intensive
tasks: information extraction from FDA documents, structured web data (SWDE),
and reading comprehension (SQuAD).

Setup for SWDE/FDA (run manually):
    git clone https://github.com/HazyResearch/based.git
    # FDA and SWDE data loaders are inside based/benchmarks/
"""

from datasets import load_dataset


if __name__ == "__main__":
    squad = load_dataset("rajpurkar/squad", split="validation")
    print("SQuAD:", squad)
