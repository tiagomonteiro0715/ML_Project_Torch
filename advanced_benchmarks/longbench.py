# -*- coding: utf-8 -*-
"""
LongBench

Problem: Needed problems challenging enough that even human experts with search
tools cannot answer correctly in a short time. The best model achieved only
50.1% accuracy.
"""

from datasets import load_dataset


if __name__ == "__main__":
    longbench_v2 = load_dataset("THUDM/LongBench-v2", split="train")
    print("LongBench v2:", longbench_v2)
    print("Sample:", longbench_v2[0].keys())
