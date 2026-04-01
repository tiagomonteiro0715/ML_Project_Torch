"""
LRA Benchmark — ListOps: Hierarchical parsing of mathematical expressions.
Can the model understand nested structure, like matching parentheses and
applying operations in the right order?

Generates: listops.pt
"""

import random
import re

import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.data import Dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------
class ListOpsDataset(Dataset):
    def __init__(self, data, max_len=2048):
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens, label = self.data[idx]

        if len(tokens) < self.max_len:
            mask = [1] * len(tokens) + [0] * (self.max_len - len(tokens))
            tokens = tokens + [0] * (self.max_len - len(tokens))
        else:
            tokens = tokens[: self.max_len]
            mask = [1] * self.max_len

        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
class ListOpsGenerator:
    def __init__(self, max_depth=10, max_args=10, seed=42, device=None):
        random.seed(seed)
        np.random.seed(seed)
        self.max_depth = max_depth
        self.max_args = max_args
        self.ops = ["MAX", "MIN", "MED", "SM"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.vocab = {"[PAD]": 0, "[": 1, "]": 2, "MAX": 3, "MIN": 4, "MED": 5, "SM": 6}
        for i in range(10):
            self.vocab[str(i)] = 7 + i
        self.vocab_size = len(self.vocab)
        self._token_pattern = re.compile(r"\[|\]|MAX|MIN|MED|SM|\d")

    def _eval(self, expr):
        expr = expr.strip()
        if expr.isdigit():
            return int(expr)
        expr = expr[1:-1].strip()
        parts = expr.split(None, 1)
        op = parts[0]
        if len(parts) == 1:
            return 0

        args, current, depth = [], [], 0
        for c in parts[1]:
            if c == "[":
                depth += 1
                current.append(c)
            elif c == "]":
                depth -= 1
                current.append(c)
            elif c == " " and depth == 0:
                if current:
                    args.append("".join(current))
                    current = []
            else:
                current.append(c)
        if current:
            args.append("".join(current))

        vals = [self._eval(arg) for arg in args]
        if not vals:
            return 0
        if op == "MAX":
            return max(vals)
        elif op == "MIN":
            return min(vals)
        elif op == "MED":
            s = sorted(vals)
            n = len(s)
            return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) // 2
        else:
            return sum(vals) % 10

    def _gen_expr(self, depth=0, curr_len=0, target_len=1000):
        if curr_len > target_len * 1.5:
            term_prob = 0.8
        elif curr_len > target_len:
            term_prob = 0.5
        elif curr_len < target_len * 0.3 and depth < self.max_depth - 2:
            term_prob = 0.1
        else:
            term_prob = 0.25

        if depth >= self.max_depth or (depth > 0 and random.random() < term_prob):
            return str(random.randint(0, 9))

        op = random.choice(self.ops)
        num_args = (
            random.randint(3, self.max_args)
            if curr_len < target_len * 0.5
            else random.randint(2, 5)
        )
        args = []
        for _ in range(num_args):
            arg = self._gen_expr(depth + 1, curr_len + len(args) * 10, target_len)
            args.append(arg)
            curr_len += len(arg)
        return f"[ {op} {' '.join(args)} ]"

    def tokenize(self, expr):
        return [self.vocab[t] for t in self._token_pattern.findall(expr)]

    def _generate_single(self, target_len, min_len, max_len):
        for _ in range(100):
            expr = self._gen_expr(target_len=target_len)
            if min_len <= len(expr) <= max_len:
                try:
                    return (self.tokenize(expr), self._eval(expr))
                except Exception:
                    pass
        return None

    def generate(self, n, min_len=500, max_len=2000, num_workers=8):
        data, target_len = [], (min_len + max_len) // 2
        pbar = tqdm(total=n, desc="Generating samples")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(self._generate_single, target_len, min_len, max_len): i
                for i in range(min(n * 2, 1000))
            }
            while len(data) < n:
                for future in as_completed(futures):
                    if len(data) >= n:
                        break
                    result = future.result()
                    if result:
                        data.append(result)
                        pbar.update(1)
                    del futures[future]
                needed = n - len(data) - len(futures)
                if needed > 0:
                    for _ in range(min(needed * 2, 100)):
                        futures[
                            executor.submit(self._generate_single, target_len, min_len, max_len)
                        ] = len(futures)
        pbar.close()
        return data[:n]


# Main
if __name__ == "__main__":
    gen = ListOpsGenerator(seed=42)

    print("Generating ListOps dataset...\n")
    print("Training set (96000 samples):")
    train_data = gen.generate(96000)

    print("\nValidation set (2000 samples):")
    val_data = gen.generate(2000)

    print("\nTest set (2000 samples):")
    test_data = gen.generate(2000)

    train_ds = ListOpsDataset(train_data)
    val_ds = ListOpsDataset(val_data)
    test_ds = ListOpsDataset(test_data)

    torch.save(
        {"train": train_ds, "val": val_ds, "test": test_ds, "vocab_size": gen.vocab_size},
        "listops.pt",
    )

    print(f"\nSaved to listops.pt")
    print(f"Vocab size: {gen.vocab_size}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")