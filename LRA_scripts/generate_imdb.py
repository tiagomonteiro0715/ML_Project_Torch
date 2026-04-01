"""
LRA Benchmark — Text: Sentiment classification on IMDb reviews.
Can the model figure out sentiment when reading raw characters instead of
words across a long document?

Generates: imdb_lra.pt
"""

import random

import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------
class IMDbDataset(Dataset):
    """PyTorch Dataset for byte-level IMDb classification."""

    def __init__(self, data, max_len=4096):
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
class IMDbGenerator:
    """
    IMDb byte-level text classification for Long Range Arena benchmark.
    - Binary classification (positive / negative sentiment)
    - Character / byte-level tokenization (vocab size 257)
    - Max sequence length: 4096 (as per LRA spec)
    """

    def __init__(self, max_len=4096, seed=42, device=None):
        random.seed(seed)
        np.random.seed(seed)
        self.max_len = max_len
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.vocab_size = 257  # 0 = PAD, 1-256 = byte values
        self.pad_token = 0

    def tokenize(self, text):
        """Convert text to byte-level tokens (1-256, 0 reserved for PAD)."""
        return [b + 1 for b in text.encode("utf-8", errors="replace")]

    def _process_single(self, example):
        text = example["text"]
        label = example["label"]  # 0 = negative, 1 = positive
        tokens = self.tokenize(text)
        return (tokens, label)

    def load_and_process(self, num_workers=8):
        """Load IMDb dataset and process to byte-level tokens."""
        print("Loading IMDb dataset from HuggingFace...")
        dataset = load_dataset("imdb")

        train_data = []
        test_data = []

        print("\nProcessing training set (25000 samples):")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self._process_single, ex) for ex in dataset["train"]]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Train"):
                train_data.append(future.result())

        print("\nProcessing test set (25000 samples):")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self._process_single, ex) for ex in dataset["test"]]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Test"):
                test_data.append(future.result())

        random.shuffle(train_data)

        val_size = 5000
        val_data = train_data[:val_size]
        train_data = train_data[val_size:]

        return train_data, val_data, test_data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gen = IMDbGenerator(max_len=4096, seed=42, device=device)
    print(f"Using device: {gen.device}")
    print(f"Vocab size: {gen.vocab_size} (byte-level)")
    print(f"Max sequence length: {gen.max_len}\n")

    train_data, val_data, test_data = gen.load_and_process(num_workers=8)

    train_ds = IMDbDataset(train_data, max_len=4096)
    val_ds = IMDbDataset(val_data, max_len=4096)
    test_ds = IMDbDataset(test_data, max_len=4096)

    torch.save(
        {
            "train": train_ds,
            "val": val_ds,
            "test": test_ds,
            "vocab_size": gen.vocab_size,
            "max_len": gen.max_len,
            "num_classes": 2,
        },
        "imdb_lra.pt",
    )

    print(f"\nSaved to imdb_lra.pt")
    print(f"Vocab size: {gen.vocab_size}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")