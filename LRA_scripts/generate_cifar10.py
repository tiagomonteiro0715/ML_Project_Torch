"""
LRA Benchmark — Image: Sequential CIFAR-10 classification (pixels as sequence).
Can the model recognize an image when pixels are fed one at a time in a long
line?

Generates: cifar10_sequential_lra.pt
"""

import random

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------
class CIFAR10SequentialDataset(Dataset):
    """PyTorch Dataset for sequential CIFAR-10 classification."""

    def __init__(self, data, seq_len=1024):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens, label = self.data[idx]

        if len(tokens) < self.seq_len:
            mask = [1] * len(tokens) + [0] * (self.seq_len - len(tokens))
            tokens = tokens + [0] * (self.seq_len - len(tokens))
        else:
            tokens = tokens[: self.seq_len]
            mask = [1] * self.seq_len

        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
class CIFAR10SequentialGenerator:
    """
    CIFAR-10 sequential image classification for Long Range Arena benchmark.
    - 10-class classification
    - Images flattened to 1D sequence: grayscale 32×32 = 1024 tokens
    - Vocab size: 257 (pixel values 0-255 shifted by 1, 0 = PAD)
    """

    def __init__(self, grayscale=True, seed=42, device=None):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.grayscale = grayscale
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.vocab_size = 257
        self.pad_token = 0
        self.seq_len = 1024 if grayscale else 3072
        self.num_classes = 10

        self.class_names = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        ]

    def load_dataset(self):
        print("Loading CIFAR-10 dataset from HuggingFace...")
        dataset = load_dataset("cifar10")
        print(f"Train samples: {len(dataset['train'])}")
        print(f"Test samples: {len(dataset['test'])}")
        return dataset

    def process_image(self, image):
        img_array = np.array(image)

        if self.grayscale:
            if len(img_array.shape) == 3:
                img_array = np.dot(img_array[..., :3], [0.299, 0.587, 0.114])
            img_array = img_array.astype(np.uint8)

        flat = img_array.flatten()
        tokens = [int(p) + 1 for p in flat]
        return tokens

    def process_dataset(self, dataset, split="train", num_workers=8):
        print(f"\nProcessing {split} split...")

        data = []
        examples = dataset[split]

        for i in tqdm(range(len(examples)), desc=f"Processing {split}"):
            image = examples[i]["img"]
            label = examples[i]["label"]
            tokens = self.process_image(image)
            data.append((tokens, label))

        return data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gen = CIFAR10SequentialGenerator(grayscale=True, seed=42, device=device)
    print(f"Using device: {gen.device}")
    print(f"Vocab size: {gen.vocab_size} (pixel values)")
    print(f"Sequence length: {gen.seq_len} ({'grayscale 32x32' if gen.grayscale else 'color 32x32x3'})")
    print(f"Number of classes: {gen.num_classes}")
    print(f"Classes: {gen.class_names}\n")

    dataset = gen.load_dataset()

    train_data = gen.process_dataset(dataset, split="train")
    test_data = gen.process_dataset(dataset, split="test")

    random.shuffle(train_data)
    val_size = 5000
    val_data = train_data[:val_size]
    train_data = train_data[val_size:]

    print(f"\nSplit sizes after validation split:")
    print(f"  Train: {len(train_data)}")
    print(f"  Val:   {len(val_data)}")
    print(f"  Test:  {len(test_data)}")

    train_ds = CIFAR10SequentialDataset(train_data, seq_len=gen.seq_len)
    val_ds = CIFAR10SequentialDataset(val_data, seq_len=gen.seq_len)
    test_ds = CIFAR10SequentialDataset(test_data, seq_len=gen.seq_len)

    torch.save(
        {
            "train": train_ds,
            "val": val_ds,
            "test": test_ds,
            "vocab_size": gen.vocab_size,
            "seq_len": gen.seq_len,
            "num_classes": gen.num_classes,
            "class_names": gen.class_names,
            "grayscale": gen.grayscale,
            "task": "image_classification",
        },
        "cifar10_sequential_lra.pt",
    )

    print(f"\n" + "=" * 50)
    print("DATASET SAVED SUCCESSFULLY")
    print("=" * 50)
    print(f"Saved to: cifar10_sequential_lra.pt")
    print(f"Vocab size: {gen.vocab_size}")
    print(f"Sequence length: {gen.seq_len}")
    print(f"Number of classes: {gen.num_classes}")
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_ds):,} images")
    print(f"  Val:   {len(val_ds):,} images")
    print(f"  Test:  {len(test_ds):,} images")