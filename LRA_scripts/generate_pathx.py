"""
LRA Benchmark — Path-X: Extended Pathfinder with 16K sequence length.
Same as Pathfinder but much harder because the sequence is 16 times longer.
(128×128 images → 16384 sequence length)

Generates: pathx_lra.pt
"""

import random

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------
class PathfinderDataset(Dataset):
    """
    PyTorch Dataset for Pathfinder / Path-X tasks.
    Accepts tensors directly (no list conversion needed).
    """

    def __init__(self, tokens, labels, seq_len):
        if isinstance(tokens, torch.Tensor):
            self.tokens = tokens
            self.labels = labels
            self.is_tensor = True
        else:
            self.data = tokens
            self.is_tensor = False
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) if self.is_tensor else len(self.data)

    def __getitem__(self, idx):
        if self.is_tensor:
            tokens = self.tokens[idx]
            label = self.labels[idx]
        else:
            tokens, label = self.data[idx]
            tokens = torch.tensor(tokens, dtype=torch.long)
            label = torch.tensor(label, dtype=torch.long)

        if len(tokens) < self.seq_len:
            mask = torch.cat([torch.ones(len(tokens)), torch.zeros(self.seq_len - len(tokens))])
            tokens = torch.cat([tokens, torch.zeros(self.seq_len - len(tokens), dtype=torch.long)])
        else:
            tokens = tokens[: self.seq_len]
            mask = torch.ones(self.seq_len)

        return {
            "input_ids": tokens.long(),
            "attention_mask": mask.long(),
            "labels": label.long(),
        }


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
class PathfinderGenerator:
    """
    GPU-accelerated Pathfinder generator for Long Range Arena benchmark.
    Task: Binary classification — are two dots connected by a path?
    """

    def __init__(self, resolution=128, seed=42, device=None,
                 contour_length=48, num_distractor_snakes=12, snake_contrast=0.5):
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.resolution = resolution
        self.seq_len = resolution * resolution
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.contour_length = contour_length
        self.num_distractor_snakes = num_distractor_snakes
        self.snake_contrast = snake_contrast

        self.vocab_size = 257
        self.pad_token = 0
        self.num_classes = 2

    def _draw_snake_batch(self, imgs, start_positions, lengths, mark_endpoints=False):
        batch_size = imgs.shape[0]
        h, w = self.resolution, self.resolution
        device = imgs.device

        x = start_positions[:, 0].float()
        y = start_positions[:, 1].float()
        angles = torch.rand(batch_size, device=device) * 2 * np.pi

        start_x = x.clone()
        start_y = y.clone()
        max_length = lengths.max().item()

        for step in range(max_length):
            angles += torch.randn(batch_size, device=device) * 0.5
            dx = torch.cos(angles) * 2
            dy = torch.sin(angles) * 2
            x = torch.clamp(x + dx, 2, w - 3)
            y = torch.clamp(y + dy, 2, h - 3)
            active = step < lengths

            for ox in range(-1, 2):
                for oy in range(-1, 2):
                    px = (x + ox).long().clamp(0, w - 1)
                    py = (y + oy).long().clamp(0, h - 1)
                    batch_idx = torch.arange(batch_size, device=device)
                    intensity = (128 * self.snake_contrast * active.float()).long()
                    imgs[batch_idx, py, px] = torch.clamp(
                        imgs[batch_idx, py, px] + intensity, 0, 255
                    )

        if mark_endpoints:
            for pos_x, pos_y in [(start_x, start_y), (x, y)]:
                for ox in range(-2, 3):
                    for oy in range(-2, 3):
                        if ox * ox + oy * oy <= 4:
                            px = (pos_x + ox).long().clamp(0, w - 1)
                            py = (pos_y + oy).long().clamp(0, h - 1)
                            batch_idx = torch.arange(batch_size, device=device)
                            imgs[batch_idx, py, px] = 255

        return x.long(), y.long()

    def _generate_batch(self, batch_size):
        h, w = self.resolution, self.resolution
        device = self.device
        margin = max(5, self.resolution // 8)

        imgs = torch.randint(0, 50, (batch_size, h, w), device=device, dtype=torch.long)
        labels = torch.randint(0, 2, (batch_size,), device=device)

        start_x = torch.randint(margin, w - margin, (batch_size,), device=device)
        start_y = torch.randint(margin, h - margin, (batch_size,), device=device)
        start_positions = torch.stack([start_x, start_y], dim=1)

        lengths = torch.full((batch_size,), self.contour_length, device=device)
        end_x, end_y = self._draw_snake_batch(imgs, start_positions, lengths, mark_endpoints=True)

        for _ in range(self.num_distractor_snakes):
            dist_x = torch.randint(margin, w - margin, (batch_size,), device=device)
            dist_y = torch.randint(margin, h - margin, (batch_size,), device=device)
            dist_positions = torch.stack([dist_x, dist_y], dim=1)
            dist_lengths = torch.randint(
                self.contour_length // 2,
                self.contour_length + 1,
                (batch_size,),
                device=device,
            )
            self._draw_snake_batch(imgs, dist_positions, dist_lengths, mark_endpoints=False)

        not_connected = labels == 0
        if not_connected.any():
            new_x = torch.randint(margin, w - margin, (batch_size,), device=device)
            new_y = torch.randint(margin, h - margin, (batch_size,), device=device)

            for _ in range(10):
                too_close = (torch.abs(new_x - start_x) < w // 3) & (
                    torch.abs(new_y - start_y) < h // 3
                )
                if not too_close.any():
                    break
                new_x = torch.where(
                    too_close,
                    torch.randint(margin, w - margin, (batch_size,), device=device),
                    new_x,
                )
                new_y = torch.where(
                    too_close,
                    torch.randint(margin, h - margin, (batch_size,), device=device),
                    new_y,
                )

            for ox in range(-2, 3):
                for oy in range(-2, 3):
                    if ox * ox + oy * oy <= 4:
                        px = (new_x + ox).clamp(0, w - 1)
                        py = (new_y + oy).clamp(0, h - 1)
                        batch_idx = torch.arange(batch_size, device=device)
                        imgs[batch_idx[not_connected], py[not_connected], px[not_connected]] = 255

        tokens = imgs.view(batch_size, -1) + 1
        return tokens, labels

    def generate(self, n_samples, batch_size=512):
        print(
            f"Generating {n_samples} samples on {self.device} "
            f"(resolution={self.resolution}x{self.resolution})..."
        )

        all_tokens = []
        all_labels = []
        num_batches = (n_samples + batch_size - 1) // batch_size

        for i in tqdm(range(num_batches), desc="Generating batches"):
            current_batch_size = min(batch_size, n_samples - i * batch_size)
            tokens, labels = self._generate_batch(current_batch_size)
            all_tokens.append(tokens.cpu())
            all_labels.append(labels.cpu())

        all_tokens = torch.cat(all_tokens, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        return all_tokens, all_labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gen = PathfinderGenerator(
        resolution=128, seed=42, device=device,
        contour_length=48, num_distractor_snakes=12,
    )

    print(f"Using device: {gen.device}")
    print(f"Resolution: {gen.resolution}x{gen.resolution}")
    print(f"Sequence length: {gen.seq_len}")
    print(f"Vocab size: {gen.vocab_size}")
    print(f"Classes: 2 (connected / not connected)\n")

    batch_size_px = 512 if torch.cuda.is_available() else 128

    train_tokens, train_labels = gen.generate(160000, batch_size=batch_size_px)
    val_tokens, val_labels = gen.generate(20000, batch_size=batch_size_px)
    test_tokens, test_labels = gen.generate(20000, batch_size=batch_size_px)

    train_ds = PathfinderDataset(train_tokens, train_labels, seq_len=gen.seq_len)
    val_ds = PathfinderDataset(val_tokens, val_labels, seq_len=gen.seq_len)
    test_ds = PathfinderDataset(test_tokens, test_labels, seq_len=gen.seq_len)

    print("Saving dataset...")
    torch.save(
        {
            "train": train_ds,
            "val": val_ds,
            "test": test_ds,
            "vocab_size": gen.vocab_size,
            "seq_len": gen.seq_len,
            "resolution": gen.resolution,
            "num_classes": gen.num_classes,
            "task": "pathx",
        },
        "pathx_lra.pt",
    )

    print(f"\n" + "-" * 40)
    print("PATH-X SAVED")
    print("-" * 40)
    print(f"Saved to: pathx_lra.pt")
    print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"\nGPU memory cleared")