# ML_Project_Torch

Benchmarking **CMS-augmented Mamba** against plain **Mamba** on the Long Range Arena (LRA).

Mamba is a selective state space model that achieves linear-time sequence modeling. This project augments it with a **Contextual Memory System (CMS)** an external memory bank with content-based addressing to improve long-range dependency handling, and measures the impact across all six LRA tasks.

## Setup

This project runs on Google Colab Pro (TPU / High-RAM runtime recommended for Pathfinder and Path-X).

### 1. Install packages

```python
!pip install uv
!uv pip install datasets huggingface_hub lightning numpy pandas pyarrow torch tqdm wandb
```

### 2. Import libraries

```python
import pickle, random, sys, re
from concurrent.futures import ThreadPoolExecutor, as_completed

import lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from datasets import load_dataset
from tqdm import tqdm
```

### 3. Verify device

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
```

## Notebook Structure

| Section | Contents |
|---|---|
| 1 | Installing Python packages |
| 2 | Importing Python libraries |
| 3 | Defining all AI models to run on the benchmarks |
| 4 | Long Range Arena benchmark implemented from scratch |
| 5 | Running all AI models on the benchmarks |

## LRA Tasks

| Task | Sequence Length | Type |
|---|---|---|
| ListOps | 2,048 | Classification |
| IMDb | 4,096 | Classification |
| ACL Retrieval | 4,096 x 2 | Retrieval |
| Sequential CIFAR-10 | 1,024 | Classification |
| Pathfinder | 16,384 | Classification |
| Path-X | 16,384 | Classification |

LRA paper: https://arxiv.org/abs/2011.04006
