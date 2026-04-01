# LRA Scripts

Data generation scripts for the [Long Range Arena (LRA)](https://github.com/google-research/long-range-arena) benchmark. Each script downloads and preprocesses a specific task into PyTorch `.pt` files.

## Setup

```bash
uv sync
```

## Tasks

| Script | Output | Description |
|---|---|---|
| `generate_listops.py` | `listops.pt` | Hierarchical parsing of nested mathematical expressions |
| `generate_imdb.py` | `imdb_lra.pt` | Character-level sentiment classification on IMDb reviews |
| `generate_cifar10.py` | `cifar10_sequential_lra.pt` | Sequential pixel-by-pixel CIFAR-10 classification (len 1024) |
| `generate_acl_retrieval.py` | `acl_retrieval_lra.pt` | Document similarity matching using ACL Anthology papers |
| `generate_pathfinder.py` | `pathfinder_lra.pt` | Path detection in 32x32 synthetic images (len 1024) |
| `generate_pathx.py` | `pathx_lra.pt` | Extended path detection in 128x128 images (len 16384) |

## Usage

Run any generator directly:

```bash
python generate_listops.py
python generate_imdb.py
# etc.
```
