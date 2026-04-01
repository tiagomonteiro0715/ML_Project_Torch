"""
LRA Benchmark — Retrieval: Document similarity matching (ACL Anthology).
Can the model tell if two documents are related by comparing their full
contents?

Generates: acl_retrieval_lra.pt
"""

import random

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------
class ACLRetrievalDataset(Dataset):
    """PyTorch Dataset for ACL document retrieval."""

    def __init__(self, data, max_len=8001):  # 4000 + 1 (sep) + 4000
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
class ACLRetrievalGenerator:
    """
    ACL Anthology document retrieval for Long Range Arena benchmark.
    - Binary classification: are two papers similar (cited) or not?
    - Byte-level tokenization (vocab size 257)
    - Max sequence length: 4000 per document (8001 total for pair)
    - Similarity defined by citation relationship
    """

    def __init__(self, max_len_per_doc=4000, seed=42, device=None):
        random.seed(seed)
        np.random.seed(seed)
        self.max_len_per_doc = max_len_per_doc
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.vocab_size = 257  # 0 = PAD, 1-256 = byte values
        self.pad_token = 0
        self.sep_token = 256  # Special separator between documents

    def tokenize(self, text):
        if text is None or (isinstance(text, float) and np.isnan(text)):
            return []
        text = str(text)
        return [b + 1 for b in text.encode("utf-8", errors="replace")]

    def load_dataset(self):
        print("Loading ACL-OCL dataset from HuggingFace...")
        print("Dataset: WINGNUS/ACL-OCL")

        print("Downloading paper metadata (515 MB)...")
        parquet_path = hf_hub_download(
            repo_id="WINGNUS/ACL-OCL",
            filename="acl-publication-info.74k.v2.parquet",
            repo_type="dataset",
        )

        print("Downloading citation graph (73 MB)...")
        citations_path = hf_hub_download(
            repo_id="WINGNUS/ACL-OCL",
            filename="acl_full_citations.parquet",
            repo_type="dataset",
        )

        print("Loading parquet files...")
        df = pd.read_parquet(parquet_path)
        citations_df = pd.read_parquet(citations_path)

        print(f"Loaded {len(df)} papers")
        print(f"Loaded {len(citations_df)} citation relationships")

        return df, citations_df

    def build_citation_graph(self, df, citations_df):
        print("Building citation-based similarity pairs...")

        df_valid = df[
            (df["abstract"].notna())
            & (df["abstract"].str.len() > 100)
            & (df["full_text"].notna())
            & (df["full_text"].str.len() > 200)
        ].copy()

        print(f"Papers with valid text: {len(df_valid)}")

        papers = df_valid[["acl_id", "abstract", "full_text", "title"]].reset_index(drop=True)
        paper_id_to_idx = {pid: idx for idx, pid in enumerate(papers["acl_id"].tolist())}

        print(f"Citation graph columns: {citations_df.columns.tolist()}")

        return papers, paper_id_to_idx, citations_df

    def create_pairs(self, papers, paper_id_to_idx, citations_df, n_pairs, positive_ratio=0.5):
        print(f"Creating {n_pairs} document pairs...")

        pairs = []
        n_positive = int(n_pairs * positive_ratio)
        n_negative = n_pairs - n_positive

        papers_list = papers.to_dict("records")
        n_papers = len(papers_list)
        valid_acl_ids = set(papers["acl_id"].tolist())

        citation_pairs = set()
        try:
            for col_pair in [
                ("citing", "cited"),
                ("source", "target"),
                ("paper_id", "cited_paper_id"),
                ("from", "to"),
            ]:
                if col_pair[0] in citations_df.columns and col_pair[1] in citations_df.columns:
                    for _, row in citations_df.iterrows():
                        citing = str(row[col_pair[0]])
                        cited = str(row[col_pair[1]])
                        if citing in valid_acl_ids and cited in valid_acl_ids:
                            citation_pairs.add((citing, cited))
                    break

            if len(citation_pairs) == 0 and len(citations_df.columns) >= 2:
                col1, col2 = citations_df.columns[0], citations_df.columns[1]
                for _, row in citations_df.head(10000).iterrows():
                    citing = str(row[col1])
                    cited = str(row[col2])
                    if citing in valid_acl_ids and cited in valid_acl_ids:
                        citation_pairs.add((citing, cited))
        except Exception as e:
            print(f"Warning: Could not parse citations: {e}")

        print(f"Found {len(citation_pairs)} valid citation pairs")

        print(f"  Creating {n_positive} positive pairs...")
        citation_list = list(citation_pairs)

        for i in tqdm(range(n_positive), desc="Positive pairs"):
            if citation_list and i < len(citation_list):
                citing_id, cited_id = citation_list[i % len(citation_list)]
                idx1 = paper_id_to_idx[citing_id]
                idx2 = paper_id_to_idx[cited_id]
            else:
                idx1 = random.randint(0, n_papers - 50)
                offset = random.randint(1, min(49, n_papers - idx1 - 1))
                idx2 = idx1 + offset

            doc1_text = (
                str(papers_list[idx1].get("abstract", ""))
                + " "
                + str(papers_list[idx1].get("full_text", ""))[:2000]
            )
            doc2_text = (
                str(papers_list[idx2].get("abstract", ""))
                + " "
                + str(papers_list[idx2].get("full_text", ""))[:2000]
            )
            pairs.append((doc1_text, doc2_text, 1))

        print(f"  Creating {n_negative} negative pairs...")
        for _ in tqdm(range(n_negative), desc="Negative pairs"):
            idx1 = random.randint(0, n_papers - 1)
            idx2 = (idx1 + random.randint(1000, n_papers - 1)) % n_papers

            doc1_text = (
                str(papers_list[idx1].get("abstract", ""))
                + " "
                + str(papers_list[idx1].get("full_text", ""))[:2000]
            )
            doc2_text = (
                str(papers_list[idx2].get("abstract", ""))
                + " "
                + str(papers_list[idx2].get("full_text", ""))[:2000]
            )
            pairs.append((doc1_text, doc2_text, 0))

        random.shuffle(pairs)
        return pairs

    def process_pairs(self, pairs):
        print("Tokenizing document pairs...")
        processed = []

        for doc1, doc2, label in tqdm(pairs, desc="Tokenizing"):
            tokens1 = self.tokenize(doc1)[: self.max_len_per_doc]
            tokens2 = self.tokenize(doc2)[: self.max_len_per_doc]
            combined = tokens1 + [self.sep_token] + tokens2
            processed.append((combined, label))

        return processed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gen = ACLRetrievalGenerator(max_len_per_doc=4000, seed=42, device=device)
    print(f"Using device: {gen.device}")
    print(f"Vocab size: {gen.vocab_size} (byte-level)")
    print(f"Max length per document: {gen.max_len_per_doc}")
    print(f"Total max sequence length: {gen.max_len_per_doc * 2 + 1}\n")

    df, citations_df = gen.load_dataset()
    papers, paper_id_to_idx, citations_df = gen.build_citation_graph(df, citations_df)

    print("\n" + "=" * 50)
    print("Creating dataset splits...")
    print("=" * 50)

    train_pairs = gen.create_pairs(papers, paper_id_to_idx, citations_df, n_pairs=20000)
    val_pairs = gen.create_pairs(papers, paper_id_to_idx, citations_df, n_pairs=2000)
    test_pairs = gen.create_pairs(papers, paper_id_to_idx, citations_df, n_pairs=2000)

    train_data = gen.process_pairs(train_pairs)
    val_data = gen.process_pairs(val_pairs)
    test_data = gen.process_pairs(test_pairs)

    max_seq_len = gen.max_len_per_doc * 2 + 1
    train_ds = ACLRetrievalDataset(train_data, max_len=max_seq_len)
    val_ds = ACLRetrievalDataset(val_data, max_len=max_seq_len)
    test_ds = ACLRetrievalDataset(test_data, max_len=max_seq_len)

    torch.save(
        {
            "train": train_ds,
            "val": val_ds,
            "test": test_ds,
            "vocab_size": gen.vocab_size,
            "max_len": max_seq_len,
            "num_classes": 2,
            "task": "retrieval",
        },
        "acl_retrieval_lra.pt",
    )

    print(f"\n" + "=" * 50)
    print("DATASET SAVED SUCCESSFULLY")
    print("=" * 50)
    print(f"Saved to: acl_retrieval_lra.pt")
    print(f"Vocab size: {gen.vocab_size}")
    print(f"Max sequence length: {max_seq_len}")
    print(f"Number of classes: 2 (similar/dissimilar)")
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_ds):,} pairs")
    print(f"  Val:   {len(val_ds):,} pairs")
    print(f"  Test:  {len(test_ds):,} pairs")

    sample = train_ds[0]
    print(f"\nSample verification:")
    print(f"  Input shape: {sample['input_ids'].shape}")
    print(f"  Mask shape:  {sample['attention_mask'].shape}")
    print(f"  Label:       {sample['labels'].item()}")