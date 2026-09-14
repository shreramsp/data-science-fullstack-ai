"""Corpus loading, chat formatting and batching for NanoLlama."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from tokenizer import WordTokenizer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_split(split: str) -> list[dict]:
    path = DATA_DIR / f"sft_{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing -- run `python data/build_corpus.py` first."
        )
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_pretrain_text() -> str:
    path = DATA_DIR / "pretrain.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing -- run `python data/build_corpus.py` first."
        )
    return path.read_text()


def build_tokenizer(train_rows: list[dict], pretrain_text: str) -> WordTokenizer:
    """Fit the vocabulary on training data only (no val/test text is seen)."""
    texts = [r["prompt"] for r in train_rows] + [r["response"] for r in train_rows]
    texts.append(pretrain_text)
    return WordTokenizer.fit(texts)


def encode_example(tok: WordTokenizer, prompt: str, response: str) -> tuple[list[int], int]:
    """Return the full chat sequence and the index where the response starts."""
    ids = [tok.bos_id, tok.user_id] + tok.encode(prompt) + [tok.assistant_id]
    start = len(ids)
    ids += tok.encode(response) + [tok.eos_id]
    return ids, start


def encode_prompt(tok: WordTokenizer, prompt: str) -> list[int]:
    return [tok.bos_id, tok.user_id] + tok.encode(prompt) + [tok.assistant_id]


class SFTDataset:
    """Chat-formatted examples with the loss masked to response tokens."""

    def __init__(self, rows: list[dict], tok: WordTokenizer, max_seq_len: int):
        self.rows = rows
        self.tok = tok
        self.max_seq_len = max_seq_len
        self.encoded: list[tuple[list[int], int]] = []
        self.truncated = 0
        for r in rows:
            ids, start = encode_example(tok, r["prompt"], r["response"])
            if len(ids) > max_seq_len + 1:
                ids = ids[: max_seq_len + 1]
                self.truncated += 1
            self.encoded.append((ids, start))

    def __len__(self) -> int:
        return len(self.encoded)

    @property
    def max_length(self) -> int:
        return max(len(ids) for ids, _ in self.encoded)

    def batch(self, indices: list[int], device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        chunk = [self.encoded[i] for i in indices]
        width = max(len(ids) for ids, _ in chunk)
        pad = self.tok.pad_id
        x = torch.full((len(chunk), width - 1), pad, dtype=torch.long)
        y = torch.full((len(chunk), width - 1), pad, dtype=torch.long)
        m = torch.zeros((len(chunk), width - 1), dtype=torch.float)
        for row, (ids, start) in enumerate(chunk):
            n = len(ids) - 1
            x[row, :n] = torch.tensor(ids[:-1])
            y[row, :n] = torch.tensor(ids[1:])
            # target position j predicts ids[j + 1]; supervise the response only
            m[row, max(start - 1, 0):n] = 1.0
        return x.to(device), y.to(device), m.to(device)

    def iter_batches(self, batch_size: int, generator: torch.Generator, device,
                     shuffle: bool = True):
        order = (torch.randperm(len(self), generator=generator).tolist()
                 if shuffle else list(range(len(self))))
        for i in range(0, len(order), batch_size):
            yield self.batch(order[i:i + batch_size], device)


class PretrainStream:
    """One long token stream; stage-1 samples random blocks from it."""

    def __init__(self, text: str, tok: WordTokenizer, block_size: int, val_fraction: float = 0.1):
        ids = []
        for line in text.splitlines():
            if line.strip():
                ids += [tok.bos_id] + tok.encode(line) + [tok.eos_id]
        data = torch.tensor(ids, dtype=torch.long)
        split = int(len(data) * (1 - val_fraction))
        self.train, self.val = data[:split], data[split:]
        self.block_size = block_size

    def batch(self, split: str, batch_size: int, generator: torch.Generator, device):
        data = self.train if split == "train" else self.val
        hi = len(data) - self.block_size - 1
        ix = torch.randint(hi, (batch_size,), generator=generator)
        x = torch.stack([data[i:i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + 1 + self.block_size] for i in ix])
        return x.to(device), y.to(device)
