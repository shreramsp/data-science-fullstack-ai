"""Word-level tokenizer for NanoLlama.

A BPE tokenizer would be the state-of-the-art choice, but on a corpus this
small a word-level vocabulary keeps the sequences short and the model's output
grammatical -- which matters more here than subword coverage. Unknown words map
to <unk>, so the chatbot degrades gracefully on out-of-domain input instead of
emitting garbage.

The vocabulary is fitted on the TRAINING split only.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

PAD, UNK, BOS, EOS, USER, ASSISTANT = "<pad>", "<unk>", "<bos>", "<eos>", "<user>", "<assistant>"
SPECIALS = [PAD, UNK, BOS, EOS, USER, ASSISTANT]
# Structural tokens are stripped when decoding; <unk> is deliberately NOT in this
# set, so an unknown token stays visible in the output instead of vanishing.
STRUCTURAL = {PAD, BOS, EOS, USER, ASSISTANT}

_WORD_RE = re.compile(r"[a-z0-9]+|[^\sa-z0-9]")


class WordTokenizer:
    def __init__(self, itos: list[str]):
        self.itos = itos
        self.stoi = {t: i for i, t in enumerate(itos)}
        self.pad_id = self.stoi[PAD]
        self.unk_id = self.stoi[UNK]
        self.bos_id = self.stoi[BOS]
        self.eos_id = self.stoi[EOS]
        self.user_id = self.stoi[USER]
        self.assistant_id = self.stoi[ASSISTANT]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @staticmethod
    def words(text: str) -> list[str]:
        return _WORD_RE.findall(text.lower())

    @classmethod
    def fit(cls, texts: list[str], min_freq: int = 1) -> "WordTokenizer":
        counts: dict[str, int] = {}
        for t in texts:
            for w in cls.words(t):
                counts[w] = counts.get(w, 0) + 1
        vocab = sorted(w for w, c in counts.items() if c >= min_freq)
        return cls(SPECIALS + vocab)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(w, self.unk_id) for w in self.words(text)]

    def decode(self, ids: list[int]) -> str:
        """Join tokens back into readable text, closing up sentence punctuation."""
        out: list[str] = []
        for i in ids:
            tok = self.itos[i] if 0 <= i < len(self.itos) else UNK
            if tok in STRUCTURAL:
                continue
            if out and tok in ",.?!;:":
                out[-1] += tok
            else:
                out.append(tok)
        return " ".join(out).strip()

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"itos": self.itos}, indent=0))

    @classmethod
    def load(cls, path: Path) -> "WordTokenizer":
        return cls(json.loads(path.read_text())["itos"])
