"""NanoLlama: a decoder-only transformer built from modern Llama-style primitives.

Primitives used (the "state of the art primitives that fit on a laptop" part of
the brief):
  * pre-norm residual blocks with RMSNorm          (Zhang & Sennrich, 2019)
  * rotary position embeddings                      (Su et al., 2021)
  * grouped-query attention                         (Ainslie et al., 2023)
  * SwiGLU feed-forward                             (Shazeer, 2020)
  * weight tying between the embedding and the output head
  * no biases in the linear layers

Everything runs on CPU or Apple MPS in single precision; the model is a few
million parameters, so no GPU-specific kernels are required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NanoLlamaConfig:
    vocab_size: int = 1024
    d_model: int = 192
    n_layers: int = 4
    n_heads: int = 6
    n_kv_heads: int = 2
    max_seq_len: int = 160
    dropout: float = 0.1
    rope_theta: float = 10000.0
    ffn_multiple: int = 32

    def to_dict(self) -> dict:
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * norm.type_as(x)


def build_rope_cache(seq_len: int, head_dim: int, theta: float, device) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)                 # (T, head_dim/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, n_head, T, head_dim). Rotates (even, odd) coordinate pairs."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    rotated = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return rotated.flatten(-2)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(B, n_kv, T, hd) -> (B, n_kv * n_rep, T, hd) for grouped-query attention."""
    if n_rep == 1:
        return x
    b, n_kv, t, hd = x.shape
    return x[:, :, None].expand(b, n_kv, n_rep, t, hd).reshape(b, n_kv * n_rep, t, hd)


class Attention(nn.Module):
    def __init__(self, cfg: NanoLlamaConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        assert cfg.n_heads % cfg.n_kv_heads == 0
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.dropout = cfg.dropout

        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin):
        b, t, _ = x.shape
        q = self.wq(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        k, v = repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep)

        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(b, t, -1)
        return self.resid_drop(self.wo(y))


class SwiGLU(nn.Module):
    def __init__(self, cfg: NanoLlamaConfig):
        super().__init__()
        hidden = int(8 * cfg.d_model / 3)
        hidden = cfg.ffn_multiple * math.ceil(hidden / cfg.ffn_multiple)
        self.w1 = nn.Linear(cfg.d_model, hidden, bias=False)   # gate
        self.w3 = nn.Linear(cfg.d_model, hidden, bias=False)   # value
        self.w2 = nn.Linear(hidden, cfg.d_model, bias=False)   # down
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    def __init__(self, cfg: NanoLlamaConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.ffn(self.ffn_norm(x))


class NanoLlama(nn.Module):
    def __init__(self, cfg: NanoLlamaConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight          # weight tying

        cos, sin = build_rope_cache(cfg.max_seq_len, cfg.d_model // cfg.n_heads,
                                    cfg.rope_theta, torch.device("cpu"))
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        for name, p in self.named_parameters():           # scaled residual init
            if name.endswith("wo.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                loss_mask: torch.Tensor | None = None):
        b, t = idx.shape
        assert t <= self.cfg.max_seq_len, f"sequence of {t} exceeds {self.cfg.max_seq_len}"
        cos = self.rope_cos[:t].to(idx.device)
        sin = self.rope_sin[:t].to(idx.device)

        x = self.drop(self.tok_emb(idx))
        for block in self.blocks:
            x = block(x, cos, sin)
        logits = self.lm_head(self.norm(x))

        if targets is None:
            return logits, None

        per_token = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
        ).view(b, t)
        if loss_mask is None:
            loss = per_token.mean()
        else:
            denom = loss_mask.sum().clamp(min=1)
            loss = (per_token * loss_mask).sum() / denom
        return logits, loss

    @torch.no_grad()
    def next_token_logits(self, idx: torch.Tensor) -> torch.Tensor:
        idx = idx[:, -self.cfg.max_seq_len:]
        logits, _ = self(idx)
        return logits[:, -1, :]

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.8,
                 top_k: int | None = 40, top_p: float | None = 0.95,
                 eos_id: int | None = None, greedy: bool = False) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            logits = self.next_token_logits(idx)
            if greedy or temperature <= 0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    k = min(top_k, logits.size(-1))
                    kth = torch.topk(logits, k, dim=-1).values[:, -1, None]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                if top_p is not None and 0 < top_p < 1:
                    srt, srt_idx = torch.sort(logits, descending=True, dim=-1)
                    cum = torch.softmax(srt, dim=-1).cumsum(dim=-1)
                    drop = cum - torch.softmax(srt, dim=-1) > top_p
                    srt = srt.masked_fill(drop, float("-inf"))
                    logits = torch.full_like(logits, float("-inf")).scatter(1, srt_idx, srt)
                nxt = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None and bool((nxt == eos_id).all()):
                break
            if idx.size(1) >= self.cfg.max_seq_len:
                break
        return idx
