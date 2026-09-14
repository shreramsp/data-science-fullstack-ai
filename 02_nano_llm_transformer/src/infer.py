"""Loading and generation helpers shared by the CLI and the Streamlit app."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import encode_prompt
from model import NanoLlama, NanoLlamaConfig
from tokenizer import WordTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
CKPT = MODELS / "nanollama.pt"
TOKENIZER = MODELS / "tokenizer.json"


def artifacts_ready() -> bool:
    return CKPT.exists() and TOKENIZER.exists()


def load(device: str = "cpu") -> tuple[NanoLlama, WordTokenizer]:
    if not artifacts_ready():
        raise FileNotFoundError(
            "No trained checkpoint. Run `python data/build_corpus.py` then "
            "`python src/train.py`."
        )
    blob = torch.load(CKPT, map_location=device, weights_only=True)
    cfg = NanoLlamaConfig(**blob["config"])
    model = NanoLlama(cfg)
    model.load_state_dict(blob["state_dict"])
    model.to(device).eval()
    return model, WordTokenizer.load(TOKENIZER)


def load_metrics() -> dict | None:
    path = MODELS / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


@torch.no_grad()
def chat(model: NanoLlama, tok: WordTokenizer, prompt: str, max_new_tokens: int = 64,
         temperature: float = 0.7, top_k: int = 40, top_p: float = 0.95,
         greedy: bool = False, seed: int | None = None, device: str = "cpu") -> dict:
    """Generate one assistant turn and return the text plus a per-step trace."""
    if seed is not None:
        torch.manual_seed(seed)
    ids = encode_prompt(tok, prompt)
    x = torch.tensor([ids], device=device)
    n_prompt = x.size(1)
    unk_in_prompt = sum(1 for i in ids if i == tok.unk_id)

    trace: list[dict] = []
    for _ in range(max_new_tokens):
        logits = model.next_token_logits(x)[0]
        probs_full = torch.softmax(logits, dim=-1)
        entropy = float(-(probs_full * torch.log(probs_full.clamp_min(1e-12))).sum())

        if greedy or temperature <= 0:
            nxt = int(logits.argmax())
        else:
            scaled = logits / temperature
            if top_k:
                k = min(top_k, scaled.numel())
                kth = torch.topk(scaled, k).values[-1]
                scaled = scaled.masked_fill(scaled < kth, float("-inf"))
            if 0 < top_p < 1:
                srt, srt_idx = torch.sort(scaled, descending=True)
                sp = torch.softmax(srt, dim=-1)
                drop = sp.cumsum(0) - sp > top_p
                srt = srt.masked_fill(drop, float("-inf"))
                scaled = torch.full_like(scaled, float("-inf")).scatter(0, srt_idx, srt)
            nxt = int(torch.multinomial(torch.softmax(scaled, dim=-1), 1))

        top = torch.topk(probs_full, min(5, probs_full.numel()))
        trace.append({
            "step": len(trace),
            "token": tok.itos[nxt],
            "probability": float(probs_full[nxt]),
            "entropy_nats": entropy,
            "top_5": [{"token": tok.itos[int(i)], "probability": float(pv)}
                      for pv, i in zip(top.values, top.indices)],
        })
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
        if nxt == tok.eos_id or x.size(1) >= model.cfg.max_seq_len:
            break

    new_ids = x[0, n_prompt:].tolist()
    logprobs = [math.log(max(t["probability"], 1e-12)) for t in trace]
    mean_logprob = sum(logprobs) / len(logprobs) if logprobs else 0.0
    return {
        "text": tok.decode(new_ids),
        "trace": trace,
        "prompt_tokens": n_prompt,
        "generated_tokens": len(new_ids),
        "unknown_prompt_tokens": unk_in_prompt,
        "mean_token_logprob": mean_logprob,
        "generation_perplexity": math.exp(-mean_logprob) if logprobs else float("nan"),
        "stopped_on_eos": bool(new_ids and new_ids[-1] == tok.eos_id),
    }


if __name__ == "__main__":                     # tiny CLI: python src/infer.py "what is k means ?"
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("prompt", nargs="+")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--greedy", action="store_true")
    a = p.parse_args()
    m, t = load()
    r = chat(m, t, " ".join(a.prompt), temperature=a.temperature, greedy=a.greedy)
    print(r["text"])
