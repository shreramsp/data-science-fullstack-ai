"""Two-stage training for NanoLlama: autoregressive pretraining -> SFT.

Stage 1 (pretraining)  : next-token prediction over declarative prose, loss on
                         every token. Teaches the model the domain's vocabulary
                         and sentence shapes.
Stage 2 (SFT)          : the same weights trained on <user>/<assistant> chat
                         sequences with the loss masked to the response tokens.
                         Teaches instruction following.

Run:  python src/train.py            (defaults are the shipped configuration)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import (PretrainStream, SFTDataset, build_tokenizer, encode_prompt,
                     load_pretrain_text, load_split)
from model import NanoLlama, NanoLlamaConfig

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
ARTIFACTS = ROOT / "artifacts"


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lr_at(step: int, total: int, peak: float, warmup: int, floor_ratio: float = 0.1) -> float:
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    cosine = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return peak * (floor_ratio + (1 - floor_ratio) * cosine)


@torch.no_grad()
def eval_sft(model, ds: SFTDataset, device, batch_size: int = 64) -> dict:
    """Masked cross-entropy plus teacher-forced response-token accuracy."""
    model.eval()
    loss_sum, tok_sum, correct = 0.0, 0.0, 0.0
    gen = torch.Generator().manual_seed(0)
    for x, y, m in ds.iter_batches(batch_size, gen, device, shuffle=False):
        logits, loss = model(x, y, m)
        n = m.sum().item()
        loss_sum += loss.item() * n
        tok_sum += n
        correct += ((logits.argmax(-1) == y).float() * m).sum().item()
    model.train()
    mean_loss = loss_sum / max(tok_sum, 1)
    return {
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20)),
        "token_accuracy": correct / max(tok_sum, 1),
        "supervised_tokens": int(tok_sum),
    }


@torch.no_grad()
def eval_pretrain(model, stream: PretrainStream, device, iters: int = 20) -> dict:
    model.eval()
    gen = torch.Generator().manual_seed(0)
    losses = []
    for _ in range(iters):
        x, y = stream.batch("val", 16, gen, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    mean = sum(losses) / len(losses)
    return {"loss": mean, "perplexity": math.exp(min(mean, 20))}


@torch.no_grad()
def eval_exact_match(model, rows, tok, device, max_new_tokens: int = 64,
                     max_samples: int = 12) -> dict:
    """Greedy-decode every held-out prompt and compare to the reference string."""
    model.eval()
    hits, by_intent = 0, {}
    all_results = []
    for r in rows:
        ids = torch.tensor([encode_prompt(tok, r["prompt"])], device=device)
        out = model.generate(ids, max_new_tokens, greedy=True, eos_id=tok.eos_id)
        pred = tok.decode(out[0, ids.size(1):].tolist())
        ref = tok.decode(tok.encode(r["response"]))
        ok = pred.strip() == ref.strip()
        hits += ok
        agg = by_intent.setdefault(r["intent"], [0, 0])
        agg[0] += ok
        agg[1] += 1
        all_results.append({"prompt": r["prompt"], "reference": ref,
                            "prediction": pred, "exact_match": bool(ok)})
    model.train()

    # Every failure first (there are always few), then fill the rest with the
    # earliest successes -- a "first 12 rows" sample would otherwise happily
    # miss every failure if they weren't near the top of the file.
    failures = [s for s in all_results if not s["exact_match"]]
    successes = [s for s in all_results if s["exact_match"]]
    samples = (failures + successes)[:max_samples]

    return {
        "exact_match": hits / max(len(rows), 1),
        "n": len(rows),
        "by_intent": {k: {"exact_match": v[0] / v[1], "n": v[1]}
                      for k, v in sorted(by_intent.items())},
        "n_failures": len(failures),
        "samples": samples,
    }


def majority_response_baseline(train_rows, eval_rows) -> float:
    counts: dict[str, int] = {}
    for r in train_rows:
        counts[r["response"]] = counts.get(r["response"], 0) + 1
    top = max(counts, key=counts.get)
    return sum(r["response"] == top for r in eval_rows) / max(len(eval_rows), 1)


def train(cfg: NanoLlamaConfig, args, tok, pre_stream, train_ds, val_ds, device,
          log: list | None = None, verbose: bool = True):
    torch.manual_seed(args.seed)
    model = NanoLlama(cfg).to(device)
    gen = torch.Generator().manual_seed(args.seed)
    log = [] if log is None else log

    # ---- Stage 1: autoregressive pretraining -----------------------------
    opt = torch.optim.AdamW(model.parameters(), lr=args.pretrain_lr,
                            betas=(0.9, 0.95), weight_decay=args.weight_decay)
    for step in range(args.pretrain_steps):
        lr = lr_at(step, args.pretrain_steps, args.pretrain_lr, args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = pre_stream.batch("train", args.batch_size, gen, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        if step % args.eval_every == 0 or step == args.pretrain_steps - 1:
            ev = eval_pretrain(model, pre_stream, device)
            log.append({"stage": "pretrain", "step": step, "lr": lr,
                        "train_loss": loss.item(), "val_loss": ev["loss"]})
            if verbose:
                print(f"[pretrain {step:4d}/{args.pretrain_steps}] "
                      f"train {loss.item():.3f}  val {ev['loss']:.3f}  lr {lr:.2e}")

    # ---- Stage 2: supervised fine-tuning ---------------------------------
    opt = torch.optim.AdamW(model.parameters(), lr=args.sft_lr,
                            betas=(0.9, 0.95), weight_decay=args.weight_decay)
    best = {"val_loss": float("inf"), "state": None, "step": -1}
    step = 0
    steps_per_epoch = math.ceil(len(train_ds) / args.batch_size)
    total = args.sft_epochs * steps_per_epoch
    for epoch in range(args.sft_epochs):
        for x, y, m in train_ds.iter_batches(args.batch_size, gen, device):
            lr = lr_at(step, total, args.sft_lr, args.warmup)
            for g in opt.param_groups:
                g["lr"] = lr
            _, loss = model(x, y, m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            if step % args.eval_every == 0 or step == total - 1:
                ev = eval_sft(model, val_ds, device)
                log.append({"stage": "sft", "step": step, "epoch": epoch, "lr": lr,
                            "train_loss": loss.item(), "val_loss": ev["loss"],
                            "val_token_accuracy": ev["token_accuracy"]})
                if ev["loss"] < best["val_loss"]:
                    best = {"val_loss": ev["loss"], "step": step,
                            "state": {k: v.detach().cpu().clone()
                                      for k, v in model.state_dict().items()}}
                if verbose:
                    print(f"[sft e{epoch} {step:4d}/{total}] train {loss.item():.3f}  "
                          f"val {ev['loss']:.3f}  tok-acc {ev['token_accuracy']:.3f}")
            step += 1

    if best["state"] is not None:               # checkpoint selection = early stopping
        model.load_state_dict(best["state"])
    return model, log, best


def main() -> None:
    p = argparse.ArgumentParser(description="Train NanoLlama")
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=6)
    p.add_argument("--n-kv-heads", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max-seq-len", type=int, default=96)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--pretrain-steps", type=int, default=600)
    p.add_argument("--pretrain-lr", type=float, default=3e-3)
    p.add_argument("--sft-epochs", type=int, default=8)
    p.add_argument("--sft-lr", type=float, default=1.5e-3)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup", type=int, default=40)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    device = pick_device(args.device)
    MODELS.mkdir(exist_ok=True)
    ARTIFACTS.mkdir(exist_ok=True)

    train_rows, val_rows, test_rows = (load_split(s) for s in ("train", "val", "test"))
    pretrain_text = load_pretrain_text()
    tok = build_tokenizer(train_rows, pretrain_text)

    cfg = NanoLlamaConfig(vocab_size=tok.vocab_size, d_model=args.d_model,
                          n_layers=args.n_layers, n_heads=args.n_heads,
                          n_kv_heads=args.n_kv_heads, dropout=args.dropout,
                          max_seq_len=args.max_seq_len)
    pre_stream = PretrainStream(pretrain_text, tok, block_size=min(64, cfg.max_seq_len))
    train_ds = SFTDataset(train_rows, tok, cfg.max_seq_len)
    val_ds = SFTDataset(val_rows, tok, cfg.max_seq_len)
    test_ds = SFTDataset(test_rows, tok, cfg.max_seq_len)

    print(f"device={device}  vocab={tok.vocab_size}  train={len(train_ds)} "
          f"val={len(val_ds)} test={len(test_ds)}  longest_seq={train_ds.max_length}")

    t0 = time.time()
    model, log, best = train(cfg, args, tok, pre_stream, train_ds, val_ds, device)
    train_seconds = time.time() - t0

    pretrain_end = [r for r in log if r["stage"] == "pretrain"][-1]
    val_m = eval_sft(model, val_ds, device)
    test_m = eval_sft(model, test_ds, device)
    # Measured on the *final* weights, i.e. after SFT -- so it shows how far the
    # model drifted from the raw prose distribution, not how stage 1 itself went.
    pre_m = eval_pretrain(model, pre_stream, device)
    gen_val = eval_exact_match(model, val_rows, tok, device)
    gen_test = eval_exact_match(model, test_rows, tok, device)
    baseline = majority_response_baseline(train_rows, test_rows)

    metrics = {
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": str(device),
        "train_seconds": round(train_seconds, 1),
        "parameters": model.num_parameters(),
        "parameters_non_embedding": model.num_parameters(non_embedding=True),
        "config": cfg.to_dict(),
        "hyperparameters": vars(args),
        "best_checkpoint_step": best["step"],
        "pretrain_stage_end_val_loss": pretrain_end["val_loss"],
        "pretrain_val_after_sft": pre_m,
        "sft_val": val_m,
        "sft_test": test_m,
        "generation_val": {k: v for k, v in gen_val.items() if k != "samples"},
        "generation_test": {k: v for k, v in gen_test.items() if k != "samples"},
        "majority_response_baseline_exact_match": baseline,
        "dataset": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds),
                    "vocab_size": tok.vocab_size},
    }

    tok.save(MODELS / "tokenizer.json")
    torch.save({"config": cfg.to_dict(), "state_dict": model.state_dict()},
               MODELS / "nanollama.pt")
    (MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (ARTIFACTS / "train_log.json").write_text(json.dumps(log, indent=2) + "\n")
    (ARTIFACTS / "generation_samples.json").write_text(
        json.dumps({"val": gen_val["samples"], "test": gen_test["samples"]}, indent=2) + "\n")

    print(json.dumps({k: v for k, v in metrics.items()
                      if k in ("parameters", "train_seconds", "sft_val", "sft_test",
                               "generation_test", "majority_response_baseline_exact_match")},
                     indent=2))


if __name__ == "__main__":
    main()
