"""Autoresearch: greedy coordinate-ascent hill climbing over NanoLlama's design.

Each trial trains a fresh model on a deliberately short budget and scores it on
the *validation* split only -- the test split is never touched here, so the
final held-out numbers stay honest.

Search procedure (hill climbing):
  1. Score the incumbent configuration.
  2. Walk the hyperparameter axes one at a time. For each axis, train every legal
     alternative value while the other axes stay fixed.
  3. Accept the single best improving move on that axis, then move on.
  4. Repeat for `--passes` sweeps or until a full sweep finds no improvement.

Every trial -- accepted or rejected -- is written to artifacts/hillclimb.json.

Run:  python src/autoresearch.py
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from argparse import Namespace
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import PretrainStream, SFTDataset, build_tokenizer, load_pretrain_text, load_split
from model import NanoLlamaConfig
from train import eval_sft, pick_device, train

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"

# Search space. Axis order is the order the sweep visits them in.
SPACE: dict[str, list] = {
    "d_model": [96, 128, 192, 256],
    "n_layers": [2, 3, 4, 6],
    "n_heads": [4, 6, 8],
    "n_kv_heads": [1, 2, 4],
    "sft_lr": [5e-4, 1e-3, 1.5e-3, 3e-3],
    "dropout": [0.0, 0.1, 0.2],
}

START = {"d_model": 128, "n_layers": 3, "n_heads": 4, "n_kv_heads": 2,
         "sft_lr": 1e-3, "dropout": 0.1}


def is_legal(point: dict) -> bool:
    d, h, kv = point["d_model"], point["n_heads"], point["n_kv_heads"]
    if d % h or h % kv:
        return False
    head_dim = d // h
    return head_dim % 2 == 0 and head_dim >= 16


def run_trial(point: dict, base_args: Namespace, tok, pre_stream, train_ds, val_ds,
              device) -> dict:
    cfg = NanoLlamaConfig(vocab_size=tok.vocab_size, d_model=point["d_model"],
                          n_layers=point["n_layers"], n_heads=point["n_heads"],
                          n_kv_heads=point["n_kv_heads"], dropout=point["dropout"],
                          max_seq_len=base_args.max_seq_len)
    args = copy.deepcopy(base_args)
    args.sft_lr = point["sft_lr"]
    t0 = time.time()
    model, _, _ = train(cfg, args, tok, pre_stream, train_ds, val_ds, device,
                        log=[], verbose=False)
    metrics = eval_sft(model, val_ds, device)
    params = model.num_parameters()
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    return {"point": dict(point), "val_loss": metrics["loss"],
            "val_perplexity": metrics["perplexity"],
            "val_token_accuracy": metrics["token_accuracy"],
            "parameters": params, "seconds": round(time.time() - t0, 1)}


def main() -> None:
    p = argparse.ArgumentParser(description="Hill-climbing architecture search")
    p.add_argument("--passes", type=int, default=2)
    p.add_argument("--pretrain-steps", type=int, default=150)
    p.add_argument("--sft-epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-seq-len", type=int, default=96)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", type=str, default="auto")
    cli = p.parse_args()

    device = pick_device(cli.device)
    ARTIFACTS.mkdir(exist_ok=True)

    train_rows, val_rows = load_split("train"), load_split("val")
    pretrain_text = load_pretrain_text()
    tok = build_tokenizer(train_rows, pretrain_text)
    pre_stream = PretrainStream(pretrain_text, tok, block_size=min(64, cli.max_seq_len))
    train_ds = SFTDataset(train_rows, tok, cli.max_seq_len)
    val_ds = SFTDataset(val_rows, tok, cli.max_seq_len)

    base_args = Namespace(
        batch_size=cli.batch_size, pretrain_steps=cli.pretrain_steps,
        pretrain_lr=3e-3, sft_epochs=cli.sft_epochs, sft_lr=1e-3,
        weight_decay=0.1, grad_clip=1.0, warmup=40, eval_every=10_000,
        seed=cli.seed, max_seq_len=cli.max_seq_len,
    )

    trials: list[dict] = []
    incumbent = dict(START)
    t0 = time.time()

    res = run_trial(incumbent, base_args, tok, pre_stream, train_ds, val_ds, device)
    res.update(trial=0, axis="start", decision="incumbent")
    trials.append(res)
    best_loss = res["val_loss"]
    print(f"[trial 0] start val_loss={best_loss:.4f} params={res['parameters']:,}")

    for sweep in range(cli.passes):
        improved_this_sweep = False
        for axis, values in SPACE.items():
            candidates = []
            for value in values:
                if value == incumbent[axis]:
                    continue
                point = {**incumbent, axis: value}
                if not is_legal(point):
                    trials.append({"trial": len(trials), "sweep": sweep, "axis": axis,
                                   "point": point, "decision": "illegal"})
                    continue
                res = run_trial(point, base_args, tok, pre_stream, train_ds, val_ds, device)
                res.update(trial=len(trials), sweep=sweep, axis=axis, decision="evaluated")
                trials.append(res)
                candidates.append(res)
                print(f"[trial {res['trial']:2d}] {axis}={value} "
                      f"val_loss={res['val_loss']:.4f} acc={res['val_token_accuracy']:.3f} "
                      f"params={res['parameters']:,} ({res['seconds']}s)")
            if not candidates:
                continue
            winner = min(candidates, key=lambda r: r["val_loss"])
            if winner["val_loss"] < best_loss:
                best_loss = winner["val_loss"]
                incumbent = dict(winner["point"])
                winner["decision"] = "accepted"
                improved_this_sweep = True
                print(f"  -> accept {axis}={incumbent[axis]} (val_loss {best_loss:.4f})")
            else:
                for c in candidates:
                    c["decision"] = "rejected"
                print(f"  -> keep {axis}={incumbent[axis]}")
        if not improved_this_sweep:
            print(f"sweep {sweep}: no improvement, stopping early")
            break

    report = {
        "method": "greedy coordinate-ascent hill climbing",
        "objective": "validation masked cross-entropy (lower is better)",
        "budget_per_trial": {"pretrain_steps": cli.pretrain_steps,
                             "sft_epochs": cli.sft_epochs,
                             "batch_size": cli.batch_size},
        "search_space": SPACE,
        "start": START,
        "best_point": incumbent,
        "best_val_loss": best_loss,
        "n_trials_evaluated": sum(t["decision"] != "illegal" for t in trials),
        "total_seconds": round(time.time() - t0, 1),
        "device": str(device),
        "trials": trials,
    }
    (ARTIFACTS / "hillclimb.json").write_text(json.dumps(report, indent=2) + "\n")
    print("\nbest point:", json.dumps(incumbent), f"val_loss={best_loss:.4f}")
    print(f"wrote {ARTIFACTS / 'hillclimb.json'}")


if __name__ == "__main__":
    main()
