#!/usr/bin/env python3
"""Fine-tune a pretrained encoder for LEDGAR provision classification (CPU-friendly).

Protocol (mirrors the TF-IDF + SVM study so the two are comparable):
  * weights are trained on the official *train* split only;
  * the *validation* split picks the epoch (and, in pilot mode, the learning rate);
  * the *test* split is scored once, with the selected checkpoint, at the very end.

Outputs in --out:
  log.jsonl                  one JSON line per logging step / epoch
  history.csv                per-epoch validation metrics and timings
  val_logits.npy, test_logits.npy   logits of the selected epoch (row order = parquet order)
  logits_validation.parquet, logits_test.parquet   the same, readable from R (id + one column per class)
  predictions_test.csv / predictions_validation.csv   id, label_id, pred_id, confidence
  summary.json               configuration, timings, selected epoch
  checkpoint.pt              resumable training state (deleted with --no-keep-checkpoint)

Example:
  python transformer/finetune.py --data data/raw --out models/legal-bert-small --epochs 3
  python transformer/finetune.py --data data/raw --out /tmp/pilot --train-subset 6000 \
      --val-subset 2000 --epochs 1 --lr 1e-4 --no-test          # pilot run
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="directory with train/validation/test.parquet")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="nlpaueb/legal-bert-small-uncased")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.06, help="share of steps used for linear warm-up")
    p.add_argument("--seed", type=int, default=20261116)
    p.add_argument("--threads", type=int, default=os.cpu_count())
    p.add_argument("--train-subset", type=int, default=0, help="use a random subset of train (pilot runs)")
    p.add_argument("--val-subset", type=int, default=0)
    p.add_argument("--no-test", action="store_true", help="never touch the test split (pilot runs)")
    p.add_argument("--log-every", type=int, default=100)
    return p.parse_args()


def read_split(data_dir, split):
    df = pd.read_parquet(os.path.join(data_dir, f"{split}.parquet"))
    df["id"] = [f"{split}-{i:05d}" for i in range(1, len(df) + 1)]   # same ids as provisionr::ledgar_read()
    return df


def encode(tokenizer, texts, max_length):
    """Tokenise without padding; padding is done per batch."""
    enc = tokenizer(list(texts), truncation=True, max_length=max_length, padding=False)
    return enc["input_ids"]


def make_batches(lengths, batch_size, shuffle, rng):
    """Length-bucketed batches: sort by length inside large random chunks.

    Keeps the padding per batch small (a large speed-up on CPU) while the
    order of batches, and their membership across epochs, stays random.
    """
    idx = np.arange(len(lengths))
    if not shuffle:
        idx = idx[np.argsort(lengths, kind="stable")]
        return [idx[i:i + batch_size] for i in range(0, len(idx), batch_size)]
    rng.shuffle(idx)
    chunk = batch_size * 50
    batches = []
    for start in range(0, len(idx), chunk):
        part = idx[start:start + chunk]
        part = part[np.argsort(lengths[part], kind="stable")]
        batches += [part[i:i + batch_size] for i in range(0, len(part), batch_size)]
    order = rng.permutation(len(batches))
    return [batches[i] for i in order]


def collate(input_ids, batch_idx, pad_id):
    rows = [input_ids[i] for i in batch_idx]
    width = max(len(r) for r in rows)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for r, row in enumerate(rows):
        ids[r, :len(row)] = torch.tensor(row)
        mask[r, :len(row)] = 1
    return ids, mask


@torch.no_grad()
def predict_logits(model, input_ids, pad_id, batch_size=128):
    model.eval()
    lengths = np.array([len(x) for x in input_ids])
    out = np.zeros((len(input_ids), model.config.num_labels), dtype=np.float32)
    for batch in make_batches(lengths, batch_size, shuffle=False, rng=None):
        ids, mask = collate(input_ids, batch, pad_id)
        out[batch] = model(input_ids=ids, attention_mask=mask).logits.float().numpy()
    return out


def scores(y_true, logits):
    """Accuracy (= micro-F1) and macro-F1 over classes present in truth or prediction."""
    pred = logits.argmax(1)
    k = logits.shape[1]
    tp = np.bincount(y_true[pred == y_true], minlength=k)
    support = np.bincount(y_true, minlength=k)
    predicted = np.bincount(pred, minlength=k)
    present = (support + predicted) > 0
    f1 = np.where(present, 2 * tp / np.maximum(support + predicted, 1), 0.0)
    return {"accuracy": float((pred == y_true).mean()), "macro_f1": float(f1[present].mean())}


def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(a.threads)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    train, val = read_split(a.data, "train"), read_split(a.data, "validation")
    if a.train_subset:
        train = train.sample(n=a.train_subset, random_state=a.seed).reset_index(drop=True)
    if a.val_subset:
        val = val.sample(n=a.val_subset, random_state=a.seed).reset_index(drop=True)
    num_labels = 100

    tok = AutoTokenizer.from_pretrained(a.model)
    t0 = time.time()
    x_train, x_val = encode(tok, train["text"], a.max_length), encode(tok, val["text"], a.max_length)
    y_train, y_val = train["label"].to_numpy(), val["label"].to_numpy()
    len_train = np.array([len(x) for x in x_train])
    truncated = float((len_train >= a.max_length).mean())
    print(f"tokenised {len(x_train)} train / {len(x_val)} val in {time.time() - t0:.0f}s; "
          f"mean length {len_train.mean():.0f} tokens, {truncated:.1%} truncated at {a.max_length}", flush=True)

    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=num_labels)
    n_params = sum(p.numel() for p in model.parameters())

    decay = [p for n, p in model.named_parameters() if not any(s in n for s in ("bias", "LayerNorm.weight"))]
    no_decay = [p for n, p in model.named_parameters() if any(s in n for s in ("bias", "LayerNorm.weight"))]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=a.lr)
    steps_per_epoch = math.ceil(len(x_train) / a.batch_size)
    total_steps = steps_per_epoch * a.epochs
    warmup_steps = int(a.warmup * total_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / max(1, warmup_steps) if s < warmup_steps
        else max(0.0, (total_steps - s) / max(1, total_steps - warmup_steps)))

    # -- resume ---------------------------------------------------------------
    ckpt_path = os.path.join(a.out, "checkpoint.pt")
    start_epoch, history, train_seconds, best = 0, [], 0.0, {"accuracy": -1}
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start_epoch, history, train_seconds, best = ck["epoch"], ck["history"], ck["train_seconds"], ck["best"]
        rng = np.random.default_rng(a.seed + start_epoch)
        print(f"resumed after epoch {start_epoch}", flush=True)

    log = open(os.path.join(a.out, "log.jsonl"), "a")
    for epoch in range(start_epoch, a.epochs):
        model.train()
        t_epoch, running, seen = time.time(), 0.0, 0
        batches = make_batches(len_train, a.batch_size, shuffle=True, rng=rng)
        for step, batch in enumerate(batches, 1):
            ids, mask = collate(x_train, batch, tok.pad_token_id)
            loss = model(input_ids=ids, attention_mask=mask, labels=torch.tensor(y_train[batch])).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            running += loss.item() * len(batch); seen += len(batch)
            if step % a.log_every == 0 or step == len(batches):
                rec = {"epoch": epoch + 1, "step": step, "of": len(batches), "train_loss": running / seen,
                       "lr": sched.get_last_lr()[0], "samples_per_s": seen / (time.time() - t_epoch)}
                log.write(json.dumps(rec) + "\n"); log.flush()
                print(json.dumps(rec), flush=True)
        epoch_seconds = time.time() - t_epoch
        train_seconds += epoch_seconds

        val_logits = predict_logits(model, x_val, tok.pad_token_id)
        m = scores(y_val, val_logits)
        row = {"epoch": epoch + 1, "train_loss": running / seen, "val_accuracy": m["accuracy"],
               "val_macro_f1": m["macro_f1"], "epoch_minutes": epoch_seconds / 60}
        history.append(row)
        log.write(json.dumps(row) + "\n"); log.flush()
        print(json.dumps(row), flush=True)
        if m["accuracy"] > best["accuracy"]:
            best = {"epoch": epoch + 1, **m}
            torch.save(model.state_dict(), os.path.join(a.out, "best_model.pt"))
            np.save(os.path.join(a.out, "val_logits.npy"), val_logits)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch + 1, "history": history, "train_seconds": train_seconds, "best": best},
                   ckpt_path)
        pd.DataFrame(history).to_csv(os.path.join(a.out, "history.csv"), index=False)

    # -- final scoring with the selected epoch ---------------------------------
    model.load_state_dict(torch.load(os.path.join(a.out, "best_model.pt"), map_location="cpu"))
    val_logits = np.load(os.path.join(a.out, "val_logits.npy"))

    def write_predictions(df, logits, split):
        wide = pd.DataFrame(logits, columns=[f"c{j}" for j in range(logits.shape[1])])
        wide.insert(0, "id", df["id"].to_numpy())
        wide.to_parquet(os.path.join(a.out, f"logits_{split}.parquet"), index=False)
        z = logits - logits.max(1, keepdims=True)
        prob = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        pd.DataFrame({"id": df["id"], "label_id": df["label"], "pred_id": logits.argmax(1),
                      "confidence": prob.max(1)}).to_csv(os.path.join(a.out, f"predictions_{split}.csv"), index=False)

    write_predictions(val, val_logits, "validation")
    summary = {"model": a.model, "parameters": n_params, "max_length": a.max_length, "epochs": a.epochs,
               "batch_size": a.batch_size, "lr": a.lr, "weight_decay": a.weight_decay, "warmup": a.warmup,
               "seed": a.seed, "threads": a.threads, "train_rows": len(x_train),
               "mean_train_tokens": float(len_train.mean()), "share_truncated": truncated,
               "train_minutes": train_seconds / 60, "selected_epoch": best["epoch"],
               "val_accuracy": best["accuracy"], "val_macro_f1": best["macro_f1"], "torch": torch.__version__}
    if not a.no_test:
        test = read_split(a.data, "test")
        x_test = encode(tok, test["text"], a.max_length)
        t0 = time.time()
        test_logits = predict_logits(model, x_test, tok.pad_token_id)
        summary["test_predict_seconds"] = time.time() - t0
        np.save(os.path.join(a.out, "test_logits.npy"), test_logits)
        write_predictions(test, test_logits, "test")
        summary.update({f"test_{k}": v for k, v in scores(test["label"].to_numpy(), test_logits).items()})
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
