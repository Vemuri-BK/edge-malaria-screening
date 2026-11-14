"""
STEP 4 - TRAINING
=================
Fine-tunes the model on the training split and keeps the checkpoint with the
best validation AUROC.

Per epoch it logs
    model metrics : train loss, val loss, val accuracy / sensitivity / specificity / AUROC
    system metrics: epoch time, throughput (images/s), GPU util & memory,
                    peak GPU memory allocated by PyTorch, CPU %, process RAM

Outputs (in out_dir)
    best_model.pth   weights of the best epoch
    history.csv      one row per epoch with all metrics above

Usage (script)
    python src/train.py --data_root <path> --split_csv splits/patient_split.csv
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config as C
from dataset import build_loaders
from metrics import compute_metrics, softmax_np
from models import EdgeClassifier
from sysmon import SystemMonitor


def set_seed(seed: int = C.SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_logits(model, loader, device):
    """Run the model over a loader; return (logits [N,2], labels [N], mean loss)."""
    model.to(device)          # works even if the model was loaded on the CPU
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    all_logits, all_labels, total_loss = [], [], 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            out = model(x)
        out = out.float()
        total_loss += crit(out, y).item()
        all_logits.append(out.cpu().numpy())
        all_labels.append(y.cpu().numpy())
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    return logits, labels, total_loss / len(labels)


def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    crit = nn.CrossEntropyLoss()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss = crit(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(y)
        n += len(y)
    return total_loss / n, n


def fit(model, loaders, out_dir, epochs=C.EPOCHS, lr=C.LR, weight_decay=C.WEIGHT_DECAY):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    history, best_auroc = [], -1.0
    print(f"Training {model.arch} on {device} for {epochs} epochs")

    for epoch in range(1, epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        mon = SystemMonitor(interval=1.0).start()
        t0 = time.time()

        train_loss, n_seen = train_one_epoch(model, loaders["train"], optimizer, scaler, device)
        train_time = time.time() - t0
        val_logits, val_labels, val_loss = predict_logits(model, loaders["val"], device)
        epoch_time = time.time() - t0
        mon.stop()
        scheduler.step()

        m = compute_metrics(val_labels, softmax_np(val_logits)[:, 1])
        sysm = mon.summary()
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": m["accuracy"],
            "val_sensitivity": m["sensitivity"],
            "val_specificity": m["specificity"],
            "val_auroc": m["auroc"],
            "epoch_time_s": epoch_time,
            "train_throughput_img_s": n_seen / train_time,
            "gpu_peak_alloc_mb": (torch.cuda.max_memory_allocated() / 1e6) if device.type == "cuda" else None,
            "gpu_util_mean": sysm.get("gpu_util_mean"),
            "gpu_mem_max_mb": sysm.get("gpu_mem_mb_max"),
            "cpu_percent_mean": sysm.get("cpu_percent_mean"),
            "proc_rss_max_mb": sysm.get("proc_rss_mb_max"),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)   # saved every epoch

        flag = ""
        if m["auroc"] > best_auroc:
            best_auroc = m["auroc"]
            torch.save(model.state_dict(), out_dir / "best_model.pth")
            flag = "  <- best"
        print(f"ep {epoch:02d} | loss {train_loss:.4f}/{val_loss:.4f} | "
              f"acc {m['accuracy']:.4f} sens {m['sensitivity']:.4f} spec {m['specificity']:.4f} "
              f"auroc {m['auroc']:.4f} | {epoch_time:.0f}s {row['train_throughput_img_s']:.0f} img/s{flag}")

    hist = pd.DataFrame(history)
    if len(hist) == epochs:
        (out_dir / "COMPLETE").write_text("training finished\n")
    model.load_state_dict(torch.load(out_dir / "best_model.pth", map_location=device))
    return model, hist


def main():
    ap = argparse.ArgumentParser(description="Step 4: train")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--split_csv", default=str(C.SPLITS_DIR / "patient_split.csv"))
    ap.add_argument("--arch", default=C.ARCH)
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--out_dir", default=str(C.OUTPUT_DIR / C.ARCH))
    args = ap.parse_args()

    set_seed()
    loaders, _ = build_loaders(args.split_csv, args.data_root)
    model = EdgeClassifier(args.arch, pretrained=C.PRETRAINED)
    fit(model, loaders, args.out_dir, epochs=args.epochs)


if __name__ == "__main__":
    main()
