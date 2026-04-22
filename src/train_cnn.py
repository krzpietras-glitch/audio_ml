"""
Phase 1.3 — CNN Training Loop
Usage:
    python -m src.train_cnn --data C:/AI/audio_ml/data --fold 1 --epochs 50
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.dataset import build_loaders
from src.cnn_classifier import CNNClassifier
from src.utils import accuracy, save_checkpoint, load_checkpoint, plot_confusion_matrix


def parse_args():
    p = argparse.ArgumentParser(description="Train ESC-50 CNN Classifier")
    p.add_argument("--data",       default="data",   help="Root dir containing ESC-50-master/")
    p.add_argument("--fold",       type=int, default=1, choices=range(1, 6))
    p.add_argument("--epochs",     type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--wd",         type=float, default=1e-4, help="Weight decay")
    p.add_argument("--workers",    type=int, default=4)
    p.add_argument("--ckpt-dir",   default="checkpoints")
    p.add_argument("--resume",     default=None, help="Path to checkpoint to resume from")
    p.add_argument("--no-cuda",    action="store_true")
    return p.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss, total_acc = 0.0, 0.0
    for specs, labels in tqdm(loader, desc="  train", leave=False):
        specs  = specs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(specs)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_acc  += accuracy(logits.detach(), labels)

    n = len(loader)
    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc = 0.0, 0.0
    all_preds, all_labels = [], []

    for specs, labels in tqdm(loader, desc="    val", leave=False):
        specs  = specs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(specs)
            loss   = criterion(logits, labels)

        total_loss += loss.item()
        total_acc  += accuracy(logits, labels)
        all_preds.append(logits.argmax(1).cpu())
        all_labels.append(labels.cpu())

    n = len(loader)
    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    return total_loss / n, total_acc / n, all_preds, all_labels


def build_confusion_matrix(preds, labels, num_classes=50):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for p, l in zip(preds, labels):
        cm[l, p] += 1
    return cm


def main():
    args = parse_args()
    device = torch.device(
        "cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu"
    )
    print(f"Device: {device}")

    # ── data ──────────────────────────────────
    train_loader, val_loader = build_loaders(
        root       = args.data,
        test_fold  = args.fold,
        batch_size = args.batch_size,
        num_workers= args.workers,
    )
    print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    # ── model ─────────────────────────────────
    model     = CNNClassifier(num_classes=50).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler    = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    start_epoch = 0
    best_acc    = 0.0

    if args.resume:
        start_epoch, best_acc = load_checkpoint(
            args.resume, model, optimizer, device=device
        )
        best_acc = best_acc or 0.0

    # ── training loop ─────────────────────────
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        vl_loss, vl_acc, preds, labels = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
            f"val loss {vl_loss:.4f} acc {vl_acc:.3f} | "
            f"{elapsed:.1f}s"
        )

        # ── save best ─────────────────────────
        if vl_acc > best_acc:
            best_acc = vl_acc
            save_checkpoint(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "epoch": epoch + 1, "metric": best_acc},
                path=str(Path(args.ckpt_dir) / f"cnn_fold{args.fold}_best.pt"),
            )
            print(f"  ↑ new best val acc: {best_acc:.4f}")

        # ── confusion matrix on last epoch ────
        if epoch == args.epochs - 1:
            cm = build_confusion_matrix(preds, labels)
            plot_confusion_matrix(
                cm,
                save_path=str(Path(args.ckpt_dir) / f"confusion_fold{args.fold}.png"),
            )

    print(f"\nTraining complete. Best val accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
