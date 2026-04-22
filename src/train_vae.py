"""
Phase 1.4 — VAE Training Loop
Usage:
    python -m src.train_vae --data C:/AI/audio_ml/data --epochs 100
"""

import argparse
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.dataset import build_loaders
from src.vae import SpectrogramVAE, vae_loss
from src.utils import save_checkpoint, load_checkpoint, plot_spectrogram


def parse_args():
    p = argparse.ArgumentParser(description="Train Spectrogram VAE")
    p.add_argument("--data",       default="data")
    p.add_argument("--fold",       type=int, default=1)
    p.add_argument("--epochs",     type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--wd",         type=float, default=1e-5)
    p.add_argument("--beta",       type=float, default=1e-3,
                   help="KL weight; increase slowly if latent collapses")
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--workers",    type=int, default=4)
    p.add_argument("--ckpt-dir",   default="checkpoints")
    p.add_argument("--resume",     default=None)
    p.add_argument("--no-cuda",    action="store_true")
    return p.parse_args()


def train_one_epoch(model, loader, optimizer, device, scaler, beta):
    model.train()
    total_loss = total_recon = total_kl = 0.0

    for specs, _ in tqdm(loader, desc="  train", leave=False):
        specs = specs.to(device, non_blocking=True)
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            x_hat, mu, logvar = model(specs)
            loss, recon, kl   = vae_loss(x_hat, specs, mu, logvar, beta=beta)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss  += loss.item()
        total_recon += recon.item()
        total_kl    += kl.item()

    n = len(loader)
    return total_loss / n, total_recon / n, total_kl / n


@torch.no_grad()
def evaluate(model, loader, device, beta):
    model.eval()
    total_loss = total_recon = total_kl = 0.0

    for specs, _ in tqdm(loader, desc="    val", leave=False):
        specs = specs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            x_hat, mu, logvar = model(specs)
            loss, recon, kl   = vae_loss(x_hat, specs, mu, logvar, beta=beta)

        total_loss  += loss.item()
        total_recon += recon.item()
        total_kl    += kl.item()

    n = len(loader)
    return total_loss / n, total_recon / n, total_kl / n


def main():
    args   = parse_args()
    device = torch.device(
        "cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu"
    )
    print(f"Device: {device}  |  beta={args.beta}  |  latent_dim={args.latent_dim}")

    # ── data (VAE mode: returns spec, spec) ───
    train_loader, val_loader = build_loaders(
        root        = args.data,
        test_fold   = args.fold,
        batch_size  = args.batch_size,
        num_workers = args.workers,
        vae_mode    = True,
    )

    # ── model ─────────────────────────────────
    model     = SpectrogramVAE(latent_dim=args.latent_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler    = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    start_epoch = 0
    best_loss   = float("inf")

    if args.resume:
        start_epoch, best_loss = load_checkpoint(
            args.resume, model, optimizer, device=device
        )
        best_loss = best_loss or float("inf")

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── training loop ─────────────────────────
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        tr_loss, tr_recon, tr_kl = train_one_epoch(
            model, train_loader, optimizer, device, scaler, args.beta
        )
        vl_loss, vl_recon, vl_kl = evaluate(
            model, val_loader, device, args.beta
        )
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"train loss {tr_loss:.4f} (recon {tr_recon:.4f} kl {tr_kl:.4f}) | "
            f"val loss {vl_loss:.4f} (recon {vl_recon:.4f} kl {vl_kl:.4f}) | "
            f"{elapsed:.1f}s"
        )

        # ── save best checkpoint ───────────────
        if vl_loss < best_loss:
            best_loss = vl_loss
            save_checkpoint(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "epoch": epoch + 1, "metric": best_loss,
                 "latent_dim": args.latent_dim},
                path=str(ckpt_dir / "vae_best.pt"),
            )
            print(f"  ↓ new best val loss: {best_loss:.6f}")

        # ── periodic checkpoint every 10 epochs ─
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "epoch": epoch + 1, "metric": vl_loss,
                 "latent_dim": args.latent_dim},
                path=str(ckpt_dir / f"vae_epoch{epoch+1}.pt"),
            )

        # ── save reconstruction sample at epoch 1, 10, 50, 100 ──
        if epoch + 1 in {1, 10, 50, 100, args.epochs}:
            model.eval()
            with torch.no_grad():
                sample_spec = next(iter(val_loader))[0][:1].to(device)
                recon, _, _ = model(sample_spec)
            plot_spectrogram(
                sample_spec[0].cpu(),
                title=f"Original (epoch {epoch+1})",
                save_path=str(ckpt_dir / f"recon_orig_ep{epoch+1}.png"),
            )
            plot_spectrogram(
                recon[0].cpu(),
                title=f"Reconstruction (epoch {epoch+1})",
                save_path=str(ckpt_dir / f"recon_pred_ep{epoch+1}.png"),
            )

    print(f"\nVAE training complete. Best val loss: {best_loss:.6f}")


if __name__ == "__main__":
    main()
