"""
Phase 1.5 — Sample from the VAE latent space.
Draws random z ~ N(0, I), decodes to spectrogram, converts to WAV via Griffin-Lim.

Usage:
    python sample.py --checkpoint checkpoints/vae_best.pt --n 5
"""

import argparse
from pathlib import Path

import torch

from src.vae import SpectrogramVAE
from src.utils import spec_to_wav, save_wav, plot_spectrogram


def parse_args():
    p = argparse.ArgumentParser(description="Sample from trained VAE")
    p.add_argument("--checkpoint", required=True, help="Path to vae_best.pt")
    p.add_argument("--n",          type=int, default=5, help="Number of samples to generate")
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--output-dir", default="outputs/samples")
    p.add_argument("--no-cuda",    action="store_true")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(
        "cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu"
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device)
    latent_dim = ckpt.get("latent_dim", args.latent_dim)
    model = SpectrogramVAE(latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded VAE from {args.checkpoint} (latent_dim={latent_dim})")

    print(f"Generating {args.n} sample(s)...")
    with torch.no_grad():
        specs = model.sample(args.n, device=device)   # (n, 1, 128, 128)

    for i, spec in enumerate(specs):
        # Save spectrogram plot
        plot_spectrogram(
            spec,
            title=f"Sample {i+1}",
            save_path=str(out_dir / f"sample_{i+1:02d}.png"),
        )

        # Convert to WAV
        waveform = spec_to_wav(spec.cpu())
        save_wav(waveform, str(out_dir / f"sample_{i+1:02d}.wav"))

    print(f"\nDone. Outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
