"""
Phase 1.5 — Interpolate between two audio clips in the VAE latent space.
Encodes each clip, walks a straight line between their latent vectors,
decodes each step, and saves them as WAVs.

Usage:
    python interpolate.py \\
        --a data/ESC-50-master/audio/1-100032-A-0.wav \\
        --b data/ESC-50-master/audio/1-100038-A-14.wav \\
        --checkpoint checkpoints/vae_best.pt \\
        --steps 10
"""

import argparse
from pathlib import Path

import torch
import torchaudio

from src.vae import SpectrogramVAE
from src.dataset import SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS, N_SAMPLES, SPEC_TIME
from src.utils import spec_to_wav, save_wav, plot_spectrogram
import torchaudio.transforms as T


def parse_args():
    p = argparse.ArgumentParser(description="Latent-space interpolation")
    p.add_argument("--a",          required=True, help="Path to first WAV file")
    p.add_argument("--b",          required=True, help="Path to second WAV file")
    p.add_argument("--checkpoint", required=True, help="Path to vae_best.pt")
    p.add_argument("--steps",      type=int, default=10)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--output-dir", default="outputs/interpolation")
    p.add_argument("--no-cuda",    action="store_true")
    return p.parse_args()


def load_as_spec(wav_path: str, device) -> torch.Tensor:
    """Load a WAV file and convert to a normalised 128×128 log-mel spectrogram."""
    import soundfile as sf
    data, sr = sf.read(wav_path, always_2d=True)    # (samples, channels)
    waveform  = torch.from_numpy(data.T).float()    # (channels, samples)
    # Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # Resample
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    # Pad/trim
    length = waveform.shape[-1]
    if length < N_SAMPLES:
        waveform = torch.nn.functional.pad(waveform, (0, N_SAMPLES - length))
    else:
        waveform = waveform[..., :N_SAMPLES]

    mel_transform = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel = mel_transform(waveform)
    log_mel = torch.log1p(mel)

    # Normalise
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)

    # Fix time to SPEC_TIME
    t = log_mel.shape[-1]
    if t >= SPEC_TIME:
        log_mel = log_mel[..., :SPEC_TIME]
    else:
        log_mel = torch.nn.functional.pad(log_mel, (0, SPEC_TIME - t))

    return log_mel.unsqueeze(0).to(device)   # (1, 1, 128, 128)


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
    print(f"Loaded VAE (latent_dim={latent_dim})")

    # Encode both clips
    with torch.no_grad():
        spec_a = load_as_spec(args.a, device)
        spec_b = load_as_spec(args.b, device)
        z_a    = model.encode(spec_a)   # (1, latent_dim)
        z_b    = model.encode(spec_b)

    print(f"Interpolating between:\n  A: {args.a}\n  B: {args.b}")
    print(f"Steps: {args.steps}")

    # Linear interpolation
    with torch.no_grad():
        for step in range(args.steps + 1):
            alpha = step / args.steps
            z     = (1 - alpha) * z_a + alpha * z_b
            spec  = model.decode(z)     # (1, 1, 128, 128)

            tag = f"step_{step:02d}_alpha{alpha:.2f}"

            plot_spectrogram(
                spec[0].cpu(),
                title=f"Interpolation α={alpha:.2f}",
                save_path=str(out_dir / f"{tag}.png"),
            )

            waveform = spec_to_wav(spec[0].cpu())
            save_wav(waveform, str(out_dir / f"{tag}.wav"))

    print(f"\nDone. {args.steps + 1} steps saved to {out_dir}/")


if __name__ == "__main__":
    main()
