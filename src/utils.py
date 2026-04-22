"""
Shared utilities: checkpointing, spectrogram plotting, Griffin-Lim WAV export.
"""

import os
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import torchaudio
import torchaudio.transforms as T

from src.dataset import SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS


# ──────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────
def save_checkpoint(state: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f"[ckpt] saved → {path}")


def load_checkpoint(path: str, model, optimizer=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    epoch = ckpt.get("epoch", 0)
    metric = ckpt.get("metric", None)
    print(f"[ckpt] loaded ← {path}  (epoch {epoch}, metric={metric})")
    return epoch, metric


# ──────────────────────────────────────────────
# Spectrogram visualisation
# ──────────────────────────────────────────────
def plot_spectrogram(spec: torch.Tensor, title: str = "", save_path: str | None = None):
    """
    spec : (1, n_mels, T)  or  (n_mels, T)
    """
    if spec.dim() == 3:
        spec = spec.squeeze(0)
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(
        spec.cpu().numpy(),
        aspect="auto",
        origin="lower",
        cmap="magma",
    )
    ax.set_title(title)
    ax.set_xlabel("Time frames")
    ax.set_ylabel("Mel bins")
    plt.colorbar(im, ax=ax, format="%+2.0f")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"[plot] saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names=None, save_path: str | None = None):
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    if class_names is not None:
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=90, fontsize=7)
        ax.set_yticks(range(len(class_names)))
        ax.set_yticklabels(class_names, fontsize=7)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"[plot] confusion matrix → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ──────────────────────────────────────────────
# Griffin-Lim spectrogram → waveform inversion
# ──────────────────────────────────────────────
def spec_to_wav(log_mel_spec: torch.Tensor, n_iter: int = 60) -> torch.Tensor:
    """
    Approximate inversion of a log-mel-spectrogram to a waveform.

    Args:
        log_mel_spec : (1, n_mels, T) — log1p-compressed, unnormalised preferred
        n_iter       : Griffin-Lim iterations

    Returns:
        waveform     : (1, num_samples)
    """
    if log_mel_spec.dim() == 3:
        log_mel_spec = log_mel_spec.squeeze(0)   # (n_mels, T)

    # Undo log1p compression
    mel_spec = torch.expm1(log_mel_spec.clamp(min=-10))

    # Mel → linear spectrogram using pseudo-inverse of mel filterbank
    mel_to_linear = T.InverseMelScale(
        n_stft       = N_FFT // 2 + 1,
        n_mels       = N_MELS,
        sample_rate  = SAMPLE_RATE,
    )
    linear_spec = mel_to_linear(mel_spec.unsqueeze(0))   # (1, n_stft, T)

    # Griffin-Lim
    griffin_lim = T.GriffinLim(
        n_fft      = N_FFT,
        hop_length = HOP_LENGTH,
        n_iter     = n_iter,
    )
    waveform = griffin_lim(linear_spec)   # (1, num_samples)
    return waveform


def save_wav(waveform: torch.Tensor, path: str, sample_rate: int = SAMPLE_RATE):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    torchaudio.save(path, waveform.cpu(), sample_rate)
    print(f"[wav] saved → {path}")


# ──────────────────────────────────────────────
# Metric helpers
# ──────────────────────────────────────────────
def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()
