"""
Phase 1.2 — Data Pipeline
ESC-50 Dataset: WAV → resample → log-mel-spectrogram → normalise → (spec, label)
"""

import os
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import soundfile as sf
import torchaudio
import torchaudio.transforms as T

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
SAMPLE_RATE   = 22_050
DURATION_SEC  = 5
N_SAMPLES     = SAMPLE_RATE * DURATION_SEC   # 110 250
N_FFT         = 2_048
HOP_LENGTH    = 512
N_MELS        = 128
SPEC_TIME     = 128   # fixed width after crop/pad for VAE compatibility
# After log-mel: freq=128, time≈216. We crop/pad time to SPEC_TIME.


class ESC50Dataset(Dataset):
    """
    Args:
        root        : path that contains ``ESC-50-master/``
        fold        : int 1-5 (test fold) OR list of ints (train folds)
        augment     : apply time-shift + SpecAugment during training
        vae_mode    : if True, returns (spec, spec) instead of (spec, label)
                      and pads/crops time axis to SPEC_TIME
    """

    def __init__(
        self,
        root: str,
        fold,                      # int or list[int]
        augment: bool = False,
        vae_mode: bool = False,
    ):
        self.root      = Path(root) / "ESC-50-master"
        self.audio_dir = self.root / "audio"
        self.augment   = augment
        self.vae_mode  = vae_mode

        folds = [fold] if isinstance(fold, int) else list(fold)
        self.samples = self._load_meta(folds)

        # Mel-spectrogram transform (CPU, deterministic)
        self.mel_transform = T.MelSpectrogram(
            sample_rate = SAMPLE_RATE,
            n_fft       = N_FFT,
            hop_length  = HOP_LENGTH,
            n_mels      = N_MELS,
        )

        # SpecAugment
        self.time_mask = T.TimeMasking(time_mask_param=30)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=20)

    # ── internal ──────────────────────────────
    def _load_meta(self, folds):
        meta_path = self.root / "meta" / "esc50.csv"
        samples = []
        with open(meta_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row["fold"]) in folds:
                    samples.append({
                        "path"   : self.audio_dir / row["filename"],
                        "label"  : int(row["target"]),
                        "fold"   : int(row["fold"]),
                    })
        return samples

    def _load_waveform(self, path: Path):
        data, sr = sf.read(str(path), always_2d=True)  # (samples, channels)
        waveform = torch.from_numpy(data.T).float()     # (channels, samples)
        # Mix down to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # Resample
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        # Pad or trim to exactly N_SAMPLES
        length = waveform.shape[-1]
        if length < N_SAMPLES:
            waveform = torch.nn.functional.pad(waveform, (0, N_SAMPLES - length))
        else:
            waveform = waveform[..., :N_SAMPLES]
        return waveform  # shape: (1, N_SAMPLES)

    def _time_shift(self, waveform, max_shift=0.1):
        shift = int(random.uniform(-max_shift, max_shift) * N_SAMPLES)
        return torch.roll(waveform, shift, dims=-1)

    def _to_log_mel(self, waveform):
        mel = self.mel_transform(waveform)         # (1, n_mels, time)
        log_mel = torch.log1p(mel)
        return log_mel

    def _normalise(self, spec):
        mean = spec.mean()
        std  = spec.std() + 1e-8
        return (spec - mean) / std

    def _fix_time(self, spec):
        """Pad or crop time dimension to SPEC_TIME (for VAE)."""
        t = spec.shape[-1]
        if t >= SPEC_TIME:
            return spec[..., :SPEC_TIME]
        else:
            return torch.nn.functional.pad(spec, (0, SPEC_TIME - t))

    # ── public ────────────────────────────────
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        waveform = self._load_waveform(item["path"])

        # Augmentation (training only)
        if self.augment:
            waveform = self._time_shift(waveform)

        spec = self._to_log_mel(waveform)  # (1, 128, ~216)

        if self.augment:
            spec = self.time_mask(spec)
            spec = self.freq_mask(spec)

        spec = self._normalise(spec)

        if self.vae_mode:
            spec = self._fix_time(spec)    # (1, 128, 128)
            return spec, spec              # target == input for reconstruction

        return spec, item["label"]


# ──────────────────────────────────────────────
# Helper: build train/val DataLoaders for a given fold
# ──────────────────────────────────────────────
def build_loaders(root, test_fold=1, batch_size=32, num_workers=4, vae_mode=False):
    train_folds = [f for f in range(1, 6) if f != test_fold]
    train_ds = ESC50Dataset(root, fold=train_folds, augment=True,  vae_mode=vae_mode)
    val_ds   = ESC50Dataset(root, fold=test_fold,   augment=False, vae_mode=vae_mode)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader
