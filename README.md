# Audio ML Foundations
**ESC-50 CNN Classifier → Spectrogram VAE**

A standalone PyTorch learning project covering audio preprocessing, CNNs,
encoder/decoder architectures, and latent-space generation — the core mechanics
shared by modern audio generation systems (MusicGen, Stable Audio, AudioLDM).

---

## Project Structure

```
audio_ml/
├── src/
│   ├── dataset.py          # ESC50Dataset — mel-spectrogram pipeline
│   ├── cnn_classifier.py   # 5-block CNN for 50-class classification
│   ├── vae.py              # Convolutional VAE with reparameterization
│   ├── train_cnn.py        # CNN training loop (Phase 1.3)
│   ├── train_vae.py        # VAE training loop (Phase 1.4)
│   └── utils.py            # Shared helpers (plotting, checkpoint I/O)
├── notebooks/
│   └── walkthrough.ipynb   # End-to-end walkthrough notebook
├── data/                   # ESC-50 dataset (git-ignored; run download_data.py)
├── checkpoints/            # Saved model weights (git-ignored)
├── outputs/                # Generated WAVs and plots (git-ignored)
├── download_data.py        # Downloads & extracts ESC-50
├── sample.py               # Phase 1.5 — sample from VAE latent space
├── interpolate.py          # Phase 1.5 — interpolate between two sounds
├── setup_env.ps1           # Creates venv and installs all dependencies
└── requirements.txt
```

---

## Quick Start

### 1. Environment Setup

```powershell
# Run from the project root
.\setup_env.ps1
```

This creates a local `venv/`, installs PyTorch (CUDA 12.6) and all dependencies.

### 2. Download ESC-50 Dataset (~600 MB)

```powershell
.\venv\Scripts\python.exe download_data.py
```

### 3. Train the CNN Classifier

```powershell
.\venv\Scripts\python.exe -m src.train_cnn --fold 1 --epochs 50
```

### 4. Train the Spectrogram VAE

```powershell
.\venv\Scripts\python.exe -m src.train_vae --epochs 100
```

### 5. Explore the Latent Space

```powershell
# Sample random sounds
.\venv\Scripts\python.exe sample.py --checkpoint checkpoints/vae_best.pt --n 5

# Interpolate between two audio files
.\venv\Scripts\python.exe interpolate.py --a data/ESC-50-master/audio/1-100032-A-0.wav --b data/ESC-50-master/audio/1-100038-A-14.wav --steps 10
```

---

## Model Details

### CNN Classifier (Phase 1.3)
- 5 convolutional blocks (32→64→128→256→256 channels)
- BatchNorm + ReLU + MaxPool per block
- AdaptiveAvgPool → Linear(256, 50)
- AdamW + cosine LR schedule
- Target: **≥70% accuracy** on ESC-50 fold 1

### Spectrogram VAE (Phase 1.4)
- Convolutional encoder → μ and log σ² (128-dim latent)
- Reparameterization trick
- Transposed-conv decoder reconstructs 128×128 log-mel-spectrograms
- Loss: MSE reconstruction + β·KL divergence
- Griffin-Lim inversion for WAV export

---

## Hardware Notes

- **GPU (RTX 2080+):** Full batch sizes, fast training
- **CPU only:** Reduce `--batch-size` to 16; CNN ~OK, VAE will be slow

---

## What This Teaches

- Audio preprocessing: sampling, FFT, mel-spectrograms
- Full PyTorch training loop
- Convolutional architectures applied to 2D spectrogram images
- Encoder/decoder design and latent-space geometry
- Conceptual foundations of modern audio generators
