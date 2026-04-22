"""
Phase 1.4 — Spectrogram VAE
Convolutional VAE operating on 128×128 log-mel-spectrograms.

Input / output shape : (B, 1, 128, 128)
Latent dim           : 128
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


LATENT_DIM   = 128
SPATIAL_AFTER_ENC = 8   # 128 → /2 → /2 → /2 → /2 = 8


# ──────────────────────────────────────────────
# Encoder
# ──────────────────────────────────────────────
class Encoder(nn.Module):
    """
    (B,1,128,128)
    → Conv(1→32,  s=2) → (B,32,64,64)
    → Conv(32→64, s=2) → (B,64,32,32)
    → Conv(64→128,s=2) → (B,128,16,16)
    → Conv(128→256,s=2)→ (B,256,8,8)
    → Flatten → Linear → mu, logvar  (B, LATENT_DIM)
    """
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            self._block(1,    32),
            self._block(32,   64),
            self._block(64,  128),
            self._block(128, 256),
        )
        flat_dim = 256 * SPATIAL_AFTER_ENC * SPATIAL_AFTER_ENC   # 16384
        self.fc_mu     = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

    @staticmethod
    def _block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        h = self.net(x)                         # (B, 256, 8, 8)
        h = h.flatten(1)                        # (B, 16384)
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


# ──────────────────────────────────────────────
# Decoder
# ──────────────────────────────────────────────
class Decoder(nn.Module):
    """
    (B, LATENT_DIM)
    → Linear → (B,256,8,8)
    → ConvT(256→128,s=2) → (B,128,16,16)
    → ConvT(128→64, s=2) → (B,64,32,32)
    → ConvT(64→32,  s=2) → (B,32,64,64)
    → ConvT(32→1,   s=2) → (B,1,128,128)
    """
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        flat_dim = 256 * SPATIAL_AFTER_ENC * SPATIAL_AFTER_ENC
        self.fc  = nn.Linear(latent_dim, flat_dim)

        self.net = nn.Sequential(
            self._block(256, 128),
            self._block(128, 64),
            self._block(64,  32),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            # No BN/activation on last layer; output is unbounded log-mel
        )

    @staticmethod
    def _block(in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, z):
        h = self.fc(z)                          # (B, 16384)
        h = h.view(-1, 256, SPATIAL_AFTER_ENC, SPATIAL_AFTER_ENC)   # (B,256,8,8)
        return self.net(h)                      # (B, 1, 128, 128)


# ──────────────────────────────────────────────
# VAE
# ──────────────────────────────────────────────
class SpectrogramVAE(nn.Module):
    """
    Full VAE = Encoder + reparameterize + Decoder.

    Loss (returned by vae_loss):
        total = MSE_reconstruction + beta * KL
    """

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.latent_dim = latent_dim

    # ── reparameterization trick ───────────────
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu   # deterministic at eval time

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z  = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return self.reparameterize(mu, logvar)

    def decode(self, z):
        return self.decoder(z)

    def sample(self, n: int, device):
        """Draw n samples from the prior N(0,I)."""
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decoder(z)


# ──────────────────────────────────────────────
# Loss function
# ──────────────────────────────────────────────
def vae_loss(x_hat, x, mu, logvar, beta: float = 1e-3):
    """
    Args:
        x_hat   : reconstructed spectrogram  (B,1,128,128)
        x       : original spectrogram       (B,1,128,128)
        mu      : encoder mean               (B, latent_dim)
        logvar  : encoder log-variance       (B, latent_dim)
        beta    : weight for KL term (start small, e.g. 1e-3)

    Returns:
        total, recon_loss, kl_loss  (all scalar tensors)
    """
    recon = F.mse_loss(x_hat, x, reduction="mean")
    # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon + beta * kl
    return total, recon, kl


if __name__ == "__main__":
    model = SpectrogramVAE()
    dummy = torch.randn(4, 1, 128, 128)
    x_hat, mu, logvar = model(dummy)
    print(f"x_hat shape : {x_hat.shape}")    # (4, 1, 128, 128)
    print(f"mu shape    : {mu.shape}")        # (4, 128)
    total, recon, kl = vae_loss(x_hat, dummy, mu, logvar)
    print(f"loss total={total:.4f}  recon={recon:.4f}  kl={kl:.4f}")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {params:,}")
