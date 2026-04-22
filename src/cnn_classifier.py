"""
Phase 1.3 — CNN Classifier
5-block convolutional network: log-mel-spectrogram → 50 class logits.
Input shape : (B, 1, 128, T)  — T is variable (AdaptiveAvgPool handles it)
Output shape: (B, 50)
"""

import torch
import torch.nn as nn


def _conv_block(in_ch, out_ch, pool=True):
    layers = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)


class CNNClassifier(nn.Module):
    """
    Architecture:
        Block 1: Conv(1  →  32) + BN + ReLU + MaxPool  → (B, 32,  64, T/2)
        Block 2: Conv(32 →  64) + BN + ReLU + MaxPool  → (B, 64,  32, T/4)
        Block 3: Conv(64 → 128) + BN + ReLU + MaxPool  → (B,128,  16, T/8)
        Block 4: Conv(128→ 256) + BN + ReLU + MaxPool  → (B,256,   8, T/16)
        Block 5: Conv(256→ 256) + BN + ReLU            → (B,256,   8, T/16)
        AdaptiveAvgPool(1,1)                            → (B,256,   1, 1)
        Linear(256, num_classes)                        → (B, 50)
    """

    def __init__(self, num_classes: int = 50, dropout: float = 0.5):
        super().__init__()
        self.blocks = nn.Sequential(
            _conv_block(1,    32, pool=True),
            _conv_block(32,   64, pool=True),
            _conv_block(64,  128, pool=True),
            _conv_block(128, 256, pool=True),
            _conv_block(256, 256, pool=False),  # no pool on last block
        )
        self.pool    = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(256, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, 1, 128, T)
        x = self.blocks(x)          # (B, 256, 8, T')
        x = self.pool(x)            # (B, 256, 1, 1)
        x = x.flatten(1)            # (B, 256)
        x = self.dropout(x)
        return self.head(x)         # (B, 50)

    def feature_extract(self, x):
        """Return 256-dim embedding before the classifier head."""
        x = self.blocks(x)
        x = self.pool(x)
        return x.flatten(1)


if __name__ == "__main__":
    model = CNNClassifier()
    dummy = torch.randn(4, 1, 128, 216)
    out   = model(dummy)
    print(f"CNNClassifier output: {out.shape}")  # expect (4, 50)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {params:,}")
