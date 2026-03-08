"""
encoder.py
─────────────────────────────────────────────────────────────────
Convolutional encoder for the Biomedical Sound Separation VAE.

Takes a mel-spectrogram and encodes it into a latent
distribution (mu, logvar) for the reparameterization trick.

Input shape  : [B, 1, n_mels, T]  e.g. [16, 1, 64, 94]
Output shape : mu     [B, latent_dim]
               logvar [B, latent_dim]
─────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    """
    Convolutional encoder that maps a mel-spectrogram to a
    latent distribution (mu, logvar).

    Architecture:
        4 × Conv2d blocks (each: Conv → BatchNorm → LeakyReLU)
        Adaptive average pool → flatten
        Two linear heads: mu and logvar

    Args:
        in_channels : Input channels (1 for mono spectrogram).
        latent_dim  : Dimensionality of the latent space.
        base_channels: Number of filters in first conv layer.
                       Doubles at each subsequent layer.
    """

    def __init__(
        self,
        in_channels: int   = 1,
        latent_dim: int    = 128,
        base_channels: int = 32,
    ):
        super().__init__()

        self.latent_dim = latent_dim

        # ── Convolutional blocks ──────────────────────────────
        # Each block halves the spatial dimensions via stride=2
        # Channels: 1 → 32 → 64 → 128 → 256
        self.conv_blocks = nn.Sequential(

            # Block 1: [B, 1,   64, 94] → [B, 32,  32, 47]
            self._conv_block(in_channels,         base_channels),

            # Block 2: [B, 32,  32, 47] → [B, 64,  16, 24]
            self._conv_block(base_channels,       base_channels * 2),

            # Block 3: [B, 64,  16, 24] → [B, 128,  8, 12]
            self._conv_block(base_channels * 2,   base_channels * 4),

            # Block 4: [B, 128,  8, 12] → [B, 256,  4,  6]
            self._conv_block(base_channels * 4,   base_channels * 8),
        )

        # ── Adaptive pooling → fixed size regardless of input ─
        # Output: [B, 256, 4, 6] → [B, 256, 2, 2]
        self.pool = nn.AdaptiveAvgPool2d((2, 2))

        # ── Flatten size: 256 * 2 * 2 = 1024 ─────────────────
        self.flat_dim = base_channels * 8 * 2 * 2

        # ── Latent heads ──────────────────────────────────────
        self.fc_mu     = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        """Single conv block: Conv2d → BatchNorm2d → LeakyReLU."""
        return nn.Sequential(
            nn.Conv2d(
                in_ch, out_ch,
                kernel_size=4, stride=2, padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input spectrogram [B, 1, n_mels, T]

        Returns:
            mu     : Mean of latent distribution     [B, latent_dim]
            logvar : Log-variance of latent dist.    [B, latent_dim]
        """
        x = self.conv_blocks(x)     # [B, 256, 4, 6]
        x = self.pool(x)            # [B, 256, 2, 2]
        x = x.view(x.size(0), -1)  # [B, 1024]

        mu     = self.fc_mu(x)      # [B, latent_dim]
        logvar = self.fc_logvar(x)  # [B, latent_dim]

        return mu, logvar