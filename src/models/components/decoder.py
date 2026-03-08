"""
decoder.py
─────────────────────────────────────────────────────────────────
Convolutional decoder for the Biomedical Sound Separation VAE.

Takes a latent vector z and reconstructs a mel-spectrogram.
Used twice in the VAE — once for heart, once for lung.

Input shape  : [B, latent_dim]
Output shape : [B, 1, n_mels, T]  e.g. [16, 1, 64, 94]
─────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn


class ConvDecoder(nn.Module):
    """
    Convolutional decoder that maps a latent vector back to a
    mel-spectrogram.

    Architecture:
        Linear projection → reshape
        4 × ConvTranspose2d blocks (each: ConvT → BatchNorm → ReLU)
        Final conv → Sigmoid (output in [0, 1])

    Args:
        latent_dim    : Dimensionality of the latent space.
        out_channels  : Output channels (1 for mono spectrogram).
        base_channels : Number of filters (mirrors encoder).
        output_size   : Target output (H, W) = (n_mels, T).
    """

    def __init__(
        self,
        latent_dim: int     = 128,
        out_channels: int   = 1,
        base_channels: int  = 32,
        output_size: tuple  = (64, 94),
    ):
        super().__init__()

        self.output_size   = output_size
        self.base_channels = base_channels

        # Starting spatial size before upsampling
        # Mirrors encoder: pool output was [B, 256, 2, 2]
        self.start_h = 2
        self.start_w = 2
        self.start_ch = base_channels * 8   # 256

        # ── Project latent vector → feature map ───────────────
        self.fc = nn.Linear(
            latent_dim,
            self.start_ch * self.start_h * self.start_w  # 256*2*2=1024
        )

        # ── Upsampling blocks ──────────────────────────────────
        # Mirror of encoder in reverse
        # [B, 256, 2,  2] → [B, 128, 4,  4]
        # [B, 128, 4,  4] → [B,  64, 8,  8]
        # [B,  64, 8,  8] → [B,  32,16, 16]
        # [B,  32,16, 16] → [B,  16,32, 32]
        self.deconv_blocks = nn.Sequential(
            self._deconv_block(base_channels * 8, base_channels * 4),
            self._deconv_block(base_channels * 4, base_channels * 2),
            self._deconv_block(base_channels * 2, base_channels),
            self._deconv_block(base_channels,     base_channels // 2),
        )

        # ── Final conv to output channels ─────────────────────
        self.final_conv = nn.Conv2d(
            base_channels // 2, out_channels,
            kernel_size=3, stride=1, padding=1,
        )

        # ── Adaptive upsample to exact target size ────────────
        # Ensures output is exactly [B, 1, 64, 94]
        # regardless of rounding in ConvTranspose2d
        self.upsample = nn.Upsample(
            size=output_size, mode="bilinear", align_corners=False
        )

        # Sigmoid: output values in [0, 1] to match normalized input
        self.sigmoid = nn.Sigmoid()

    def _deconv_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        """Single deconv block: ConvTranspose2d → BatchNorm2d → ReLU."""
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_ch, out_ch,
                kernel_size=4, stride=2, padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent vector [B, latent_dim]

        Returns:
            Reconstructed spectrogram [B, 1, n_mels, T]
        """
        x = self.fc(z)                                          # [B, 1024]
        x = x.view(x.size(0), self.start_ch,
                   self.start_h, self.start_w)                  # [B, 256, 2, 2]
        x = self.deconv_blocks(x)                               # [B, 16, 32, 32]
        x = self.final_conv(x)                                  # [B, 1,  32, 32]
        x = self.upsample(x)                                    # [B, 1,  64, 94]
        return self.sigmoid(x)