"""
vae.py
─────────────────────────────────────────────────────────────────
Variational Autoencoder for Biomedical Sound Separation.

Architecture:
    One shared encoder   → encodes mixed spectrogram to (mu, logvar)
    Two separate decoders → one for heart, one for lung

The shared latent space forces the model to learn a joint
representation that captures information about both sources
simultaneously, which the two decoders then disentangle.

Input  : Mixed mel-spectrogram  [B, 1, 64, 94]
Outputs: Heart mel-spectrogram  [B, 1, 64, 94]
         Lung  mel-spectrogram  [B, 1, 64, 94]
─────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
from src.models.components.encoder import ConvEncoder
from src.models.components.decoder import ConvDecoder


class BioSoundVAE(nn.Module):
    """
    Biomedical Sound Separation VAE.

    One encoder, two decoders (heart + lung).
    Uses the reparameterization trick for differentiable sampling.

    Args:
        latent_dim    : Size of the latent space.
        base_channels : Base number of conv filters.
        input_size    : (H, W) of input spectrogram (n_mels, T).
    """

    def __init__(
        self,
        latent_dim: int    = 128,
        base_channels: int = 32,
        input_size: tuple  = (64, 94),
    ):
        super().__init__()

        self.latent_dim = latent_dim

        # ── Shared Encoder ────────────────────────────────────
        self.encoder = ConvEncoder(
            in_channels   = 1,
            latent_dim    = latent_dim,
            base_channels = base_channels,
        )

        # ── Dual Decoders ─────────────────────────────────────
        self.heart_decoder = ConvDecoder(
            latent_dim    = latent_dim,
            out_channels  = 1,
            base_channels = base_channels,
            output_size   = input_size,
        )

        self.lung_decoder = ConvDecoder(
            latent_dim    = latent_dim,
            out_channels  = 1,
            base_channels = base_channels,
            output_size   = input_size,
        )

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Reparameterization trick:
            z = mu + epsilon * std
            epsilon ~ N(0, I)

        Allows gradients to flow through the sampling operation.

        Args:
            mu     : Mean of latent distribution     [B, latent_dim]
            logvar : Log-variance of latent dist.    [B, latent_dim]

        Returns:
            z: Sampled latent vector                 [B, latent_dim]
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # At inference time, just use the mean (no randomness)
            return mu

    def forward(
        self,
        x: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Full forward pass.

        Args:
            x: Mixed spectrogram [B, 1, n_mels, T]

        Returns:
            Dictionary with:
                heart_recon : Reconstructed heart spec [B, 1, n_mels, T]
                lung_recon  : Reconstructed lung spec  [B, 1, n_mels, T]
                mu          : Latent mean              [B, latent_dim]
                logvar      : Latent log-variance      [B, latent_dim]
                z           : Sampled latent vector    [B, latent_dim]
        """
        # ── Encode ────────────────────────────────────────────
        mu, logvar = self.encoder(x)

        # ── Sample ────────────────────────────────────────────
        z = self.reparameterize(mu, logvar)

        # ── Decode ────────────────────────────────────────────
        heart_recon = self.heart_decoder(z)
        lung_recon  = self.lung_decoder(z)

        return {
            "heart_recon" : heart_recon,
            "lung_recon"  : lung_recon,
            "mu"          : mu,
            "logvar"      : logvar,
            "z"           : z,
        }

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode only — returns (mu, logvar)."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode only — returns (heart_recon, lung_recon)."""
        return self.heart_decoder(z), self.lung_decoder(z)

    def sample(self, n: int, device: torch.device) -> tuple:
        """
        Generate n random samples from the prior N(0, I).
        Useful for visualizing what the model has learned.

        Args:
            n      : Number of samples to generate.
            device : torch device.

        Returns:
            Tuple of (heart_spectrograms, lung_spectrograms)
        """
        z = torch.randn(n, self.latent_dim).to(device)
        return self.decode(z)