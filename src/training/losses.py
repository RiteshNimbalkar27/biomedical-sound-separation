"""
losses.py — VAE Loss with Free Bits + Cosine KL Annealing
Keeps original __init__ signature so trainer.py needs no changes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple


# ── Reconstruction Loss ───────────────────────────────────────────────────────

def reconstruction_loss(
    heart_recon  : torch.Tensor,
    lung_recon   : torch.Tensor,
    heart_target : torch.Tensor,
    lung_target  : torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MSE + L1 reconstruction loss for heart and lung spectrograms."""
    heart_mse = F.mse_loss(heart_recon, heart_target)
    lung_mse  = F.mse_loss(lung_recon,  lung_target)
    heart_l1  = F.l1_loss(heart_recon,  heart_target)
    lung_l1   = F.l1_loss(lung_recon,   lung_target)

    recon_heart = heart_mse + 0.5 * heart_l1
    recon_lung  = lung_mse  + 0.5 * lung_l1
    total_recon = recon_heart + recon_lung

    return total_recon, recon_heart, recon_lung


# ── KL Divergence with Free Bits ──────────────────────────────────────────────

def kl_divergence_loss(
    mu        : torch.Tensor,
    logvar    : torch.Tensor,
    free_bits : float = 0.5,
) -> torch.Tensor:
    """
    KL divergence with free bits to prevent posterior collapse.

    Free bits guarantee the encoder uses at least `free_bits` nats
    per latent dimension — preventing the decoder from ignoring z.
    """
    # Per-dimension KL: [B, latent_dim]
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

    # Clamp: only penalize above free_bits threshold
    kl_free = torch.clamp(kl_per_dim, min=free_bits)

    return kl_free.mean()


# ── KL Annealing Scheduler ────────────────────────────────────────────────────

class KLAnnealer:
    """
    Cosine KL annealing schedule.
    Smoothly ramps KL weight from 0 → max_weight over warmup_epochs.
    """
    def __init__(
        self,
        warmup_epochs : int   = 30,
        max_weight    : float = 0.01,
    ):
        self.warmup_epochs = warmup_epochs
        self.max_weight    = max_weight

    def get_weight(self, epoch: int) -> float:
        if epoch >= self.warmup_epochs:
            return self.max_weight
        progress = epoch / self.warmup_epochs
        cosine   = (1 - math.cos(math.pi * progress)) / 2
        return self.max_weight * cosine


# ── Combined VAE Loss ─────────────────────────────────────────────────────────

class VAELoss(nn.Module):
    """
    Combined VAE loss for biomedical sound separation.

    Args:
        warmup_epochs : KL annealing warmup period.
        max_kl_weight : Maximum KL weight after warmup. Use 0.01 for audio.
        recon_weight  : Weight for reconstruction loss.
        free_bits     : Minimum KL per latent dim (prevents collapse).
    """

    def __init__(
        self,
        warmup_epochs : int   = 30,
        max_kl_weight : float = 0.01,   # ← was 1.0, now 0.01 for audio
        recon_weight  : float = 1.0,
        free_bits     : float = 0.5,    # ← new: prevents posterior collapse
    ):
        super().__init__()
        self.recon_weight = recon_weight
        self.free_bits    = free_bits
        self.annealer     = KLAnnealer(
            warmup_epochs = warmup_epochs,
            max_weight    = max_kl_weight,
        )

    def forward(
        self,
        output : dict,
        heart  : torch.Tensor,
        lung   : torch.Tensor,
        epoch  : int,
    ) -> dict:
        """
        Args:
            output : Model output dict with heart_recon, lung_recon, mu, logvar
            heart  : Ground truth heart spectrogram [B, 1, F, T]
            lung   : Ground truth lung spectrogram  [B, 1, F, T]
            epoch  : Current epoch for KL annealing

        Returns:
            Dict with total, recon, recon_heart, recon_lung, kl, kl_weight
        """
        recon, recon_heart, recon_lung = reconstruction_loss(
            heart_recon  = output["heart_recon"],
            lung_recon   = output["lung_recon"],
            heart_target = heart,
            lung_target  = lung,
        )

        kl        = kl_divergence_loss(output["mu"], output["logvar"], self.free_bits)
        kl_weight = self.annealer.get_weight(epoch)
        total     = self.recon_weight * recon + kl_weight * kl

        return {
            "total"       : total,
            "recon"       : recon,
            "recon_heart" : recon_heart,
            "recon_lung"  : recon_lung,
            "kl"          : kl,
            "kl_weight"   : kl_weight,
        }