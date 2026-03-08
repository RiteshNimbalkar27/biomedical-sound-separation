"""
losses.py
─────────────────────────────────────────────────────────────────
Loss functions for the Biomedical Sound Separation VAE.

Total VAE Loss:
    L = reconstruction_loss + kl_weight * kl_loss

Where:
    reconstruction_loss = MSE(heart_recon, heart) 
                        + MSE(lung_recon, lung)
    kl_loss             = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_weight           = annealed from 0 → 1 over warmup epochs

KL Annealing:
    Prevents posterior collapse by slowly introducing the KL
    penalty. Without this, the VAE often ignores the latent
    space entirely (all outputs look the same).
─────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Individual Loss Terms ─────────────────────────────────────────────────────

def reconstruction_loss(
    heart_recon: torch.Tensor,
    lung_recon: torch.Tensor,
    heart_target: torch.Tensor,
    lung_target: torch.Tensor,
    reduction: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute reconstruction loss for both heart and lung outputs.

    Uses Mean Squared Error (MSE) between reconstructed and
    target mel-spectrograms.

    Args:
        heart_recon  : Reconstructed heart spectrogram [B, 1, F, T]
        lung_recon   : Reconstructed lung spectrogram  [B, 1, F, T]
        heart_target : Ground truth heart spectrogram  [B, 1, F, T]
        lung_target  : Ground truth lung spectrogram   [B, 1, F, T]
        reduction    : 'mean' or 'sum'

    Returns:
        Tuple of (total_recon_loss, heart_loss, lung_loss)
    """
    heart_loss = F.mse_loss(heart_recon, heart_target, reduction=reduction)
    lung_loss  = F.mse_loss(lung_recon,  lung_target,  reduction=reduction)
    total      = heart_loss + lung_loss

    return total, heart_loss, lung_loss


def kl_divergence_loss(
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    """
    Compute KL divergence between the learned latent distribution
    q(z|x) = N(mu, sigma^2) and the prior p(z) = N(0, I).

    Closed-form formula:
        KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))

    Normalized by batch size for stable loss scaling.

    Args:
        mu     : Latent mean          [B, latent_dim]
        logvar : Latent log-variance  [B, latent_dim]

    Returns:
        Scalar KL divergence loss.
    """
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Normalize by batch size
    kl = kl / mu.size(0)

    return kl


# ── KL Annealing Scheduler ────────────────────────────────────────────────────

class KLAnnealer:
    """
    Gradually increases the KL loss weight from 0 to max_weight
    over a specified number of warmup epochs.

    This prevents posterior collapse — a common failure mode in
    VAEs where the model ignores the latent space entirely.

    Annealing schedule: linear ramp from 0 → max_weight

    Args:
        warmup_epochs : Number of epochs to ramp up KL weight.
        max_weight    : Final KL weight after warmup. Default 1.0.
        start_epoch   : Epoch to start annealing from. Default 0.

    Usage:
        annealer = KLAnnealer(warmup_epochs=20)
        for epoch in range(100):
            kl_weight = annealer.get_weight(epoch)
            loss = recon_loss + kl_weight * kl_loss
    """

    def __init__(
        self,
        warmup_epochs: int = 20,
        max_weight: float  = 1.0,
        start_epoch: int   = 0,
    ):
        self.warmup_epochs = warmup_epochs
        self.max_weight    = max_weight
        self.start_epoch   = start_epoch

    def get_weight(self, epoch: int) -> float:
        """
        Get the KL weight for the current epoch.

        Args:
            epoch: Current training epoch (0-indexed).

        Returns:
            Float KL weight in [0, max_weight].
        """
        if epoch < self.start_epoch:
            return 0.0

        adjusted = epoch - self.start_epoch

        if adjusted >= self.warmup_epochs:
            return self.max_weight

        # Linear ramp
        return self.max_weight * (adjusted / self.warmup_epochs)


# ── Combined VAE Loss ─────────────────────────────────────────────────────────

class VAELoss(nn.Module):
    """
    Combined VAE loss for biomedical sound separation.

    Wraps reconstruction loss + KL divergence + annealing
    into a single callable module.

    Args:
        warmup_epochs : KL annealing warmup period.
        max_kl_weight : Maximum KL loss weight.
        recon_weight  : Weight for reconstruction loss. Default 1.0.

    Usage:
        criterion = VAELoss(warmup_epochs=20)

        output = model(mixed)
        loss_dict = criterion(
            output    = output,
            heart     = heart_target,
            lung      = lung_target,
            epoch     = current_epoch,
        )
        loss_dict['total'].backward()
    """

    def __init__(
        self,
        warmup_epochs: int  = 20,
        max_kl_weight: float = 1.0,
        recon_weight: float  = 1.0,
    ):
        super().__init__()
        self.recon_weight = recon_weight
        self.annealer     = KLAnnealer(
            warmup_epochs = warmup_epochs,
            max_weight    = max_kl_weight,
        )

    def forward(
        self,
        output: dict,
        heart: torch.Tensor,
        lung: torch.Tensor,
        epoch: int,
    ) -> dict[str, torch.Tensor]:
        """
        Compute the full VAE loss.

        Args:
            output : Dict from model forward pass containing
                     heart_recon, lung_recon, mu, logvar
            heart  : Ground truth heart spectrogram [B, 1, F, T]
            lung   : Ground truth lung spectrogram  [B, 1, F, T]
            epoch  : Current epoch for KL annealing

        Returns:
            Dictionary with:
                total       : Total loss (scalar) ← use for .backward()
                recon       : Total reconstruction loss
                recon_heart : Heart reconstruction loss
                recon_lung  : Lung reconstruction loss
                kl          : KL divergence loss
                kl_weight   : Current KL annealing weight
        """
        # ── Reconstruction loss ───────────────────────────────
        recon, recon_heart, recon_lung = reconstruction_loss(
            heart_recon  = output["heart_recon"],
            lung_recon   = output["lung_recon"],
            heart_target = heart,
            lung_target  = lung,
        )

        # ── KL divergence ─────────────────────────────────────
        kl = kl_divergence_loss(output["mu"], output["logvar"])

        # ── KL annealing weight ───────────────────────────────
        kl_weight = self.annealer.get_weight(epoch)

        # ── Total loss ────────────────────────────────────────
        total = self.recon_weight * recon + kl_weight * kl

        return {
            "total"       : total,
            "recon"       : recon,
            "recon_heart" : recon_heart,
            "recon_lung"  : recon_lung,
            "kl"          : kl,
            "kl_weight"   : kl_weight,
        }