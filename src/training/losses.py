"""
losses.py — VAE Loss for Biomedical Sound Separation
─────────────────────────────────────────────────────
Experiment 2 changes vs Experiment 1:
  1. spectral_convergence() added  — penalizes frequency smearing
  2. reconstruction_loss()  updated — heart gets 2x weight + SC term
  3. kl_divergence_loss()   unchanged — free bits kept
  4. KLAnnealer             unchanged — cosine schedule kept
  5. VAELoss.__init__       unchanged — trainer.py needs no changes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple


# ── Spectral Convergence ──────────────────────────────────────────────────────

def spectral_convergence(
    recon  : torch.Tensor,
    target : torch.Tensor,
    eps    : float = 1e-8,
) -> torch.Tensor:
    """
    Spectral Convergence Loss.

    Measures how well the frequency structure of the reconstruction
    matches the target. Penalizes frequency smearing (blurry spectrograms).

    SC = ||target - recon||_F / ||target||_F

    Args:
        recon  : Reconstructed spectrogram [B, 1, F, T]
        target : Target spectrogram        [B, 1, F, T]
        eps    : Stability term

    Returns:
        Scalar spectral convergence loss.

    Why this helps:
        MSE averages over all pixels equally — it will happily produce
        a blurry mean that minimizes pixel error but loses structure.
        Spectral convergence operates on the Frobenius norm of the full
        spectrogram, which penalizes structural mismatch more strongly.
    """
    # Operate on the 2D spectrogram: squeeze channel dim → [B, F, T]
    recon_2d  = recon.squeeze(1)
    target_2d = target.squeeze(1)

    diff  = torch.norm(target_2d - recon_2d, p="fro", dim=(-2, -1))
    denom = torch.norm(target_2d,            p="fro", dim=(-2, -1)) + eps

    return (diff / denom).mean()


# ── Reconstruction Loss ───────────────────────────────────────────────────────

def reconstruction_loss(
    heart_recon  : torch.Tensor,
    lung_recon   : torch.Tensor,
    heart_target : torch.Tensor,
    lung_target  : torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Weighted reconstruction loss for heart and lung spectrograms.

    Components:
        MSE  — pixel-level accuracy
        L1   — sharpness / edge preservation
        SC   — spectral structure / frequency accuracy

    Heart gets 2x weight because:
        - Heart sounds are sparse transients (harder to reconstruct)
        - MSE alone predicts the "average" → flat blob output
        - Extra weight forces the model to focus on heart structure

    Args:
        heart_recon  : Reconstructed heart spectrogram [B, 1, F, T]
        lung_recon   : Reconstructed lung spectrogram  [B, 1, F, T]
        heart_target : Ground truth heart spectrogram  [B, 1, F, T]
        lung_target  : Ground truth lung spectrogram   [B, 1, F, T]

    Returns:
        Tuple of (total_recon_loss, heart_loss, lung_loss)
    """
    # ── MSE component — pixel-level accuracy ──────────────────────
    heart_mse = F.mse_loss(heart_recon, heart_target)
    lung_mse  = F.mse_loss(lung_recon,  lung_target)

    # ── L1 component — sharpness, preserves transient edges ───────
    heart_l1  = F.l1_loss(heart_recon, heart_target)
    lung_l1   = F.l1_loss(lung_recon,  lung_target)

    # ── Spectral convergence — penalizes frequency smearing ───────
    heart_sc  = spectral_convergence(heart_recon, heart_target)
    lung_sc   = spectral_convergence(lung_recon,  lung_target)

    # ── Combine: heart gets 2x weight ─────────────────────────────
    heart_loss = 2.0 * (heart_mse + 0.5 * heart_l1 + 0.1 * heart_sc)
    lung_loss  = 1.0 * (lung_mse  + 0.5 * lung_l1  + 0.1 * lung_sc)

    total_recon = heart_loss + lung_loss

    return total_recon, heart_loss, lung_loss


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

    Without free bits: encoder collapses to N(0,1) → posterior collapse
    With free bits:    encoder keeps at least free_bits info per dim

    Args:
        mu        : Latent mean          [B, latent_dim]
        logvar    : Latent log-variance  [B, latent_dim]
        free_bits : Minimum KL per dimension (nats). Default 0.5.

    Returns:
        Scalar KL divergence loss.
    """
    # Per-dimension KL: [B, latent_dim]
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

    # Clamp: only penalize dimensions above the free_bits threshold
    # Below threshold → no gradient → encoder stays active
    kl_free = torch.clamp(kl_per_dim, min=free_bits)

    return kl_free.mean()


# ── KL Annealing Scheduler ────────────────────────────────────────────────────

class KLAnnealer:
    """
    Cosine KL annealing schedule.
    Smoothly ramps KL weight from 0 → max_weight over warmup_epochs.

    Why cosine over linear:
        Linear ramp can cause a sudden jump when KL weight becomes
        significant. Cosine starts very slowly, accelerates through
        the middle, then slows again — much more stable.

    Args:
        warmup_epochs : Epochs to ramp from 0 → max_weight.
        max_weight    : Final KL weight. Use 0.01 for audio VAEs.
    """

    def __init__(
        self,
        warmup_epochs : int   = 30,
        max_weight    : float = 0.01,
    ):
        self.warmup_epochs = warmup_epochs
        self.max_weight    = max_weight

    def get_weight(self, epoch: int) -> float:
        """
        Get KL weight for current epoch.

        Args:
            epoch: Current training epoch (0-indexed).

        Returns:
            Float KL weight in [0, max_weight].
        """
        if epoch >= self.warmup_epochs:
            return self.max_weight

        progress = epoch / self.warmup_epochs
        cosine   = (1 - math.cos(math.pi * progress)) / 2
        return self.max_weight * cosine


# ── Combined VAE Loss ─────────────────────────────────────────────────────────

class VAELoss(nn.Module):
    """
    Combined VAE loss for biomedical sound separation.

    Loss = recon_weight * reconstruction_loss + kl_weight(epoch) * kl_loss

    Reconstruction loss:
        MSE + L1 + SpectralConvergence
        Heart gets 2x weight (sparse transients are harder)

    KL loss:
        Free bits divergence — prevents posterior collapse

    KL schedule:
        Cosine annealing — prevents epoch-1 KL explosion

    Args:
        warmup_epochs : KL annealing warmup period.      Default: 30
        max_kl_weight : Maximum KL weight after warmup.  Default: 0.01
        recon_weight  : Weight for reconstruction loss.  Default: 1.0
        free_bits     : Min KL per latent dim.           Default: 0.5

    Usage:
        criterion = VAELoss(
            warmup_epochs = 30,
            max_kl_weight = 0.01,
            recon_weight  = 1.0,
            free_bits     = 0.5,
        )
        loss_dict = criterion(output, heart_target, lung_target, epoch)
        loss_dict["total"].backward()
    """

    def __init__(
        self,
        warmup_epochs : int   = 30,
        max_kl_weight : float = 0.01,
        recon_weight  : float = 1.0,
        free_bits     : float = 0.5,
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
        Compute the full VAE loss.

        Args:
            output : Model output dict containing:
                       heart_recon [B, 1, F, T]
                       lung_recon  [B, 1, F, T]
                       mu          [B, latent_dim]
                       logvar      [B, latent_dim]
            heart  : Ground truth heart spectrogram [B, 1, F, T]
            lung   : Ground truth lung spectrogram  [B, 1, F, T]
            epoch  : Current epoch for KL annealing (0-indexed)

        Returns:
            Dictionary:
                total       : Total loss scalar ← use for .backward()
                recon       : Total reconstruction loss
                recon_heart : Heart reconstruction loss (2x weighted)
                recon_lung  : Lung reconstruction loss
                kl          : KL divergence loss (free bits)
                kl_weight   : Current KL annealing weight
        """
        # ── Reconstruction ────────────────────────────────────────
        recon, recon_heart, recon_lung = reconstruction_loss(
            heart_recon  = output["heart_recon"],
            lung_recon   = output["lung_recon"],
            heart_target = heart,
            lung_target  = lung,
        )

        # ── KL divergence ─────────────────────────────────────────
        kl        = kl_divergence_loss(output["mu"], output["logvar"], self.free_bits)

        # ── KL annealing weight ───────────────────────────────────
        kl_weight = self.annealer.get_weight(epoch)

        # ── Total loss ────────────────────────────────────────────
        total = self.recon_weight * recon + kl_weight * kl

        return {
            "total"       : total,
            "recon"       : recon,
            "recon_heart" : recon_heart,
            "recon_lung"  : recon_lung,
            "kl"          : kl,
            "kl_weight"   : kl_weight,
        }