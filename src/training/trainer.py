"""
trainer.py
─────────────────────────────────────────────────────────────────
Training loop for the Biomedical Sound Separation VAE.

Responsibilities:
    - Training and validation loops
    - Checkpoint saving and loading
    - Loss tracking and logging
    - Early stopping
    - Learning rate scheduling

Usage:
    python -m src.training.trainer
    — or —
    python -c "
    from src.training.trainer import VAETrainer
    trainer = VAETrainer(config_path='configs/base_config.yaml', experiment='vae_exp2')
    trainer.train()
    "
─────────────────────────────────────────────────────────────────
"""

import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger
from src.data.dataset import get_all_dataloaders
from src.models.vae import BioSoundVAE
from src.training.losses import VAELoss

logger = get_logger(__name__)


# ── Early Stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stop training when validation loss stops improving.

    Args:
        patience  : Epochs to wait before stopping.
        min_delta : Minimum improvement to count as progress.
    """

    def __init__(self, patience: int = 25, min_delta: float = 1e-4):
        self.patience    = patience
        self.min_delta   = min_delta
        self.counter     = 0
        self.best_loss   = float("inf")
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        """
        Call after each epoch.
        Returns True if training should stop.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            logger.info(
                f"EarlyStopping: no improvement for "
                f"{self.counter}/{self.patience} epochs"
            )
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


# ── Trainer ───────────────────────────────────────────────────────────────────

class VAETrainer:
    """
    Full training pipeline for the Biomedical Sound Separation VAE.

    Handles:
        - Device placement (GPU/CPU)
        - DataLoader setup
        - Optimizer and scheduler
        - Training + validation loops
        - Checkpoint saving/loading
        - Early stopping

    Args:
        config_path : Path to YAML config file.
        experiment  : Experiment name for saving checkpoints/logs.
    """

    def __init__(
        self,
        config_path: str = "configs/base_config.yaml",
        experiment: str  = "vae_baseline",
    ):
        self.cfg        = load_config(config_path)
        self.experiment = experiment

        # ── Device ────────────────────────────────────────────
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Device : {self.device}")

        if self.device.type == "cuda":
            logger.info(
                f"GPU    : {torch.cuda.get_device_name(0)} | "
                f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            )

        # ── Experiment directory ───────────────────────────────
        self.exp_dir  = resolve_path("experiments") / experiment
        self.ckpt_dir = self.exp_dir / "checkpoints"
        self.log_dir  = self.exp_dir / "logs"

        for d in [self.ckpt_dir, self.log_dir]:
            d.mkdir(parents=True, exist_ok=True)

        logger.info(f"Experiment dir : {self.exp_dir}")

        # ── Model ─────────────────────────────────────────────
        # latent_dim now reads from config (256 for exp2, 128 for baseline)
        self.model = BioSoundVAE(
            latent_dim    = self.cfg.model.latent_dim,
            base_channels = self.cfg.model.base_channels,   # ← reads from config
            input_size    = (self.cfg.audio.n_mels, 94),
        ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model params   : {total_params:,}")
        logger.info(f"Latent dim     : {self.cfg.model.latent_dim}")

        # ── Loss ──────────────────────────────────────────────
        # All values driven by config — no hardcoding
        self.criterion = VAELoss(
            warmup_epochs = self.cfg.loss.kl_warmup_epochs,
            max_kl_weight = self.cfg.loss.kl_weight_max,
            recon_weight  = self.cfg.loss.recon_weight,
            free_bits     = self.cfg.loss.free_bits,
        )

        # ── Optimizer ─────────────────────────────────────────
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr           = self.cfg.training.learning_rate,
            weight_decay = self.cfg.training.weight_decay,  # ← reads from config
        )

        # ── LR Scheduler ──────────────────────────────────────
        # Reduce LR when val loss plateaus
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode     = "min",
            factor   = self.cfg.training.scheduler_factor,    # ← reads from config
            patience = self.cfg.training.scheduler_patience,  # ← reads from config
        )

        # ── Early stopping ────────────────────────────────────
        # patience now reads from config (25 for exp2, was 15)
        self.early_stopping = EarlyStopping(
            patience  = self.cfg.training.early_stopping_patience,   # ← reads from config
            min_delta = self.cfg.training.early_stopping_min_delta,  # ← reads from config
        )

        # ── DataLoaders ───────────────────────────────────────
        logger.info("Loading dataloaders...")
        self.loaders = get_all_dataloaders(config_path)

        # ── Training state ────────────────────────────────────
        self.start_epoch   = 0
        self.best_val_loss = float("inf")
        self.history = {
            "train_loss"  : [], "val_loss"   : [],
            "train_recon" : [], "val_recon"  : [],
            "train_kl"    : [], "val_kl"     : [],
        }

    # ── Training Loop ─────────────────────────────────────────────────────────

    def train(self):
        """Run the full training loop."""

        logger.info("=" * 55)
        logger.info("  Starting Training")
        logger.info(f"  Epochs     : {self.cfg.training.num_epochs}")
        logger.info(f"  Batch size : {self.cfg.training.batch_size}")
        logger.info(f"  LR         : {self.cfg.training.learning_rate}")
        logger.info(f"  Latent dim : {self.cfg.model.latent_dim}")
        logger.info(f"  KL warmup  : {self.cfg.loss.kl_warmup_epochs} epochs")
        logger.info(f"  ES patience: {self.cfg.training.early_stopping_patience}")
        logger.info("=" * 55)

        for epoch in range(self.start_epoch, self.cfg.training.num_epochs):

            # ── Train one epoch ───────────────────────────────
            train_metrics = self._train_epoch(epoch)

            # ── Validate ──────────────────────────────────────
            val_metrics = self._val_epoch(epoch)

            # ── LR scheduler step ─────────────────────────────
            self.scheduler.step(val_metrics["total"])

            # ── Log epoch summary ─────────────────────────────
            self._log_epoch(epoch, train_metrics, val_metrics)

            # ── Save history ──────────────────────────────────
            self.history["train_loss"].append(train_metrics["total"])
            self.history["val_loss"].append(val_metrics["total"])
            self.history["train_recon"].append(train_metrics["recon"])
            self.history["val_recon"].append(val_metrics["recon"])
            self.history["train_kl"].append(train_metrics["kl"])
            self.history["val_kl"].append(val_metrics["kl"])

            # ── Save best checkpoint ──────────────────────────
            if val_metrics["total"] < self.best_val_loss:
                self.best_val_loss = val_metrics["total"]
                self._save_checkpoint(epoch, val_metrics["total"], is_best=True)
                logger.info(
                    f"  New best model saved "
                    f"(val_loss={self.best_val_loss:.6f})"
                )

            # ── Save periodic checkpoint every 10 epochs ──────
            if (epoch + 1) % self.cfg.training.save_every == 0:
                self._save_checkpoint(epoch, val_metrics["total"], is_best=False)

            # ── Early stopping check ──────────────────────────
            if self.early_stopping.step(val_metrics["total"]):
                logger.info(
                    f"Early stopping triggered at epoch {epoch + 1}"
                )
                break

        logger.info("=" * 55)
        logger.info("  Training Complete")
        logger.info(f"  Best val loss : {self.best_val_loss:.6f}")
        logger.info(f"  Checkpoints   : {self.ckpt_dir}")
        logger.info("=" * 55)

        self._save_history()

    # ── Single Train Epoch ────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> dict:
        """Run one training epoch. Returns averaged metrics."""
        self.model.train()

        totals = {
            "total": 0, "recon": 0,
            "recon_heart": 0, "recon_lung": 0, "kl": 0
        }
        n_batches = len(self.loaders["train"])

        pbar = tqdm(
            self.loaders["train"],
            desc=f"Epoch {epoch+1:03d} [Train]",
            leave=False,
        )

        for batch in pbar:
            mixed = batch["mixed"].to(self.device)
            heart = batch["heart"].to(self.device)
            lung  = batch["lung"].to(self.device)

            # ── Forward pass ──────────────────────────────────
            self.optimizer.zero_grad()
            output    = self.model(mixed)
            loss_dict = self.criterion(output, heart, lung, epoch)

            # ── Backward pass ─────────────────────────────────
            loss_dict["total"].backward()

            # Gradient clipping — prevents exploding gradients
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm = self.cfg.training.grad_clip_norm
            )

            self.optimizer.step()

            # ── Accumulate metrics ────────────────────────────
            for k in totals:
                if k in loss_dict:
                    v = loss_dict[k]
                    totals[k] += (v.item() if isinstance(v, torch.Tensor) else v)

            # Update progress bar
            pbar.set_postfix({
                "loss"  : f"{loss_dict['total'].item():.4f}",
                "recon" : f"{loss_dict['recon'].item():.4f}",
                "kl"    : f"{loss_dict['kl'].item():.4f}",
                "kl_w"  : f"{loss_dict['kl_weight']:.3f}",
            })

        return {k: v / n_batches for k, v in totals.items()}

    # ── Single Val Epoch ──────────────────────────────────────────────────────

    def _val_epoch(self, epoch: int) -> dict:
        """Run one validation epoch. Returns averaged metrics."""
        self.model.eval()

        totals = {
            "total": 0, "recon": 0,
            "recon_heart": 0, "recon_lung": 0, "kl": 0
        }
        n_batches = len(self.loaders["val"])

        with torch.no_grad():
            pbar = tqdm(
                self.loaders["val"],
                desc=f"           [Val]  ",
                leave=False,
            )

            for batch in pbar:
                mixed = batch["mixed"].to(self.device)
                heart = batch["heart"].to(self.device)
                lung  = batch["lung"].to(self.device)

                output    = self.model(mixed)
                loss_dict = self.criterion(output, heart, lung, epoch)

                for k in totals:
                    if k in loss_dict:
                        v = loss_dict[k]
                        totals[k] += (v.item() if isinstance(v, torch.Tensor) else v)

                pbar.set_postfix({
                    "val_loss": f"{loss_dict['total'].item():.4f}",
                })

        return {k: v / n_batches for k, v in totals.items()}

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log_epoch(
        self,
        epoch : int,
        train : dict,
        val   : dict,
    ):
        """Log a clean one-line summary for the epoch."""
        lr = self.optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch+1:03d} | "
            f"Train loss: {train['total']:.4f} "
            f"(recon={train['recon']:.4f}, kl={train['kl']:.4f}) | "
            f"Val loss: {val['total']:.4f} "
            f"(recon={val['recon']:.4f}, kl={val['kl']:.4f}) | "
            f"LR: {lr:.2e}"
        )

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def _save_checkpoint(
        self,
        epoch    : int,
        val_loss : float,
        is_best  : bool = False,
    ):
        """Save model checkpoint with full config for reproducibility."""
        state = {
            "epoch"          : epoch + 1,
            "model_state"    : self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "val_loss"       : val_loss,
            "experiment"     : self.experiment,
            "config"         : {
                "latent_dim"   : self.cfg.model.latent_dim,
                "base_channels": self.cfg.model.base_channels,
                "n_mels"       : self.cfg.audio.n_mels,
                "sample_rate"  : self.cfg.audio.sample_rate,
            },
        }

        path = (
            self.ckpt_dir / "best_model.pt"
            if is_best
            else self.ckpt_dir / f"checkpoint_epoch{epoch+1:03d}.pt"
        )

        torch.save(state, path)

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load a saved checkpoint to resume training.

        Args:
            checkpoint_path: Path to .pt checkpoint file.
        """
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.scheduler.load_state_dict(state["scheduler_state"])
        self.start_epoch   = state["epoch"]
        self.best_val_loss = state["val_loss"]

        logger.info(
            f"Resumed from epoch {self.start_epoch} "
            f"(val_loss={self.best_val_loss:.6f})"
        )

    # ── History ───────────────────────────────────────────────────────────────

    def _save_history(self):
        """Save training history to numpy file."""
        path = self.exp_dir / "training_history.npy"
        np.save(str(path), self.history)
        logger.info(f"Training history saved to {path}")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    trainer = VAETrainer(
        config_path = "configs/base_config.yaml",
        experiment  = "vae_exp2_latent256_weighted_loss",  # ← new experiment name
    )
    trainer.train()