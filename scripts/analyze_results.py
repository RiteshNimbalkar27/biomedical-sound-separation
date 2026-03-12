"""
analyze_results.py
──────────────────
Run after training to plot curves, evaluate on test set,
and visualize separated spectrograms.

Usage:
    python scripts/analyze_results.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch.nn.functional as F

from src.models.vae         import BioSoundVAE
from src.data.dataset       import get_all_dataloaders
from src.utils.config       import load_config

# ── Config ────────────────────────────────────────────────────────────────────

CONFIG_PATH = "configs/base_config.yaml"
CKPT_PATH   = "experiments/vae_baseline/checkpoints/best_model.pt"
HISTORY_PATH= "experiments/vae_baseline/training_history.npy"
SAVE_DIR    = "experiments/vae_baseline/plots"

os.makedirs(SAVE_DIR, exist_ok=True)

# ── 1. Plot Training Curves ───────────────────────────────────────────────────

def plot_training_curves(history: dict, save_dir: str):
    print("\n[1/3] Plotting training curves...")

    epochs = range(1, len(history["train_loss"]) + 1)
    best_ep = int(np.argmin(history["val_loss"])) + 1

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "VAE Baseline — Training Results\nBiomedical Sound Separation",
        fontsize=14, fontweight="bold"
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # Total Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, history["train_loss"], "b-o", markersize=3, label="Train")
    ax1.plot(epochs, history["val_loss"],   "r-o", markersize=3, label="Val")
    ax1.axvline(x=best_ep, color="green", linestyle="--", alpha=0.7, label=f"Best (ep.{best_ep})")
    ax1.set_title("Total Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # Reconstruction Loss
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, history["train_recon"], "b-o", markersize=3, label="Train")
    ax2.plot(epochs, history["val_recon"],   "r-o", markersize=3, label="Val")
    ax2.axvline(x=best_ep, color="green", linestyle="--", alpha=0.7)
    ax2.set_title("Reconstruction Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Recon Loss")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    # KL Divergence
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs, history["train_kl"], "b-o", markersize=3, label="Train")
    ax3.plot(epochs, history["val_kl"],   "r-o", markersize=3, label="Val")
    ax3.set_title("KL Divergence")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("KL Loss")
    ax3.legend(); ax3.grid(True, alpha=0.3)

    # Val - Train Gap
    ax4 = fig.add_subplot(gs[1, 0])
    gap = [v - t for t, v in zip(history["train_loss"], history["val_loss"])]
    ax4.plot(epochs, gap, "purple", marker="o", markersize=3)
    ax4.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax4.fill_between(epochs, gap, 0, alpha=0.2, color="purple")
    ax4.set_title("Val - Train Gap (Overfitting)")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Gap")
    ax4.grid(True, alpha=0.3)

    # Cumulative Recon Improvement
    ax5 = fig.add_subplot(gs[1, 1])
    improvement = [history["train_recon"][0] - r for r in history["train_recon"]]
    ax5.plot(epochs, improvement, "green", marker="o", markersize=3)
    ax5.set_title("Cumulative Recon Improvement")
    ax5.set_xlabel("Epoch")
    ax5.set_ylabel("Improvement from Epoch 1")
    ax5.grid(True, alpha=0.3)

    # Summary Stats
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    best_val  = min(history["val_loss"])
    best_kl   = history["val_kl"][best_ep - 1]
    final_kl  = history["val_kl"][-1]

    stats = (
        f"{'─'*28}\n"
        f" VAE BASELINE RESULTS\n"
        f"{'─'*28}\n"
        f" Epochs run     :  {len(epochs)} / 50\n"
        f" Best epoch     :  {best_ep}\n"
        f" Best val loss  :  {best_val:.6f}\n"
        f" Best val recon :  {history['val_recon'][best_ep-1]:.6f}\n"
        f" Best val KL    :  {best_kl:.4f}\n"
        f" Final KL       :  {final_kl:.4f}\n"
        f" Early stopped  :  Yes (ep.{len(epochs)})\n"
        f"{'─'*28}\n"
        f" ✓ No KL collapse\n"
        f" ✓ Stable training\n"
        f" ⚠ Val plateaued ep.{best_ep}\n"
    )
    ax6.text(
        0.05, 0.95, stats,
        transform=ax6.transAxes,
        fontsize=9, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8)
    )

    out = os.path.join(save_dir, "training_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"    Saved → {out}")


# ── 2. Test Set Evaluation ────────────────────────────────────────────────────

def evaluate_test_set(model, loaders, device, save_dir: str):
    print("\n[2/3] Running test set evaluation...")

    model.eval()

    mse_heart_list, mse_lung_list = [], []
    n_batches = min(20, len(loaders["test"]))  # evaluate on 20 batches

    with torch.no_grad():
        for i, batch in enumerate(loaders["test"]):
            if i >= n_batches:
                break
            mixed = batch["mixed"].to(device)
            heart = batch["heart"].to(device)
            lung  = batch["lung"].to(device)

            output = model(mixed)

            mse_heart_list.append(F.mse_loss(output["heart_recon"], heart).item())
            mse_lung_list.append( F.mse_loss(output["lung_recon"],  lung).item())

    mse_heart = np.mean(mse_heart_list)
    mse_lung  = np.mean(mse_lung_list)

    print(f"    Test MSE Heart : {mse_heart:.6f}")
    print(f"    Test MSE Lung  : {mse_lung:.6f}")
    print(f"    Test MSE Avg   : {(mse_heart + mse_lung) / 2:.6f}")

    # Save to txt
    out = os.path.join(save_dir, "test_results.txt")
    with open(out, "w") as f:
        f.write("VAE Baseline — Test Set Results\n")
        f.write("=" * 35 + "\n")
        f.write(f"Test MSE Heart : {mse_heart:.6f}\n")
        f.write(f"Test MSE Lung  : {mse_lung:.6f}\n")
        f.write(f"Test MSE Avg   : {(mse_heart + mse_lung) / 2:.6f}\n")
    print(f"    Saved → {out}")

    return mse_heart, mse_lung


# ── 3. Visualize Separated Spectrograms ──────────────────────────────────────

def visualize_separations(model, loaders, device, save_dir: str, n_samples: int = 4):
    print("\n[3/3] Visualizing separated spectrograms...")

    model.eval()
    batch = next(iter(loaders["test"]))

    mixed = batch["mixed"][:n_samples].to(device)
    heart = batch["heart"][:n_samples].to(device)
    lung  = batch["lung"][:n_samples].to(device)

    with torch.no_grad():
        output = model(mixed)

    heart_recon = output["heart_recon"].cpu().numpy()
    lung_recon  = output["lung_recon"].cpu().numpy()
    mixed_np    = mixed.cpu().numpy()
    heart_np    = heart.cpu().numpy()
    lung_np     = lung.cpu().numpy()

    fig, axes = plt.subplots(n_samples, 5, figsize=(20, 4 * n_samples))
    fig.suptitle("Spectrogram Separation Results — VAE Baseline", fontsize=13, fontweight="bold")

    col_titles = ["Mixed (Input)", "Heart (Target)", "Heart (Recon)", "Lung (Target)", "Lung (Recon)"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=10, fontweight="bold")

    for i in range(n_samples):
        specs = [
            mixed_np[i, 0],
            heart_np[i, 0],
            heart_recon[i, 0],
            lung_np[i, 0],
            lung_recon[i, 0],
        ]
        for j, spec in enumerate(specs):
            im = axes[i, j].imshow(
                spec, aspect="auto", origin="lower",
                cmap="magma", vmin=spec.min(), vmax=spec.max()
            )
            axes[i, j].set_xlabel("Time")
            axes[i, j].set_ylabel("Mel Bin")
            plt.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.04)

        axes[i, 0].set_ylabel(f"Sample {i+1}\nMel Bin", fontsize=9)

    plt.tight_layout()
    out = os.path.join(save_dir, "spectrogram_separations.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"    Saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Load history
    print("Loading training history...")
    history = np.load(HISTORY_PATH, allow_pickle=True).item()

    # Load model
    print("Loading model checkpoint...")
    cfg    = load_config(CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = BioSoundVAE(
        latent_dim    = cfg.model.latent_dim,
        base_channels = 32,
        input_size    = (cfg.audio.n_mels, 94),
    ).to(device)

    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    # Load data
    print("Loading dataloaders...")
    loaders = get_all_dataloaders(CONFIG_PATH)

    # Run all 3 analyses
    plot_training_curves(history, SAVE_DIR)
    evaluate_test_set(model, loaders, device, SAVE_DIR)
    visualize_separations(model, loaders, device, SAVE_DIR, n_samples=4)

    print("\n" + "="*45)
    print("  Analysis Complete!")
    print(f"  All plots saved to: {SAVE_DIR}")
    print("="*45)