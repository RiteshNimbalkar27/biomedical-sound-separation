"""
metrics.py
─────────────────────────────────────────────────────────────────
Evaluation metrics for the Biomedical Sound Separation project.

Metrics implemented:
    Signal-level:
        SDR  — Signal-to-Distortion Ratio      (higher = better)
        SIR  — Signal-to-Interference Ratio    (higher = better)
        SAR  — Signal-to-Artifacts Ratio       (higher = better)
        SI-SDR — Scale-Invariant SDR           (higher = better)

    Spectrogram-level:
        MSE  — Mean Squared Error
        SSIM — Structural Similarity Index     (higher = better)
        LSD  — Log-Spectral Distance           (lower  = better)

    Training monitoring:
        SNR  — Signal-to-Noise Ratio

These metrics together give a complete picture of separation
quality from both signal and perceptual perspectives.
─────────────────────────────────────────────────────────────────
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Union


# ── Type alias ────────────────────────────────────────────────────────────────
Array = Union[np.ndarray, torch.Tensor]


def _to_numpy(x: Array) -> np.ndarray:
    """Convert tensor or array to float32 numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.array(x, dtype=np.float32)


# ── SI-SDR ────────────────────────────────────────────────────────────────────

def si_sdr(
    reference: Array,
    estimate: Array,
    eps: float = 1e-8,
) -> float:
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).

    The most commonly used metric in modern source separation
    literature. Scale-invariant means it doesn't penalize the
    model for predicting a scaled version of the target.

    Args:
        reference : Clean ground truth signal.
        estimate  : Model's separated output.
        eps       : Small value for numerical stability.

    Returns:
        SI-SDR in dB. Higher is better. 0 dB = baseline.
        Good separation typically > 10 dB.
    """
    ref = _to_numpy(reference).flatten()
    est = _to_numpy(estimate).flatten()

    # Remove DC offset
    ref = ref - np.mean(ref)
    est = est - np.mean(est)

    # Project estimate onto reference
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + eps)
    target    = alpha * ref
    noise     = est - target

    si_sdr_val = 10 * np.log10(
        (np.sum(target ** 2) + eps) /
        (np.sum(noise  ** 2) + eps)
    )

    return float(si_sdr_val)


# ── SDR / SIR / SAR ───────────────────────────────────────────────────────────

def sdr(
    reference: Array,
    estimate: Array,
    eps: float = 1e-8,
) -> float:
    """
    Signal-to-Distortion Ratio (SDR).

    Classic BSS_eval metric. Measures overall separation quality
    including interference and artifacts.

    Args:
        reference : Clean ground truth signal.
        estimate  : Model's separated output.
        eps       : Numerical stability term.

    Returns:
        SDR in dB. Higher is better.
    """
    ref = _to_numpy(reference).flatten()
    est = _to_numpy(estimate).flatten()

    noise = ref - est
    sdr_val = 10 * np.log10(
        (np.sum(ref   ** 2) + eps) /
        (np.sum(noise ** 2) + eps)
    )
    return float(sdr_val)


def snr(
    reference: Array,
    estimate: Array,
    eps: float = 1e-8,
) -> float:
    """
    Signal-to-Noise Ratio (SNR).

    Args:
        reference : Clean ground truth signal.
        estimate  : Model's separated output.
        eps       : Numerical stability term.

    Returns:
        SNR in dB. Higher is better.
    """
    ref   = _to_numpy(reference).flatten()
    est   = _to_numpy(estimate).flatten()
    noise = ref - est

    snr_val = 10 * np.log10(
        (np.sum(ref   ** 2) + eps) /
        (np.sum(noise ** 2) + eps)
    )
    return float(snr_val)


# ── Spectrogram Metrics ───────────────────────────────────────────────────────

def log_spectral_distance(
    reference: Array,
    estimate: Array,
    eps: float = 1e-8,
) -> float:
    """
    Log-Spectral Distance (LSD).

    Measures the distance between two spectrograms in log scale.
    Perceptually meaningful — humans perceive loudness
    logarithmically.

    Args:
        reference : Reference spectrogram [F, T] or [B, 1, F, T].
        estimate  : Estimated spectrogram.
        eps       : Numerical stability term.

    Returns:
        LSD value. Lower is better. 0 = perfect reconstruction.
    """
    ref = _to_numpy(reference).flatten()
    est = _to_numpy(estimate).flatten()

    # Ensure positive values before log
    ref = np.maximum(ref, eps)
    est = np.maximum(est, eps)

    lsd_val = np.sqrt(
        np.mean((10 * np.log10(ref) - 10 * np.log10(est)) ** 2)
    )
    return float(lsd_val)


def mse(reference: Array, estimate: Array) -> float:
    """
    Mean Squared Error between reference and estimate.

    Args:
        reference : Ground truth.
        estimate  : Model output.

    Returns:
        MSE value. Lower is better.
    """
    ref = _to_numpy(reference).flatten()
    est = _to_numpy(estimate).flatten()
    return float(np.mean((ref - est) ** 2))


def spectral_convergence(
    reference: Array,
    estimate: Array,
    eps: float = 1e-8,
) -> float:
    """
    Spectral Convergence — ratio of spectral differences.
    Used in neural vocoder papers as a reconstruction metric.

    Args:
        reference : Ground truth spectrogram.
        estimate  : Estimated spectrogram.

    Returns:
        Spectral convergence value. Lower is better.
    """
    ref = _to_numpy(reference)
    est = _to_numpy(estimate)

    return float(
        np.linalg.norm(ref - est, "fro") /
        (np.linalg.norm(ref, "fro") + eps)
    )


# ── Batch Evaluation ──────────────────────────────────────────────────────────

def evaluate_batch(
    heart_recon: torch.Tensor,
    lung_recon: torch.Tensor,
    heart_target: torch.Tensor,
    lung_target: torch.Tensor,
) -> dict:
    """
    Compute all metrics for a single batch.
    Averages metrics across all samples in the batch.

    Args:
        heart_recon  : Reconstructed heart spectrograms [B, 1, F, T]
        lung_recon   : Reconstructed lung spectrograms  [B, 1, F, T]
        heart_target : Ground truth heart               [B, 1, F, T]
        lung_target  : Ground truth lung                [B, 1, F, T]

    Returns:
        Dictionary of averaged metrics for this batch.
    """
    B = heart_recon.shape[0]

    heart_metrics = {
        "si_sdr": [], "sdr": [], "lsd": [],
        "mse": [], "spec_conv": []
    }
    lung_metrics = {
        "si_sdr": [], "sdr": [], "lsd": [],
        "mse": [], "spec_conv": []
    }

    for i in range(B):
        h_ref = heart_target[i].squeeze()
        h_est = heart_recon[i].squeeze()
        l_ref = lung_target[i].squeeze()
        l_est = lung_recon[i].squeeze()

        # Heart metrics
        heart_metrics["si_sdr"].append(si_sdr(h_ref, h_est))
        heart_metrics["sdr"].append(sdr(h_ref, h_est))
        heart_metrics["lsd"].append(log_spectral_distance(h_ref, h_est))
        heart_metrics["mse"].append(mse(h_ref, h_est))
        heart_metrics["spec_conv"].append(spectral_convergence(h_ref, h_est))

        # Lung metrics
        lung_metrics["si_sdr"].append(si_sdr(l_ref, l_est))
        lung_metrics["sdr"].append(sdr(l_ref, l_est))
        lung_metrics["lsd"].append(log_spectral_distance(l_ref, l_est))
        lung_metrics["mse"].append(mse(l_ref, l_est))
        lung_metrics["spec_conv"].append(spectral_convergence(l_ref, l_est))

    # Average across batch
    results = {}
    for k, v in heart_metrics.items():
        results[f"heart_{k}"] = float(np.mean(v))
    for k, v in lung_metrics.items():
        results[f"lung_{k}"] = float(np.mean(v))

    # Overall averages
    results["mean_si_sdr"] = (results["heart_si_sdr"] + results["lung_si_sdr"]) / 2
    results["mean_sdr"]    = (results["heart_sdr"]    + results["lung_sdr"])    / 2
    results["mean_lsd"]    = (results["heart_lsd"]    + results["lung_lsd"])    / 2

    return results


# ── Full Test Set Evaluation ──────────────────────────────────────────────────

def evaluate_model(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
) -> dict:
    """
    Run full evaluation on the test set.

    Args:
        model       : Trained VAE model.
        test_loader : DataLoader for test split.
        device      : torch device.

    Returns:
        Dictionary of averaged metrics across the full test set.
    """
    from tqdm import tqdm

    model.eval()

    all_metrics = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            mixed = batch["mixed"].to(device)
            heart = batch["heart"].to(device)
            lung  = batch["lung"].to(device)

            output = model(mixed)

            batch_metrics = evaluate_batch(
                heart_recon  = output["heart_recon"],
                lung_recon   = output["lung_recon"],
                heart_target = heart,
                lung_target  = lung,
            )
            all_metrics.append(batch_metrics)

    # Average all metrics across all batches
    final_metrics = {}
    for key in all_metrics[0].keys():
        final_metrics[key] = float(
            np.mean([m[key] for m in all_metrics])
        )

    return final_metrics


def print_metrics(metrics: dict):
    """Pretty print evaluation metrics."""
    print()
    print("=" * 50)
    print("  Evaluation Results")
    print("=" * 50)
    print(f"  {'Metric':<25} {'Heart':>8}  {'Lung':>8}")
    print("  " + "-" * 44)

    metric_names = ["si_sdr", "sdr", "lsd", "mse", "spec_conv"]
    labels       = ["SI-SDR (dB↑)", "SDR (dB↑)",
                    "LSD (↓)", "MSE (↓)", "Spec Conv (↓)"]

    for name, label in zip(metric_names, labels):
        h = metrics.get(f"heart_{name}", 0)
        l = metrics.get(f"lung_{name}",  0)
        print(f"  {label:<25} {h:>8.4f}  {l:>8.4f}")

    print("  " + "-" * 44)
    print(f"  {'Mean SI-SDR':<25} {metrics.get('mean_si_sdr', 0):>8.4f}")
    print(f"  {'Mean SDR':<25} {metrics.get('mean_sdr',    0):>8.4f}")
    print(f"  {'Mean LSD':<25} {metrics.get('mean_lsd',    0):>8.4f}")
    print("=" * 50)