"""
audio_utils.py
─────────────────────────────────────────────
Core audio utility functions for the
Biomedical Sound Separation project.

Handles loading, saving, plotting, and
basic inspection of audio files.
"""

import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path


# ── Loading & Saving ──────────────────────────────────────────────────────────

def load_audio(file_path: str, sample_rate: int = 4000) -> tuple[np.ndarray, int]:
    """
    Load an audio file and resample to the target sample rate.

    Args:
        file_path:   Path to the audio file (.wav, .mp3, .flac, etc.)
        sample_rate: Target sample rate in Hz. Defaults to 4000 Hz.

    Returns:
        Tuple of (waveform as np.ndarray, sample_rate)
    """
    waveform, sr = librosa.load(file_path, sr=sample_rate, mono=True)
    return waveform, sr


def save_audio(waveform: np.ndarray, file_path: str, sample_rate: int = 4000):
    """
    Save a waveform array to a .wav file.

    Args:
        waveform:    Audio signal as 1D numpy array.
        file_path:   Output path for the .wav file.
        sample_rate: Sample rate of the audio. Defaults to 4000 Hz.
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(file_path, waveform, sample_rate)


# ── Normalization ─────────────────────────────────────────────────────────────

def normalize_audio(waveform: np.ndarray) -> np.ndarray:
    """
    Normalize audio to the range [-1, 1] using peak normalization.

    Args:
        waveform: Raw audio signal.

    Returns:
        Peak-normalized audio signal.
    """
    peak = np.max(np.abs(waveform))
    if peak == 0:
        return waveform  # Avoid division by zero on silent clips
    return waveform / peak


# ── Feature Extraction ────────────────────────────────────────────────────────

def compute_melspectrogram(
    waveform: np.ndarray,
    sample_rate: int = 4000,
    n_fft: int = 512,
    hop_length: int = 128,
    n_mels: int = 64,
) -> np.ndarray:
    """
    Compute a mel-scaled spectrogram from a waveform.

    Args:
        waveform:    Audio signal.
        sample_rate: Sample rate in Hz.
        n_fft:       FFT window size.
        hop_length:  Hop size between frames.
        n_mels:      Number of mel filter banks.

    Returns:
        Log-power mel-spectrogram as a 2D numpy array (n_mels x time_frames).
    """
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    # Convert to log scale (dB), more perceptually meaningful
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db


def compute_mfcc(
    waveform: np.ndarray,
    sample_rate: int = 4000,
    n_mfcc: int = 13,
    n_fft: int = 512,
    hop_length: int = 128,
) -> np.ndarray:
    """
    Compute MFCCs from a waveform.
    Useful as a compact feature for classification or conditioning.

    Args:
        waveform:    Audio signal.
        sample_rate: Sample rate in Hz.
        n_mfcc:      Number of MFCC coefficients.
        n_fft:       FFT window size.
        hop_length:  Hop size between frames.

    Returns:
        MFCC matrix as a 2D numpy array (n_mfcc x time_frames).
    """
    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    return mfcc


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_waveform(waveform: np.ndarray, sample_rate: int = 4000, title: str = "Waveform"):
    """
    Plot the time-domain waveform of an audio signal.

    Args:
        waveform:    Audio signal.
        sample_rate: Sample rate in Hz.
        title:       Plot title.
    """
    plt.figure(figsize=(12, 3))
    librosa.display.waveshow(waveform, sr=sample_rate, alpha=0.7)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()


def plot_melspectrogram(
    mel_db: np.ndarray,
    sample_rate: int = 4000,
    hop_length: int = 128,
    title: str = "Mel-Spectrogram",
):
    """
    Plot a log-power mel-spectrogram.

    Args:
        mel_db:      Log-power mel-spectrogram (output of compute_melspectrogram).
        sample_rate: Sample rate in Hz.
        hop_length:  Hop size used during STFT.
        title:       Plot title.
    """
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(
        mel_db,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        cmap="magma",
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_comparison(
    waveforms: list[np.ndarray],
    labels: list[str],
    sample_rate: int = 4000,
    title: str = "Signal Comparison",
):
    """
    Plot multiple waveforms stacked vertically for easy comparison.
    Useful for visualizing mixed vs. separated signals side by side.

    Args:
        waveforms:   List of audio signals to compare.
        labels:      List of labels for each signal.
        sample_rate: Sample rate in Hz.
        title:       Overall plot title.
    """
    n = len(waveforms)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)
    fig.suptitle(title, fontsize=14)

    if n == 1:
        axes = [axes]

    for ax, waveform, label in zip(axes, waveforms, labels):
        librosa.display.waveshow(waveform, sr=sample_rate, ax=ax, alpha=0.7)
        ax.set_title(label)
        ax.set_ylabel("Amplitude")

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


# ── Inspection ────────────────────────────────────────────────────────────────

def get_audio_info(file_path: str, sample_rate: int = 4000) -> dict:
    """
    Return basic metadata about an audio file.

    Args:
        file_path:   Path to the audio file.
        sample_rate: Target sample rate for loading.

    Returns:
        Dictionary with duration, sample_rate, num_samples, file_path.
    """
    waveform, sr = load_audio(file_path, sample_rate)
    return {
        "file_path": str(file_path),
        "sample_rate": sr,
        "num_samples": len(waveform),
        "duration_sec": round(len(waveform) / sr, 3),
        "min_amplitude": round(float(np.min(waveform)), 5),
        "max_amplitude": round(float(np.max(waveform)), 5),
    }