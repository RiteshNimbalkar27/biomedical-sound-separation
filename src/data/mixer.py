"""
mixer.py
─────────────────────────────────────────────────────────────────
Synthetic mixing pipeline for the Biomedical Sound Separation
project.

Takes preprocessed heart and lung segments and creates:
  - Mixed audio  (heart + lung at a controlled SNR)
  - Clean heart  (ground truth)
  - Clean lung   (ground truth)

Strategy: Option B — Oversampled Pairing
  Every heart segment is paired with a randomly sampled lung
  segment. Lung segments are reused (with shuffle) to cover
  all 33,122 heart segments → ~33,122 training triplets.

Output structure:
  data/mixed/
      train/
          heart/   → clean heart segments
          lung/    → clean lung segments
          mixed/   → heart + lung mixtures
      val/
          heart/
          lung/
          mixed/
      test/
          heart/
          lung/
          mixed/
─────────────────────────────────────────────────────────────────
"""

import os
import random
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Core Mixing Function ──────────────────────────────────────────────────────

def mix_signals(
    heart: np.ndarray,
    lung: np.ndarray,
    snr_db: float,
) -> np.ndarray:
    """
    Mix heart and lung signals at a specified Signal-to-Noise Ratio.

    The heart signal is treated as the "signal" and lung as the
    "noise" for SNR calculation purposes. The mixed output is
    peak-normalized to [-1, 1] to prevent clipping.

    Args:
        heart:  Clean heart segment (1D numpy array).
        lung:   Clean lung segment  (1D numpy array).
        snr_db: Desired SNR in decibels.
                Positive → heart dominates.
                Zero     → equal energy.
                Negative → lung dominates.

    Returns:
        Mixed waveform as 1D numpy array, peak-normalized.
    """
    # Ensure both signals are the same length
    min_len = min(len(heart), len(lung))
    heart   = heart[:min_len]
    lung    = lung[:min_len]

    # Calculate RMS energies
    heart_rms = np.sqrt(np.mean(heart ** 2)) + 1e-9
    lung_rms  = np.sqrt(np.mean(lung  ** 2)) + 1e-9

    # Scale lung signal to achieve the desired SNR
    # SNR = 20 * log10(heart_rms / scaled_lung_rms)
    # => scaled_lung_rms = heart_rms / 10^(snr_db/20)
    target_lung_rms  = heart_rms / (10 ** (snr_db / 20.0))
    lung_scale       = target_lung_rms / lung_rms
    scaled_lung      = lung * lung_scale

    # Mix
    mixed = heart + scaled_lung

    # Peak normalize to prevent clipping
    peak = np.max(np.abs(mixed))
    if peak > 0:
        mixed = mixed / peak

    return mixed


# ── Splitting Utility ─────────────────────────────────────────────────────────

def split_file_list(
    files: list,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
) -> tuple[list, list, list]:
    """
    Split a list of file paths into train / val / test sets.

    Args:
        files:      Full list of file paths.
        val_split:  Fraction for validation set.
        test_split: Fraction for test set.
        seed:       Random seed for reproducibility.

    Returns:
        Tuple of (train_files, val_files, test_files).
    """
    random.seed(seed)
    files = files.copy()
    random.shuffle(files)

    n          = len(files)
    n_test     = int(n * test_split)
    n_val      = int(n * val_split)

    test_files  = files[:n_test]
    val_files   = files[n_test : n_test + n_val]
    train_files = files[n_test + n_val:]

    return train_files, val_files, test_files


# ── Main Mixing Pipeline ──────────────────────────────────────────────────────

def create_mixed_dataset(
    heart_dir: str,
    lung_dir: str,
    output_dir: str,
    snr_min_db: float = -5.0,
    snr_max_db: float = 5.0,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
) -> dict:
    """
    Create the full mixed dataset using Option B oversampled pairing.

    For every heart segment, randomly sample one lung segment.
    Lung segments are shuffled and cycled to cover all heart segments.
    SNR is sampled uniformly from [snr_min_db, snr_max_db] for each pair.

    Args:
        heart_dir:   Directory of preprocessed heart segments.
        lung_dir:    Directory of preprocessed lung segments.
        output_dir:  Root output directory for mixed dataset.
        snr_min_db:  Minimum SNR in dB.
        snr_max_db:  Maximum SNR in dB.
        val_split:   Fraction for validation.
        test_split:  Fraction for test.
        seed:        Random seed.

    Returns:
        Summary dictionary with split sizes and output paths.
    """
    random.seed(seed)
    np.random.seed(seed)

    # ── Collect files ─────────────────────────────────────────
    heart_files = sorted(Path(heart_dir).rglob("*.wav"))
    lung_files  = sorted(Path(lung_dir).rglob("*.wav"))

    if not heart_files:
        raise FileNotFoundError(f"No heart .wav files found in: {heart_dir}")
    if not lung_files:
        raise FileNotFoundError(f"No lung .wav files found in: {lung_dir}")

    logger.info(f"Heart segments available : {len(heart_files)}")
    logger.info(f"Lung  segments available : {len(lung_files)}")

    # ── Split heart files into train/val/test ─────────────────
    train_heart, val_heart, test_heart = split_file_list(
        heart_files, val_split, test_split, seed
    )

    logger.info(f"Heart split → Train: {len(train_heart)} | "
                f"Val: {len(val_heart)} | Test: {len(test_heart)}")

    # ── Split lung files into train/val/test ──────────────────
    train_lung, val_lung, test_lung = split_file_list(
        lung_files, val_split, test_split, seed
    )

    logger.info(f"Lung  split → Train: {len(train_lung)} | "
                f"Val: {len(val_lung)} | Test: {len(test_lung)}")

    # ── Build pairs and mix each split ────────────────────────
    splits = {
        "train": (train_heart, train_lung),
        "val":   (val_heart,   val_lung),
        "test":  (test_heart,  test_lung),
    }

    summary = {}

    for split_name, (h_files, l_files) in splits.items():
        logger.info(f"\nMixing split: {split_name.upper()}")

        # Create output subdirectories
        out_heart = Path(output_dir) / split_name / "heart"
        out_lung  = Path(output_dir) / split_name / "lung"
        out_mixed = Path(output_dir) / split_name / "mixed"

        for d in [out_heart, out_lung, out_mixed]:
            d.mkdir(parents=True, exist_ok=True)

        # Cycle lung files to match number of heart files (Option B)
        # Shuffle lung files first for random pairing
        l_files_shuffled = l_files.copy()
        random.shuffle(l_files_shuffled)

        # Repeat lung list enough times to cover all heart files
        repeated_lung = []
        while len(repeated_lung) < len(h_files):
            repeated_lung.extend(l_files_shuffled)
            random.shuffle(l_files_shuffled)   # re-shuffle each cycle
        repeated_lung = repeated_lung[:len(h_files)]

        pairs_saved  = 0
        pairs_failed = 0

        for idx, (h_path, l_path) in enumerate(
            tqdm(zip(h_files, repeated_lung),
                 total=len(h_files),
                 desc=f"Mixing {split_name}")
        ):
            try:
                # Load both segments
                heart_wav, _ = sf.read(str(h_path))
                lung_wav,  _ = sf.read(str(l_path))

                # Convert to float32
                heart_wav = heart_wav.astype(np.float32)
                lung_wav  = lung_wav.astype(np.float32)

                # Sample a random SNR for this pair
                snr = np.random.uniform(snr_min_db, snr_max_db)

                # Create mixture
                mixed_wav = mix_signals(heart_wav, lung_wav, snr)

                # Align all three to the same length (the mixture length)
                min_len   = len(mixed_wav)
                heart_wav = heart_wav[:min_len]
                lung_wav  = lung_wav[:min_len]

                # Build consistent output filename
                out_name  = f"pair_{idx:06d}.wav"

                sf.write(str(out_heart / out_name), heart_wav, 4000)
                sf.write(str(out_lung  / out_name), lung_wav,  4000)
                sf.write(str(out_mixed / out_name), mixed_wav, 4000)

                pairs_saved += 1

            except Exception as e:
                logger.error(f"Failed pair {idx} ({h_path.name} + {l_path.name}): {e}")
                pairs_failed += 1
                continue

        summary[split_name] = {
            "pairs_saved":  pairs_saved,
            "pairs_failed": pairs_failed,
            "output_dir":   str(Path(output_dir) / split_name),
        }

        logger.info(f"  Pairs saved  : {pairs_saved}")
        logger.info(f"  Pairs failed : {pairs_failed}")

    return summary


# ── Convenience Entry Point ───────────────────────────────────────────────────

def run_mixing(config_path: str = "configs/base_config.yaml"):
    """
    Run the full mixing pipeline using settings from the YAML config.
    """
    cfg = load_config(config_path)

    heart_dir  = str(resolve_path(cfg.paths.processed) / "heart")
    lung_dir   = str(resolve_path(cfg.paths.processed) / "lung")
    output_dir = str(resolve_path(cfg.paths.mixed))

    logger.info("=" * 55)
    logger.info("  Biomedical Sound Separation — Mixing Pipeline")
    logger.info("=" * 55)
    logger.info(f"Heart dir  : {heart_dir}")
    logger.info(f"Lung dir   : {lung_dir}")
    logger.info(f"Output dir : {output_dir}")
    logger.info(f"SNR range  : [{cfg.mixing.snr_min_db}, {cfg.mixing.snr_max_db}] dB")
    logger.info(f"Splits     : val={cfg.training.val_split} | test={cfg.training.test_split}")
    logger.info("=" * 55)

    summary = create_mixed_dataset(
        heart_dir  = heart_dir,
        lung_dir   = lung_dir,
        output_dir = output_dir,
        snr_min_db = cfg.mixing.snr_min_db,
        snr_max_db = cfg.mixing.snr_max_db,
        val_split  = cfg.training.val_split,
        test_split = cfg.training.test_split,
        seed       = cfg.training.seed,
    )

    logger.info("\n" + "=" * 55)
    logger.info("  Mixing Complete — Final Summary")
    logger.info("=" * 55)
    for split, info in summary.items():
        logger.info(f"  {split.upper():<6} → {info['pairs_saved']} pairs saved")
    logger.info("=" * 55)

    return summary


# ── Run directly ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_mixing()