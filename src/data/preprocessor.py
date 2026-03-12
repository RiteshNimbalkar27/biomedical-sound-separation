"""
preprocessor.py
─────────────────────────────────────────────────────────────────
Audio preprocessing pipeline for the Biomedical Sound Separation
project.

Responsibilities:
  - Resample audio to a consistent sample rate
  - Segment long recordings into fixed-length windows
  - Normalize audio amplitude
  - Handle silence / low-energy segments (optional filtering)
  - Save processed segments to disk

Supports both PhysioNet (heart) and ICBHI (lung) datasets.
─────────────────────────────────────────────────────────────────
"""

import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

from src.utils.config import load_config, resolve_path
from src.utils.audio_utils import load_audio, save_audio, normalize_audio
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Core Preprocessing Functions ─────────────────────────────────────────────

def resample_audio(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample a waveform from orig_sr to target_sr.

    Args:
        waveform:  Input audio signal.
        orig_sr:   Original sample rate.
        target_sr: Target sample rate.

    Returns:
        Resampled waveform.
    """
    if orig_sr == target_sr:
        return waveform
    resampled = librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)
    logger.debug(f"Resampled {orig_sr}Hz -> {target_sr}Hz")
    return resampled


def segment_audio(
    waveform: np.ndarray,
    sample_rate: int,
    segment_length_sec: float = 3.0,
    overlap_sec: float = 0.0,
) -> list[np.ndarray]:
    """
    Split a waveform into fixed-length segments.

    Args:
        waveform:           Full audio signal.
        sample_rate:        Sample rate in Hz.
        segment_length_sec: Duration of each segment in seconds.
        overlap_sec:        Overlap between consecutive segments in seconds.
                            Set to 0.0 for no overlap (default).

    Returns:
        List of audio segment arrays, each of fixed length.
        The last incomplete segment is discarded.
    """
    segment_samples = int(segment_length_sec * sample_rate)
    hop_samples     = segment_samples - int(overlap_sec * sample_rate)

    if hop_samples <= 0:
        raise ValueError("Overlap must be smaller than segment_length.")

    segments = []
    start = 0

    while start + segment_samples <= len(waveform):
        segment = waveform[start : start + segment_samples]
        segments.append(segment)
        start += hop_samples

    logger.debug(f"Segmented into {len(segments)} segments of {segment_length_sec}s")
    return segments


def is_silent(waveform: np.ndarray, threshold_db: float = -50.0) -> bool:
    """
    Check whether an audio segment is effectively silent.
    Used to filter out uninformative segments before training.

    Args:
        waveform:     Audio segment.
        threshold_db: Energy threshold in dB. Segments below this
                      are considered silent. Default: -50 dB.

    Returns:
        True if silent, False if contains meaningful audio.
    """
    if np.max(np.abs(waveform)) == 0:
        return True

    rms = np.sqrt(np.mean(waveform ** 2))
    rms_db = 20 * np.log10(rms + 1e-9)
    return rms_db < threshold_db


def pad_or_trim(waveform: np.ndarray, target_length: int) -> np.ndarray:
    """
    Ensure a waveform is exactly target_length samples.
    Pads with zeros if shorter, trims if longer.

    Args:
        waveform:      Input audio signal.
        target_length: Desired number of samples.

    Returns:
        Waveform of exactly target_length samples.
    """
    if len(waveform) >= target_length:
        return waveform[:target_length]
    else:
        pad_length = target_length - len(waveform)
        return np.pad(waveform, (0, pad_length), mode="constant")


# ── Dataset-Level Processing ──────────────────────────────────────────────────

def process_dataset(
    input_dir: str,
    output_dir: str,
    sample_rate: int = 4000,
    segment_length_sec: float = 3.0,
    overlap_sec: float = 0.0,
    filter_silence: bool = True,
    silence_threshold_db: float = -50.0,
    supported_formats: tuple = (".wav", ".mp3", ".flac", ".aiff", ".ogg"),
) -> dict:
    """
    Process an entire dataset directory:
      1. Scan for all supported audio files
      2. Load and resample each file
      3. Normalize amplitude
      4. Segment into fixed-length clips
      5. Filter silent segments (optional)
      6. Save segments to output_dir as .wav files

    Args:
        input_dir:            Directory containing raw audio files.
        output_dir:           Directory to save processed segments.
        sample_rate:          Target sample rate in Hz.
        segment_length_sec:   Length of each segment in seconds.
        overlap_sec:          Overlap between segments in seconds.
        filter_silence:       Whether to discard silent segments.
        silence_threshold_db: dB threshold for silence detection.
        supported_formats:    Audio file extensions to process.

    Returns:
        Summary dictionary with processing statistics.
    """
    input_path  = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect all audio files recursively
    audio_files = [
        f for f in input_path.rglob("*")
        if f.suffix.lower() in supported_formats
    ]

    if not audio_files:
        logger.warning(f"No audio files found in: {input_dir}")
        return {"total_files": 0, "total_segments": 0, "skipped_silent": 0}

    logger.info(f"Found {len(audio_files)} audio files in {input_dir}")

    total_segments = 0
    skipped_silent = 0
    failed_files   = 0

    for audio_file in tqdm(audio_files, desc=f"Processing {input_path.name}"):
        try:
            # ── Load ──────────────────────────────────────────────
            # Load at native SR first, then resample explicitly
            # so we have full control over the process
            waveform, orig_sr = librosa.load(str(audio_file), sr=None, mono=True)

            # ── Resample ──────────────────────────────────────────
            waveform = resample_audio(waveform, orig_sr, sample_rate)

            # ── Normalize ─────────────────────────────────────────
            waveform = normalize_audio(waveform)

            # ── Segment ───────────────────────────────────────────
            segments = segment_audio(waveform, sample_rate, segment_length_sec, overlap_sec)

            # ── Save each segment ─────────────────────────────────
            stem = audio_file.stem  # original filename without extension

            for idx, segment in enumerate(segments):

                # Filter silence
                if filter_silence and is_silent(segment, silence_threshold_db):
                    skipped_silent += 1
                    continue

                # Build output filename: original_stem_seg000.wav
                out_filename = f"{stem}_seg{idx:03d}.wav"
                out_path     = output_path / out_filename

                save_audio(segment, str(out_path), sample_rate)
                total_segments += 1

        except Exception as e:
            logger.error(f"Failed to process {audio_file.name}: {e}")
            failed_files += 1
            continue

    summary = {
        "total_files":     len(audio_files),
        "total_segments":  total_segments,
        "skipped_silent":  skipped_silent,
        "failed_files":    failed_files,
        "output_dir":      str(output_path),
        "sample_rate":     sample_rate,
        "segment_length":  segment_length_sec,
    }

    logger.info("─" * 50)
    logger.info("Preprocessing complete.")
    logger.info(f"  Files processed : {len(audio_files) - failed_files}")
    logger.info(f"  Segments saved  : {total_segments}")
    logger.info(f"  Silent skipped  : {skipped_silent}")
    logger.info(f"  Failed files    : {failed_files}")
    logger.info(f"  Output dir      : {output_path}")
    logger.info("─" * 50)

    return summary


# ── Convenience Entry Point ───────────────────────────────────────────────────

def run_preprocessing(config_path: str = "configs/base_config.yaml"):
    """
    Run preprocessing for both PhysioNet and ICBHI datasets
    using settings from the YAML config file.
    """
    cfg = load_config(config_path)

    # Read overlap from config — default to 0.0 if not present
    overlap = getattr(cfg.audio, "overlap", 0.0)

    datasets = [
        {
            "name":       "PhysioNet (Heart)",
            "input_dir":  resolve_path(cfg.paths.raw_physionet),
            "output_dir": resolve_path(cfg.paths.processed) / "heart",
        },
        {
            "name":       "ICBHI (Lung)",
            "input_dir":  resolve_path(cfg.paths.raw_icbhi),
            "output_dir": resolve_path(cfg.paths.processed) / "lung",
        },
    ]

    all_summaries = {}

    for dataset in datasets:
        logger.info(f"\nProcessing dataset: {dataset['name']}")
        logger.info(f"  Input  -> {dataset['input_dir']}")
        logger.info(f"  Output -> {dataset['output_dir']}")

        summary = process_dataset(
            input_dir            = str(dataset["input_dir"]),
            output_dir           = str(dataset["output_dir"]),
            sample_rate          = cfg.audio.sample_rate,
            segment_length_sec   = cfg.audio.segment_length,
            overlap_sec          = overlap,             # ← now passed from config
            filter_silence       = True,
            silence_threshold_db = -50.0,
        )

        all_summaries[dataset["name"]] = summary

    return all_summaries
    """
    Run preprocessing for both PhysioNet and ICBHI datasets
    using settings from the YAML config file.

    This is the main function to call when you want to preprocess
    all raw data before training.

    Args:
        config_path: Path to the YAML configuration file.
    """
    cfg = load_config(config_path)

    datasets = [
        {
            "name":       "PhysioNet (Heart)",
            "input_dir":  resolve_path(cfg.paths.raw_physionet),
            "output_dir": resolve_path(cfg.paths.processed) / "heart",
        },
        {
            "name":       "ICBHI (Lung)",
            "input_dir":  resolve_path(cfg.paths.raw_icbhi),
            "output_dir": resolve_path(cfg.paths.processed) / "lung",
        },
    ]

    all_summaries = {}

    for dataset in datasets:
        logger.info(f"\nProcessing dataset: {dataset['name']}")
        logger.info(f"  Input  -> {dataset['input_dir']}")
        logger.info(f"  Output -> {dataset['output_dir']}")

        summary = process_dataset(
            input_dir          = str(dataset["input_dir"]),
            output_dir         = str(dataset["output_dir"]),
            sample_rate        = cfg.audio.sample_rate,
            segment_length_sec = cfg.audio.segment_length,
            filter_silence     = True,
            silence_threshold_db = -50.0,
        )

        all_summaries[dataset["name"]] = summary

    return all_summaries


# ── Run directly ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_preprocessing()