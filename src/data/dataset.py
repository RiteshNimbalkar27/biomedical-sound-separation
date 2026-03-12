"""
dataset.py
─────────────────────────────────────────────────────────────────
PyTorch Dataset and DataLoader for the Biomedical Sound
Separation project.

Reads pre-mixed triplets from data/mixed/ and returns
mel-spectrogram tensors ready for model training.

Each sample returns:
    {
        "mixed" : FloatTensor [1, n_mels, T]  ← model input
        "heart" : FloatTensor [1, n_mels, T]  ← target 1
        "lung"  : FloatTensor [1, n_mels, T]  ← target 2
        "id"    : str                          ← pair filename
    }
─────────────────────────────────────────────────────────────────
"""

import torch
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Core Dataset Class ────────────────────────────────────────────────────────

class BioSoundDataset(Dataset):
    """
    PyTorch Dataset for biomedical sound separation.

    Loads (mixed, heart, lung) triplets from the mixed dataset
    directory and returns them as mel-spectrogram tensors.

    Args:
        split:       One of 'train', 'val', 'test'.
        mixed_dir:   Root directory of the mixed dataset.
        sample_rate: Audio sample rate in Hz.
        n_fft:       FFT window size for spectrogram.
        hop_length:  Hop size between STFT frames.
        n_mels:      Number of mel filterbank channels.
        augment:     Whether to apply data augmentation (train only).
    """

    def __init__(
        self,
        split: str,
        mixed_dir: str,
        sample_rate: int  = 4000,
        n_fft: int        = 512,
        hop_length: int   = 128,
        n_mels: int       = 64,
        augment: bool     = False,
    ):
        super().__init__()

        assert split in ("train", "val", "test"), \
            f"split must be 'train', 'val', or 'test'. Got: {split}"

        self.split       = split
        self.sample_rate = sample_rate
        self.n_fft       = n_fft
        self.hop_length  = hop_length
        self.n_mels      = n_mels
        self.augment     = augment

        # ── Build paths ───────────────────────────────────────
        split_dir        = Path(mixed_dir) / split
        self.heart_dir   = split_dir / "heart"
        self.lung_dir    = split_dir / "lung"
        self.mixed_dir   = split_dir / "mixed"

        # Validate directories exist
        for d in [self.heart_dir, self.lung_dir, self.mixed_dir]:
            if not d.exists():
                raise FileNotFoundError(
                    f"Expected directory not found: {d}\n"
                    f"Make sure you have run src/data/mixer.py first."
                )

        # ── Collect all pair filenames ───────���─────────────────
        self.pair_files = sorted([
            f.name for f in self.mixed_dir.glob("*.wav")
        ])

        if not self.pair_files:
            raise ValueError(f"No .wav files found in {self.mixed_dir}")

        logger.info(
            f"BioSoundDataset [{split}] — {len(self.pair_files)} pairs loaded"
        )

    def __len__(self) -> int:
        return len(self.pair_files)

    def __getitem__(self, idx: int) -> dict:
        filename = self.pair_files[idx]

        # ── Load all three waveforms ───────────────────────────
        mixed_wav = self._load_wav(self.mixed_dir / filename)
        heart_wav = self._load_wav(self.heart_dir / filename)
        lung_wav  = self._load_wav(self.lung_dir  / filename)

        # ── Optional augmentation (train split only) ──────────
        if self.augment and self.split == "train":
            mixed_wav, heart_wav, lung_wav = self._augment(
                mixed_wav, heart_wav, lung_wav
            )

        # ── Convert to mel-spectrograms ────────────────────────
        mixed_mel = self._to_melspec(mixed_wav)
        heart_mel = self._to_melspec(heart_wav)
        lung_mel  = self._to_melspec(lung_wav)

        return {
            "mixed" : mixed_mel,   # [1, n_mels, T]
            "heart" : heart_mel,   # [1, n_mels, T]
            "lung"  : lung_mel,    # [1, n_mels, T]
            "id"    : filename,
        }

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _load_wav(self, path: Path) -> np.ndarray:
        """Load a .wav file and return as float32 numpy array."""
        waveform, _ = sf.read(str(path), dtype="float32")
        return waveform

    def _to_melspec(self, waveform: np.ndarray) -> torch.Tensor:
        """
        Convert a waveform to a log-power mel-spectrogram tensor.

        Returns:
            FloatTensor of shape [1, n_mels, T]
        """
        import librosa

        mel = librosa.feature.melspectrogram(
            y          = waveform,
            sr         = self.sample_rate,
            n_fft      = self.n_fft,
            hop_length = self.hop_length,
            n_mels     = self.n_mels,
        )

        # Convert to log scale — clamp to avoid log(0)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Normalize to [0, 1] range — stable for VAE input
        mel_db = self._normalize_spectrogram(mel_db)

        # Add channel dim → [1, n_mels, T]
        return torch.FloatTensor(mel_db).unsqueeze(0)

    def _normalize_spectrogram(self, mel_db: np.ndarray) -> np.ndarray:
        """
        Normalize spectrogram to [0, 1] using min-max normalization.
        Uses fixed dB range (-80, 0) for consistency across all samples.
        """
        MIN_DB = -80.0
        MAX_DB =   0.0
        mel_db = np.clip(mel_db, MIN_DB, MAX_DB)
        return (mel_db - MIN_DB) / (MAX_DB - MIN_DB)

    def _augment(
        self,
        mixed: np.ndarray,
        heart: np.ndarray,
        lung: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply lightweight data augmentation to waveforms.
        Applied consistently to all three signals in a triplet.

        Augmentations:
            - Random amplitude scaling  (±20%)
            - Random polarity flip      (50% chance)
        """
        # Random amplitude scale — apply same scale to all three
        scale = np.random.uniform(0.8, 1.2)
        mixed = mixed * scale
        heart = heart * scale
        lung  = lung  * scale

        # Random polarity flip
        if np.random.rand() < 0.5:
            mixed = -mixed
            heart = -heart
            lung  = -lung

        return mixed, heart, lung


# ── DataLoader Factory ────────────────────────────────────────────────────────

def get_dataloader(
    split: str,
    mixed_dir: str,
    batch_size: int  = 16,
    sample_rate: int = 4000,
    n_fft: int       = 512,
    hop_length: int  = 128,
    n_mels: int      = 64,
    num_workers: int = 0,
    augment: bool    = False,
) -> DataLoader:
    """
    Build and return a DataLoader for a given split.

    Args:
        split:       'train', 'val', or 'test'.
        mixed_dir:   Root directory of mixed dataset.
        batch_size:  Number of samples per batch.
        sample_rate: Audio sample rate.
        n_fft:       FFT window size.
        hop_length:  STFT hop size.
        n_mels:      Mel filterbank channels.
        num_workers: Number of DataLoader worker processes.
                     Use 0 on Windows to avoid multiprocessing issues.
        augment:     Enable augmentation for training split.

    Returns:
        PyTorch DataLoader instance.
    """
    dataset = BioSoundDataset(
        split       = split,
        mixed_dir   = mixed_dir,
        sample_rate = sample_rate,
        n_fft       = n_fft,
        hop_length  = hop_length,
        n_mels      = n_mels,
        augment     = augment,
    )

    loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = (split == "train"),   # only shuffle training data
        num_workers = num_workers,
        pin_memory  = torch.cuda.is_available(),
        drop_last   = (split == "train"),   # avoid incomplete last batch
    )

    logger.info(
        f"DataLoader [{split}] — "
        f"{len(dataset)} samples | "
        f"batch_size={batch_size} | "
        f"{len(loader)} batches"
    )

    return loader


# ── Convenience: Load All Three Splits ───────────────────────────────────────

def get_all_dataloaders(
    config_path: str = "configs/base_config.yaml",
) -> dict[str, DataLoader]:
    """
    Load train, val, and test DataLoaders from config.

    Returns:
        Dict with keys 'train', 'val', 'test'.
    """
    cfg = load_config(config_path)

    from src.utils.config import resolve_path
    mixed_dir = str(resolve_path(cfg.paths.mixed))

    loaders = {}

    for split in ("train", "val", "test"):
        loaders[split] = get_dataloader(
            split       = split,
            mixed_dir   = mixed_dir,
            batch_size  = cfg.training.batch_size,
            sample_rate = cfg.audio.sample_rate,
            n_fft       = cfg.audio.n_fft,
            hop_length  = cfg.audio.hop_length,
            n_mels      = cfg.audio.n_mels,
            num_workers = 0,        # safe default for Windows
            augment     = (split == "train"),
        )

    return loaders