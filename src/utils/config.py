"""
config.py
─────────────────────────────────────────────
Centralized configuration loader for the
Biomedical Sound Separation project.

Loads settings from a YAML file and exposes
them as a clean, dot-accessible Python object.
"""

import yaml
import os
from pathlib import Path


class Config:
    """
    Dot-accessible configuration object.
    Loads from a YAML file and allows nested access like:
        cfg.audio.sample_rate
        cfg.training.batch_size
    """

    def __init__(self, config_dict: dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                # Recursively wrap nested dicts
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __repr__(self):
        return f"Config({self.__dict__})"


def load_config(config_path: str = "configs/base_config.yaml") -> Config:
    """
    Load a YAML config file and return a Config object.

    Args:
        config_path: Path to the YAML config file.
                     Defaults to configs/base_config.yaml

    Returns:
        Config object with dot-accessible attributes.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Make sure you are running from the project root directory."
        )

    with open(path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    return Config(config_dict)


def get_project_root() -> Path:
    """
    Returns the absolute path to the project root directory.
    Useful for building absolute paths in any module.
    """
    # Walks up from this file → src/utils/ → src/ → project root
    return Path(__file__).resolve().parents[2]


def resolve_path(relative_path: str) -> Path:
    """
    Resolves a relative path from the project root.

    Args:
        relative_path: A path string like "data/processed"

    Returns:
        Absolute Path object.
    """
    return get_project_root() / relative_path