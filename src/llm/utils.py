"""Utility module."""

import re
from pathlib import Path

import yaml


def create_vocab() -> dict:
    """Create vocablary dictionary.

    Returns:
        Vocablary dict.

    """
    file_path = Path.cwd() / "src/llm/resources/the-verdict.txt"
    with open(str(file_path), encoding="utf-8") as f:
        text = f.read()

    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]

    all_words = sorted(set(preprocessed))
    all_words.extend(["<|endoftext|>", "<|unk|>"])
    vocab = {token: integer for integer, token in enumerate(all_words)}

    return vocab


def read_config(cfg_path: str) -> dict:
    """Load and read LLM configuration.

    Args:
        cfg_path (str): Configuration file path.

    Returns:
        Loaded config as dictionary.

    """
    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    return config
