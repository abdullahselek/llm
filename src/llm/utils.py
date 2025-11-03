"""Utility module."""

import re
from pathlib import Path


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
    vocab = {token: integer for integer, token in enumerate(all_words)}

    return vocab
