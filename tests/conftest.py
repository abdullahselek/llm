"""PyTest Configurations."""

from pathlib import Path

import pytest

from llm.utils import create_vocab


@pytest.fixture()
def vocab() -> dict:
    """Vocablary dict."""
    return create_vocab()


@pytest.fixture()
def vocab_text() -> str:
    """Vocablary raw text."""
    file_path = Path.cwd() / "src/llm/resources/the-verdict.txt"
    with open(str(file_path), encoding="utf-8") as f:
        text = f.read()
    return text
