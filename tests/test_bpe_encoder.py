"""Unit tests for BPETokenizer."""

import pytest

from llm.bpe_tokenizer import BPETokenizer


@pytest.fixture()
def bpe_tokenizer() -> BPETokenizer:
    """Create a BPETokenizer object."""
    return BPETokenizer()


def test_BPETokenizer_encode(bpe_tokenizer: BPETokenizer):
    """Test BPETokenizer encode."""
    ids = bpe_tokenizer.encode("It was not till three years later that.")

    assert len(ids) == 9


def test_BPETokenizer_encode_with_test_text(bpe_tokenizer: BPETokenizer):
    """Test BPETokenizer encode."""
    ids = bpe_tokenizer.encode("Hello, this is a test text.")

    assert len(ids) == 8


def test_BPETokenizer_decode(bpe_tokenizer: BPETokenizer):
    """Test BPETokenizer decode."""
    text = "It was not till three years later that."
    ids = bpe_tokenizer.encode(text)
    output = bpe_tokenizer.decode(ids)

    assert output == text
