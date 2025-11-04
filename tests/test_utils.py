"""Tests for utils module."""

from llm.utils import create_vocab


def test_create_vocab():
    """Test function that creates vocablary."""
    vocab = create_vocab()

    assert len(vocab) == 1132
