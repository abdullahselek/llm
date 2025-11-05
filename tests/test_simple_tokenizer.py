"""Tests for SimpleTokenizer."""

import pytest

from llm.simple_tokenizer import SimpleTokenizer


@pytest.fixture()
def tokenizer(vocab: dict) -> SimpleTokenizer:
    """Create a SimpleTokenizer object."""
    return SimpleTokenizer(vocab=vocab)


def test_SimpleTokenizer_encode(tokenizer: SimpleTokenizer):
    """Test encode function of SimpleTokenizer."""
    ids = tokenizer.encode("It was not till three years later that.")

    assert len(ids) == 9


def test_SimpleTokenizer_encode_when_text_contains_unknown_words(
    tokenizer: SimpleTokenizer,
):
    """Test encode function of SimpleTokenizer."""
    ids = tokenizer.encode("Hello, this is a test text.")

    assert len(ids) == 8


def test_SimpleTokenizer_decode(tokenizer: SimpleTokenizer):
    """Test decode function of SimpleTokenizer."""
    original_text = "It was not till three years later that."
    ids = tokenizer.encode(original_text)
    text = tokenizer.decode(ids=ids)

    assert text == original_text


def test_SimpleTokenizer_encode_with_uknown_words_then_decode(
    tokenizer: SimpleTokenizer,
):
    """Test encode and decode with special character."""
    text1 = "Hello world!"
    text2 = "This is a Python unit test."
    text = " <|endoftext|> ".join([text1, text2])

    ids = tokenizer.encode(text)
    output = tokenizer.decode(ids)

    assert output == "<|unk|> <|unk|>! <|endoftext|> This is a <|unk|> <|unk|> <|unk|>."
