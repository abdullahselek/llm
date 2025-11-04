"""Tests for SimpleTokenizerV1."""

import pytest

from llm.simple_tokenizer_v1 import SimpleTokenizerV1
from llm.utils import create_vocab


@pytest.fixture()
def vocab() -> dict:
    """Vocablary dict."""
    return create_vocab()


@pytest.fixture()
def tokenizerv1(vocab: dict) -> SimpleTokenizerV1:
    """Create a SimpleTokenizerV1 object."""
    return SimpleTokenizerV1(vocab=vocab)


def test_SimpleTokenizerV1_encode(tokenizerv1: SimpleTokenizerV1):
    """Test encode function of SimpleTokenizerV1."""
    ids = tokenizerv1.encode("It was not till three years later that.")

    assert len(ids) == 9


def test_SimpleTokenizerV1_encode_when_text_contains_unknown_words(
    tokenizerv1: SimpleTokenizerV1,
):
    """Test encode function of SimpleTokenizerV1."""
    ids = tokenizerv1.encode("Hello, this is a test text.")

    assert len(ids) == 8


def test_SimpleTokenizerV1_decode(tokenizerv1: SimpleTokenizerV1):
    """Test decode function of SimpleTokenizerV1."""
    original_text = "It was not till three years later that."
    ids = tokenizerv1.encode(original_text)
    text = tokenizerv1.decode(ids=ids)

    assert text == original_text


def test_SimpleTokenizerV1_encode_with_uknown_words_then_decode(
    tokenizerv1: SimpleTokenizerV1,
):
    """Test encode and decode with special character."""
    text1 = "Hello world!"
    text2 = "This is a Python unit test."
    text = " <|endoftext|> ".join([text1, text2])

    ids = tokenizerv1.encode(text)
    output = tokenizerv1.decode(ids)

    assert output == "<|unk|> <|unk|>! <|endoftext|> This is a <|unk|> <|unk|> <|unk|>."
