"""Tests for dataset module."""

from llm.bpe_tokenizer import BPETokenizer
from llm.dataset import GPTDataset


def test_GPTDataset_loading_items(vocab_text: str):
    """Test GPTDataset initilization and loading items after encoding."""
    tokenizer = BPETokenizer()
    dataset = GPTDataset(text=vocab_text, tokenizer=tokenizer, max_length=256, stride=128)

    assert len(dataset) == 39


def test_GPTDataset_getitem_at_index(vocab_text: str):
    """Test getting item from dataset after loading and encoding items."""
    tokenizer = BPETokenizer()
    dataset = GPTDataset(text=vocab_text, tokenizer=tokenizer, max_length=256, stride=128)
    item = dataset[0]

    assert len(item) == 2
