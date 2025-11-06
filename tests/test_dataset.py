"""Tests for dataset module."""

from llm.bpe_tokenizer import BPETokenizer
from llm.dataset import GPTDataset, create_dataloader


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


def test_create_dataloader(vocab_text: str):
    """Test creating GPT data loader with raw text."""
    data_loader = create_dataloader(vocab_text, batch_size=4, max_length=256, stride=128)
    data_iter = iter(data_loader)
    first_batch = next(data_iter)

    assert len(first_batch) == 2
    assert len(first_batch[0]) == 4
    assert len(first_batch[-1]) == 4

    assert len(first_batch[0][0]) == 256
    assert len(first_batch[0][-1]) == 256
