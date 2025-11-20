"""Tests for dataset module."""

from llm.bpe_tokenizer import BPETokenizer
from llm.dataset import GPTDataset, LLMDataset, create_dataloader, create_llm_dataloader


def test_GPTDataset_loading_items(vocab_text: str):
    """Test GPTDataset initilization and loading items after encoding."""
    tokenizer = BPETokenizer()
    dataset = GPTDataset(text=vocab_text, tokenizer=tokenizer, max_length=256, stride=128)

    assert len(dataset) == 36


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


def test_LLMDataset_loading_items(vocab_text: str):
    """Test LLMDataset initilization and loading items after encoding."""
    tokenizer = BPETokenizer()
    dataset = LLMDataset(
        texts=[vocab_text, vocab_text], tokenizer=tokenizer, max_length=256, stride=128
    )

    assert len(dataset) == 72


def test_LLMDataset_getitem_at_index(vocab_text: str):
    """Test getting item from dataset after loading and encoding items."""
    tokenizer = BPETokenizer()
    dataset = LLMDataset(
        texts=[vocab_text, vocab_text], tokenizer=tokenizer, max_length=256, stride=128
    )
    item = dataset[0]

    assert len(item) == 2


def test_create_llm_dataloader(vocab_text: str):
    """Test creating LLM data loader with raw text."""
    data_loader = create_llm_dataloader(
        [vocab_text, vocab_text], batch_size=4, max_length=256, stride=128
    )
    data_iter = iter(data_loader)
    first_batch = next(data_iter)

    assert len(first_batch) == 2
    assert len(first_batch[0]) == 4
    assert len(first_batch[-1]) == 4

    assert len(first_batch[0][0]) == 256
    assert len(first_batch[0][-1]) == 256
