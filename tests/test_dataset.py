"""Tests for dataset module."""

import torch

from llm.bpe_tokenizer import BPETokenizer
from llm.dataset import (
    GPTDataset,
    IterableLLMDataset,
    LLMDataset,
    create_dataloader,
    create_llm_dataloader,
    create_llm_dataloader_from_dataset,
)
from tests.mocks import MockHFDataset


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


def test_IterableLLMDataset_padding(mock_hf_dataset: MockHFDataset):
    """Test if short texts are correctly padded with EOS tokens."""
    tokenizer = BPETokenizer()
    dataset = IterableLLMDataset(
        dataset=mock_hf_dataset, tokenizer=tokenizer, max_length=5, stride=2
    )

    iterator = iter(dataset)
    input_ids, target_ids = next(iterator)

    assert len(input_ids) == 5
    assert len(target_ids) == 5

    expected_input = torch.Tensor([16, 220, 17, 220, 18])

    assert torch.equal(input_ids, expected_input)


def test_create_llm_dataloader_from_dataset(mock_hf_dataset: MockHFDataset):
    """Test if it works with actual DataLoader."""
    data_loader = create_llm_dataloader_from_dataset(
        mock_hf_dataset, batch_size=2, max_length=5, stride=2, shuffle=False
    )

    batch = next(iter(data_loader))
    input_batch, target_batch = batch

    assert input_batch.shape == (2, 5)
    assert target_batch.shape == (2, 5)
