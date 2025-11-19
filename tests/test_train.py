"""Tests for training script."""

from datasets import load_dataset

from llm.train import process_dataset, split_dataset


def test_split_dataset():
    """Test dataset split into training and validation sets."""
    # The HF dataset we use has only training set
    # that's why I split by train to mimic the real use case
    dataset = load_dataset("csv", data_files="./tests/resources/dataset.csv", split="train")
    train_datset, val_dataset = split_dataset(dataset)

    assert len(train_datset) == 2
    assert len(val_dataset) == 1


def test_process_dataset():
    """Test dataset processing function."""
    dataset = load_dataset("csv", data_files="./tests/resources/dataset.csv", split="train")
    code_contents = process_dataset(dataset)

    assert isinstance(code_contents, list)
    assert len(code_contents) == 3
