"""Tests for training script."""

from unittest.mock import MagicMock

from llm.train import process_dataset


def test_process_dataset():
    """Test dataset processing function."""
    mock_dataset = MagicMock()
    mock_dataset.num_rows = 3
    mock_dataset.__iter__.return_value = [
        "print('hello world')",
        "assert 2 + 2 == 4",
        "code_content = dataset['content'] if 'content' in dataset else None",
    ]
    code_contents = process_dataset(mock_dataset)

    assert isinstance(code_contents, list)
    assert len(code_contents) == 3
