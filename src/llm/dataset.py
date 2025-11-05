"""GPT Dataset."""

import torch
from torch.utils.data import Dataset

from llm.bpe_tokenizer import BPETokenizer


class GPTDataset(Dataset):
    """GPTDataset that inherits from Torch Dataset."""

    def __init__(self, text: str, tokenizer: BPETokenizer, max_length: int, stride: int):
        """Initialize.

        Args:
            text (str): Text to be encoded.
            tokenizer (BPETokenizer): BPETokenizer object.
            max_length (int): Chunk length.
            stride(int): Length of overflowing tokens.

        """
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text)
        assert len(token_ids) > max_length, (
            "Number of tokenized inputs must at least be equal to max_length+1"
        )

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        """Length of dataset."""
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Input ids and target ids at index."""
        return self.input_ids[idx], self.target_ids[idx]
