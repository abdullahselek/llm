"""GPT Dataset and DataLoader."""

import torch
from torch.utils.data import DataLoader, Dataset

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


class LLMDataset(Dataset):
    """LLMDataset that inherits from Torch Dataset."""

    def __init__(
        self, texts: list[str], tokenizer: BPETokenizer, max_length: int, stride: int
    ):
        """Initialize.

        Args:
            texts (list[str]): List of texts to be encoded.
            tokenizer (BPETokenizer): BPETokenizer object.
            max_length (int): Chunk length.
            stride(int): Length of overflowing tokens.

        """
        self.input_ids = []
        self.target_ids = []

        for text in texts:
            token_ids = tokenizer.encode(text)
            # If text is shorter than max_length, can still use it
            # by creating a single chunk
            if len(token_ids) <= max_length:
                # Create a single chunk with padding if needed
                input_chunk = token_ids[:max_length]
                target_chunk = token_ids[1 : max_length + 1]

                # Pad if necessary
                if len(input_chunk) < max_length:
                    input_chunk.extend(
                        [tokenizer.eos_token_id] * (max_length - len(input_chunk))
                    )
                if len(target_chunk) < max_length:
                    target_chunk.extend(
                        [tokenizer.eos_token_id] * (max_length - len(target_chunk))
                    )

                self.input_ids.append(torch.tensor(input_chunk))
                self.target_ids.append(torch.tensor(target_chunk))
            else:
                # Normal chunking for longer texts
                for i in range(0, len(token_ids) - max_length, stride):
                    input_chunk = token_ids[i : i + max_length]
                    target_chunk = token_ids[i + 1 : i + max_length + 1]

                    # Ensure we have exactly max_length tokens
                    if len(input_chunk) == max_length and len(target_chunk) == max_length:
                        self.input_ids.append(torch.tensor(input_chunk))
                        self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        """Length of dataset."""
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Input ids and target ids at index."""
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(
    text: str,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """Create a DataLoader for GPTDataset.

    Args:
        text (str): Raw vocablary text.
        batch_size (int): Batch size, default 4.
        max_length (int): Chunk length, default 256.
        stride (int): Length of overflowing token, default 128.
        shuffle (bool): shuffle data in every epoch, default True.
        drop_last (bool): Drops last incomplete batch, default True.
        num_workers (int): Number of subprocesses to load the data, default 4.

    Returns:
        DataLoader object.

    """
    tokenizer = BPETokenizer()
    dataset = GPTDataset(text=text, tokenizer=tokenizer, max_length=max_length, stride=stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )


def create_llm_dataloader(
    texts: list[str],
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
) -> DataLoader:
    """Create a DataLoader for GPTDataset.

    Args:
        texts (list[str]): Raw vocablary text.
        batch_size (int): Batch size, default 4.
        max_length (int): Chunk length, default 256.
        stride (int): Length of overflowing token, default 128.
        shuffle (bool): shuffle data in every epoch, default True.
        drop_last (bool): Drops last incomplete batch, default True.
        num_workers (int): Number of subprocesses to load the data, default 0.
        pin_memory (bool): If True, the data loader will copy Tensors into device/CUDA
            pinned memory before returning them. Defaults to False.
        persistent_workers (bool): If True, the data loader will not shut down the worker
            processes after a dataset has been consumed once. This allows to maintain the
            workers Dataset instances alive. Defaults to False.

    Returns:
        DataLoader object.

    """
    tokenizer = BPETokenizer()
    dataset = LLMDataset(
        texts=texts, tokenizer=tokenizer, max_length=max_length, stride=stride
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
