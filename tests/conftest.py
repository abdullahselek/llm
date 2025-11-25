"""PyTest Configurations."""

from pathlib import Path

import pytest
import torch

from llm.bpe_tokenizer import BPETokenizer
from llm.utils import create_vocab
from tests.mocks import MockHFDataset


@pytest.fixture()
def vocab() -> dict:
    """Vocablary dict."""
    return create_vocab()


@pytest.fixture()
def vocab_text() -> str:
    """Vocablary raw text."""
    file_path = Path.cwd() / "src/llm/resources/the-verdict.txt"
    with open(str(file_path), encoding="utf-8") as f:
        text = f.read()
    return text


@pytest.fixture()
def bpe_tokenizer() -> BPETokenizer:
    """BPETokenizer fixture."""
    return BPETokenizer()


@pytest.fixture()
def token_ids(bpe_tokenizer: BPETokenizer) -> list[int]:
    """Fixture of token ids."""
    token_ids = bpe_tokenizer.encode(text="Designing and implementing an LLM.")
    return token_ids


@pytest.fixture()
def token_embeddings(token_ids: list[int]) -> torch.Tensor:
    """Fixture of token emmbeddings."""
    vocab_size = max(token_ids) + 1
    embedding_dim = 3
    embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)

    token_embeddings = embedding_layer(torch.tensor(token_ids))
    return token_embeddings


@pytest.fixture()
def batch_embeddings(token_embeddings: torch.Tensor) -> torch.Tensor:
    """Fixture of batch token embeddings."""
    batch = torch.stack((token_embeddings, token_embeddings), dim=0)
    return batch


@pytest.fixture()
def mock_hf_dataset() -> MockHFDataset:
    """Return Mock HuggingFace Dataset object."""
    raw_data = [
        {"content": "1 2 3"},
        {"content": "1 2 3 4 5 6 7 8"},
    ]
    return MockHFDataset(raw_data)
