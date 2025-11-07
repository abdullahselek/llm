"""Test Self Attention Layer."""

import pytest
import torch

from llm.bpe_tokenizer import BPETokenizer
from llm.self_attention import SelfAttention


@pytest.fixture()
def bpe_tokenizer() -> BPETokenizer:
    """BPETokenizer fixture."""
    return BPETokenizer()


def test_SelfAttention_compute_attention_weights(
    token_ids: list[int], token_embeddings: torch.Tensor
):
    """Test attention weights computed by SelfAttention."""
    assert token_embeddings[0].shape == torch.Size([3])

    input_dim = token_embeddings.shape[1]
    output_dim = 2

    torch.manual_seed(456)
    self_attention = SelfAttention(input_dim, output_dim)

    context_vec = self_attention(token_embeddings)

    assert context_vec.shape == (len(token_ids), output_dim)
