"""Test Self Attention Layer."""

import pytest
import torch

from llm.bpe_tokenizer import BPETokenizer
from llm.self_attention import SelfAttention


@pytest.fixture()
def bpe_tokenizer() -> BPETokenizer:
    """BPETokenizer fixture."""
    return BPETokenizer()


def test_SelfAttention_compute_attention_weights(bpe_tokenizer: BPETokenizer):
    """Test attention weights computed by SelfAttention."""
    token_ids = bpe_tokenizer.encode(text="Designing and implementing an LLM.")
    print(f"token_ids: {len(token_ids)}")

    vocab_size = max(token_ids) + 1
    embedding_dim = 3
    embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)

    token_embeddings = embedding_layer(torch.tensor(token_ids))

    assert token_embeddings[0].shape == torch.Size([3])

    input_dim = token_embeddings.shape[1]
    output_dim = 2

    torch.manual_seed(456)
    self_attention = SelfAttention(input_dim, output_dim)

    context_vec = self_attention(token_embeddings)

    assert context_vec.shape == (len(token_ids), output_dim)
