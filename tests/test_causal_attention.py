"""Test Causal Attention Layer."""

import torch

from llm.causal_attention import CausalAttention


def test_CausalAttention_compute_context_vector(
    token_ids: list[int], batch_embeddings: torch.Tensor
):
    """Test computing CausalAttention context vector."""
    input_dim = batch_embeddings[0].shape[1]
    output_dim = 2

    torch.manual_seed(123)

    context_length = batch_embeddings.shape[1]
    causal_attention = CausalAttention(input_dim, output_dim, context_length, 0.1)

    context_vectors = causal_attention(batch_embeddings)

    assert context_vectors.shape == (2, len(token_ids), output_dim)
