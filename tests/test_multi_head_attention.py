"""Test Multi Head Attention Layer."""

import torch

from llm.multi_head_attention import MultiHeadAttention


def test_MultiHeadAttention_compute_context_vectors(
    token_ids: list[int], batch_embeddings: torch.Tensor
):
    """Test compute context vectors using batch embeddings."""
    torch.manual_seed(456)

    # number of tokens
    context_length = batch_embeddings.shape[1]
    input_dim, output_dim = 3, 2

    multi_head_attention = MultiHeadAttention(
        input_dim=input_dim,
        output_dim=output_dim,
        context_length=context_length,
        dropout=0.1,
        num_heads=2,
    )

    context_vectors = multi_head_attention(batch_embeddings)

    # token_ids are used in generating token_embeddings and
    # batch_embeddings are combination of two token_embeddings
    # they are available in conftest.py
    assert context_vectors.shape == (2, len(token_ids), output_dim)


def test_MultiHeadAttention_compute_context_vectors_when_KV_cache_enabled(
    token_ids: list[int], batch_embeddings: torch.Tensor
):
    """Test compute context vectors using batch embeddings when KV cache is enabled."""
    torch.manual_seed(456)

    # number of tokens
    context_length = batch_embeddings.shape[1]
    input_dim, output_dim = 3, 2

    multi_head_attention = MultiHeadAttention(
        input_dim=input_dim,
        output_dim=output_dim,
        context_length=context_length,
        dropout=0.1,
        num_heads=2,
        qkv_bias=False,
        max_seq_len=256,
        window_size=128,
    )

    context_vectors = multi_head_attention(batch_embeddings, use_cache=True)

    # token_ids are used in generating token_embeddings and
    # batch_embeddings are combination of two token_embeddings
    # they are available in conftest.py
    assert context_vectors.shape == (2, len(token_ids), output_dim)


def test_MultiHeadAttention_reset_cache(batch_embeddings: torch.Tensor):
    """Test resetting cache of MultiHeadAttention."""
    torch.manual_seed(456)

    context_length = batch_embeddings.shape[1]
    input_dim, output_dim = 3, 2

    multi_head_attention = MultiHeadAttention(
        input_dim=input_dim,
        output_dim=output_dim,
        context_length=context_length,
        dropout=0.1,
        num_heads=2,
        qkv_bias=False,
        max_seq_len=256,
        window_size=8,
    )

    multi_head_attention(batch_embeddings, use_cache=True)

    assert len(multi_head_attention.cached_keys) > 0
    assert len(multi_head_attention.cached_values) > 0

    multi_head_attention.reset_cache()

    assert multi_head_attention.cached_keys is None
    assert multi_head_attention.cached_values is None
