"""Test Transformer Block."""

import torch

from llm.transformer_block import TransformerBlock


def test_TransformerBlock_tensor_tranformation():
    """Test TransformerBlock forward pass."""
    transformer_block = TransformerBlock(
        context_length=512,
        embedding_dim=2560,
        n_heads=32,
        drop_rate=0.1,
        qkv_bias=True,
    )
    x = torch.randn(32, 128, 2560)
    output = transformer_block(x)

    assert output.shape == (32, 128, 2560)
