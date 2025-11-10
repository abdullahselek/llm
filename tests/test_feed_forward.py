"""Test FeedForward layer of LLM."""

import torch

from llm.feed_forward import FeedForward


def test_FeedForward_apply_transformation():
    """Test FeedForward's forward pass."""
    feed_forward = FeedForward(embedding_dim=2560)
    x = torch.randn(32, 128, 2560)
    output = feed_forward(x)

    assert output.shape == (32, 128, 2560)
