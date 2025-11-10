"""Test GELU module."""

import torch

from llm.gelu import GELU


def test_GELU_activation_function():
    """Test applying GELU activation function."""
    gelu = GELU()
    x = torch.randn(3, 6, 50)
    y_gelu = gelu(x)

    assert y_gelu.shape == x.shape

    x_ = torch.randn(2, 4)
    y_gelu_ = gelu(x_)

    assert y_gelu_.shape == x_.shape
