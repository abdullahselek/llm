"""Test Normalization Layer of LLM."""

import torch

from llm.layer_norm import LayerNorm


def test_LayerNorm_apply_normalization():
    """Test applying normalization."""
    layer_norm = LayerNorm(embedding_dim=6, eps=1e-5)

    torch.manual_seed(123)

    batch_tensor = torch.randn(3, 6)
    normalized_tensor = layer_norm(batch_tensor)

    mean = normalized_tensor.mean(dim=-1, keepdim=True)
    variance = normalized_tensor.var(dim=-1, unbiased=False, keepdim=True)

    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4)
    assert torch.allclose(variance, torch.ones_like(variance), atol=1e-4)
