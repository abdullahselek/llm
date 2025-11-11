"""Test Large Language Model module."""

import torch

from llm.model import LLM


def test_LLM_forward_pass():
    """Tests LLM forward pass."""
    config = {
        "vocab_size": 1024,
        "context_length": 256,
        "embedding_dim": 256,
        "n_heads": 4,
        "n_layers": 4,
        "drop_rate": 0.1,
        "qkv_bias": False,
    }

    model = LLM(config)
    x = torch.randint(0, 1024, (32, 64))  # batch_size=32, seq_len=64
    logits = model(x)

    assert logits.shape == torch.Size([32, 64, 1024])
