"""Test Large Language Model module."""

import torch

from llm.bpe_tokenizer import BPETokenizer
from llm.model import LLM, generate_text


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


def test_LLM_generate_text():
    """Test LLM text generation."""
    config = {
        "vocab_size": 50257,
        "context_length": 1024,
        "embedding_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False,
    }
    torch.manual_seed(123)
    model = LLM(config)
    model.eval()

    context = "Hello, I am"

    bpe_tokenizer = BPETokenizer()
    ids = bpe_tokenizer.encode(context)
    encoded_tensor = torch.tensor(ids).unsqueeze(0)

    output = generate_text(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=10,
        context_size=config["context_length"],
    )
    decoded_text = bpe_tokenizer.decode(output.squeeze(0).tolist())

    assert len(output[0]) == 14
    assert len(decoded_text) > len(context)
