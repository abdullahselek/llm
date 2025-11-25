"""Tests for utils module."""

from pathlib import Path

from llm.utils import create_vocab, read_config


def test_create_vocab():
    """Test function that creates vocablary."""
    vocab = create_vocab()

    assert len(vocab) == 1132


def test_read_config():
    """Test loading and reading LLM config file."""
    cfg_path = Path.cwd() / "src/llm/configs/llm_3.55b.yaml"

    config = read_config(cfg_path=cfg_path)

    assert config["vocab_size"] == 200019
    assert config["context_length"] == 2048
    assert config["embedding_dim"] == 2560
    assert config["n_heads"] == 32
    assert config["n_layers"] == 32
    assert config["drop_rate"] == 0.1
    assert config["qkv_bias"] is False
