"""Mutli Head Attention layer with Sliding Window Attention."""

import torch.nn as nn


class MultiHeadAttentionWithSWA(nn.Module):
    """MultiHeadAttention layer with sliding window attention mechanism.

    This module implements multi head attention with optional sliding window
    constraints to limit long range dependencies.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
        sliding_window_size: int | None = None,
    ):
        """Initialize MultiHeadAttentionWithSWA object.

        Args:
            input_dim (int): The dimension of the input embeddings.
            output_dim (int): The dimension of the output embeddings.
                Must be divisible by num_heads.
            context_length (int): The maximum sequence length that the attention can process.
            dropout (float): Dropout probability for attention weights and output projection.
            num_heads (int): Number of attention heads to use.
            qkv_bias (bool, optional): If True, add bias to the Q, K, V linear projections.
                Defaults to False.
            sliding_window_size (int | None, optional): Size of the sliding window.
                If None, no sliding window is applied.  If specified, only tokens within this
                window can attend to each other. Defaults to None.

        """
        super().__init__()
        assert output_dim % num_heads == 0, "output_dim must be divisible by num_heads"

        self.output_dim = output_dim
        self.context_length = context_length
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads

        self.qkv = nn.Linear(input_dim, output_dim * 3, bias=qkv_bias)
        self.out_proj = nn.Linear(output_dim, output_dim)
        self.dropout = dropout

        self.register_buffer("cached_keys", None, persistent=False)
        self.register_buffer("cached_values", None, persistent=False)
        self.ptr_cur = 0
