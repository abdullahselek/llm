"""Transformer Block."""

import torch
import torch.nn as nn

from llm.feed_forward import FeedForward
from llm.layer_norm import LayerNorm
from llm.multi_head_attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    """Single transformer block implementing the core building block of transformer arch.

    A transformer block consists of two main components:
    1. Multi Head Self Attention mechanism with residual connection and layer normalization.
    2. Feed Forward network with residual connection and layer normalization.

    This implementation follows the standard transformer architecture pattern where each
    component is applied in sequence with residual connections to enable deeper networks
    and better gradient flow.

    The block processes input through:
    - Multi Head attention with residual connection and layer normalization.
    - Feed forward network with residual connection and layer normalization.
    """

    def __init__(
        self,
        context_length: int,
        embedding_dim: int,
        n_heads: int,
        drop_rate: float,
        qkv_bias: bool,
        window_size: int | None = None,
    ):
        """Initialze TransformerBlock object.

        Args:
            context_length (int): Maximum sequence length the model can handle.
            embedding_dim (int): Dimension of the input embeddings.
            n_heads (int): Number of attention heads in multi-head attention.
            drop_rate (float): Dropout probability for regularization.
            qkv_bias (bool): Whether to include bias terms in QKV linear projections.
            window_size (int | None): Size of the sliding window for KV cache.
                If None, uses max_seq_len.

        """
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(
            input_dim=embedding_dim,
            output_dim=embedding_dim,
            context_length=context_length,
            dropout=drop_rate,
            num_heads=n_heads,
            qkv_bias=qkv_bias,
            window_size=window_size if window_size else context_length,
        )
        self.feed_forward = FeedForward(embedding_dim=embedding_dim)
        self.layer_norm1 = LayerNorm(embedding_dim=embedding_dim)
        self.layer_norm2 = LayerNorm(embedding_dim=embedding_dim)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        """Apply the transformer block transformation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape
                (batch_size, sequence_length, embedding_dim).
            use_cache (bool): Enable using KV cache. Defaults to False.

        Returns:
            torch.Tensor: Output tensor of the same shape as input.

        """
        # First residual connection with attention
        shortcut = x
        x = self.layer_norm1(x)
        x = self.multi_head_attention(x, use_cache=use_cache)
        x = self.dropout(x)
        x = x + shortcut

        # Second residual connection with feed forward
        shortcut = x
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = x + shortcut

        return x
