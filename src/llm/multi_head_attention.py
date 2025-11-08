"""Multi Head Attention Layer."""

import torch
import torch.nn as nn

from llm.causal_attention import CausalAttention


class MultiHeadAttention(nn.Module):
    """MultiHeadAttention layer for transformer architectures.

    This module implements MultiHead CausalAttention mechanism where the input is
    projected into multiple heads, each computing attention separately, and then
    concatenated and linearly transformed back to the original dimension.

    The attention mechanism allows the model to focus on different positions of the input
    sequence when processing each position, enabling better context understanding.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ):
        """Initialize MultiHeadAttention with multiple causal attention heads.

        Args:
            input_dim (int): The dimension of the input embeddings.
            output_dim (int): The dimension of the output embeddings.
            context_length (int): The maximum sequence length that the attention can consider.
            dropout (float): Dropout probability for attention weights and outputs.
            num_heads (int): Number of parallel attention heads to use.
            qkv_bias (bool, optional): Whether to include bias terms in the query, key, and
                value linear transformations. Defaults to False.

        Note:
            Each head operates independently on the input, applying causal attention
            within its own parameter space. The outputs from all heads are concatenated
            along the feature dimension.

        """
        super().__init__()
        self.heads = nn.ModuleList(
            [
                CausalAttention(input_dim, output_dim, context_length, dropout, qkv_bias)
                for _ in range(num_heads)
            ]
        )

    def forward(self, x: torch.Tensor):
        """Compute multi head causal attention on the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape
                (batch_size, sequence_length, num_heads * output_dim).

        Note:
            Each attention head processes the input independently with causal masking,
            and the results are concatenated along the feature dimension. This allows
            the model to attend to different parts of the sequence while maintaining
            causality (future tokens cannot influence past tokens).

        """
        return torch.cat([head(x) for head in self.heads], dim=-1)
