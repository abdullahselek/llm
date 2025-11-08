"""Multi Head Attention Layer."""

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """MultiHeadAttention layer for transformer architectures.

    This module implements multihead attention mechanism where the input is
    projected into multiple heads, each computing scaled dot product attention separately,
    and then concatenated and linearly transformed back to the original dimension.

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

        Raises:
            AssertionError: If output_dim is not divisible by num_heads.

        """
        super().__init__()
        assert output_dim % num_heads == 0, "output_dim must be divisible by num_heads"

        self.output_dim = output_dim
        self.context_length = context_length
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads

        # self.W_query = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        # self.W_key = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        # self.W_value = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.qkv = nn.Linear(input_dim, output_dim * 3, bias=qkv_bias)
        self.out_proj = nn.Linear(output_dim, output_dim)
        self.dropout = dropout
        # self.dropout = nn.Dropout(dropout)
        # self.mask: torch.Tensor = torch.triu(
        #     torch.ones(context_length, context_length), diagonal=1
        # ).bool()

    def forward(self, x: torch.Tensor):
        """Compute multi head causal attention on the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape
                (batch_size, sequence_length, output_dim).

        """
        # batch, num_tokens, input_dim = x.shape
        #
        # keys = self.W_key(x)  # output shape (batch, num_tokens, output_dim)
        # queries = self.W_query(x)
        # values = self.W_value(x)
        #
        # # We implicitly split the matrix by adding a `num_heads` dimension
        # # Unroll last dim:
        # # (batch, num_tokens, output_dim) -> (batch, num_tokens, num_heads, head_dim)
        # keys = keys.view(batch, num_tokens, self.num_heads, self.head_dim)
        # queries = queries.view(batch, num_tokens, self.num_heads, self.head_dim)
        # values = values.view(batch, num_tokens, self.num_heads, self.head_dim)
        #
        # Transpose
        # (batch, num_tokens, num_heads, head_dim) -> (batch, num_heads, num_tokens, head_dimd)
        # keys = keys.transpose(1, 2)
        # queries = queries.transpose(1, 2)
        # values = values.transpose(1, 2)
        #
        # # Scaled dot-product attention with causal mask
        # attn_scores = queries @ keys.transpose(2, 3)
        #
        # # Original mask truncated to the number of tokens and converted to boolean
        # mask_bool = self.mask[:num_tokens, :num_tokens]
        #
        # # Use the mask to fill attention scores
        # attn_scores.masked_fill_(mask_bool, -torch.inf)
        #
        # attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        #
        # # Shape (batch, num_tokens, num_heads, head_dim)
        # context_vec = (attn_weights @ values).transpose(1, 2)
        #
        # # Combine heads, self.output_dim = self.num_heads * self.head_dim
        # context_vec = context_vec.contiguous().view(batch, num_tokens, self.output_dim)
        # context_vec = self.out_proj(context_vec)
        #
        # return context_vec

        batch_size, num_tokens, embed_dim = x.shape

        # (batch_size, num_tokens, embed_dim) -> (batch_size, num_tokens, embed_dim * 3)
        qkv = self.qkv(x)

        # (batch_size, num_tokens, embed_dim * 3) ->
        # (batch_size, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # (batch_size, num_tokens, 3, num_heads, head_dim) ->
        # (3, batch_size, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, batch_size, num_heads, num_tokens, head_dim) ->
        # 3 times (batch_size, num_heads, num_tokens, head_dim)
        queries, keys, values = qkv

        use_droput = 0.0 if self.training else self.dropout

        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=None, dropout_p=use_droput, is_causal=True
        )
        # Combine head where self.output_dim = self.num_heads * self.head_dim
        context_vec = (
            context_vec.transpose(1, 2)
            .contiguous()
            .view(batch_size, num_tokens, self.output_dim)
        )
        context_vec = self.out_proj(context_vec)

        return context_vec
