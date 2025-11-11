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
        max_seq_len: int | None = None,
        window_size: int | None = None,
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
            max_seq_len (int | None): Maximum sequence length allowed.
                If None, uses context_length.
            window_size (int | None): Size of the sliding window for KV cache.
                If None, uses max_seq_len.

        Raises:
            AssertionError: If output_dim is not divisible by num_heads.

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

        self.max_seq_len = max_seq_len or context_length
        self.window_size = window_size or self.max_seq_len
        self.register_buffer("cached_keys", None, persistent=False)
        self.register_buffer("cached_values", None, persistent=False)
        self.ptr_cur = 0

    def forward(self, x: torch.Tensor, use_cache: bool = False):
        """Compute multi head causal attention on the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).
            use_cache (bool): Enable using KV cache. Defaults to False.

        Returns:
            torch.Tensor: Output tensor of shape
                (batch_size, sequence_length, output_dim).

        """
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
        queries, keys_new, values_new = qkv

        if use_cache:
            if self.cached_keys is None or self.cached_keys.size(0) != batch_size:
                self.cached_keys = torch.zeros(
                    batch_size,
                    self.num_heads,
                    self.window_size,
                    self.head_dim,
                    device=x.device,
                )
                self.cached_values = torch.zeros_like(self.cached_keys)
                self.ptr_cur = 0

            if self.ptr_cur + num_tokens > self.window_size:
                overflow = self.ptr_cur + num_tokens - self.window_size
                self.cached_keys[:, :, :-overflow, :] = self.cached_keys[
                    :, :, overflow:, :
                ].clone()
                self.cached_values[:, :, :-overflow, :] = self.cached_values[
                    :, :, overflow:, :
                ].clone()
                self.ptr_cur -= overflow

            self.cached_keys[:, :, self.ptr_cur : self.ptr_cur + num_tokens, :] = keys_new
            self.cached_values[:, :, self.ptr_cur : self.ptr_cur + num_tokens, :] = values_new
            self.ptr_cur += num_tokens

            keys = self.cached_keys[:, :, : self.ptr_cur, :]
            values = self.cached_values[:, :, : self.ptr_cur, :]
        else:
            keys, values = keys_new, values_new
            self.ptr_cur = 0

        use_dropout = self.dropout if self.training else 0.0

        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=None, dropout_p=use_dropout, is_causal=True
        )
        # Combine head where self.output_dim = self.num_heads * self.head_dim
        context_vec = (
            context_vec.transpose(1, 2)
            .contiguous()
            .view(batch_size, num_tokens, self.output_dim)
        )
        context_vec = self.out_proj(context_vec)

        return context_vec

    def reset_cache(self):
        """Reset the KV cache to initial state."""
        self.cached_keys, self.cached_values = None, None
