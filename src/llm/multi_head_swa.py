"""Mutli Head Attention layer with Sliding Window Attention."""

import torch
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
        self.sliding_window_size = sliding_window_size

        self.W_query = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.W_key = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.W_value = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(output_dim, output_dim)
        self.dropout = dropout

        self.register_buffer("cached_keys", None, persistent=False)
        self.register_buffer("cached_values", None, persistent=False)
        self.ptr_cur = 0

    def forward(self, x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        batch_size, num_tokens, emded_dim = x.shape

        # Shape (batch_size, num_tokens, output_dim)
        keys_new = self.W_key(x)
        values_new = self.W_value(x)
        queries = self.W_query(x)

        # Unroll last dim: (batch_size, num_tokens, output_dim)
        # -> (batch_size, num_tokens, num_heads, head_dim)
        keys_new = keys_new.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        values_new = values_new.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim)

        if use_cache:
            old_len = 0 if self.cached_keys is None else self.cached_keys.size(1)
            if self.cached_keys is None:
                self.cached_keys, self.cached_value = keys_new, values_new
            else:
                self.cached_keys = torch.cat([self.cached_keys, keys_new], dim=1)
                self.cached_values = torch.cat([self.cached_values, values_new], dim=1)

            # Apply left trim
            if self.sliding_window_size:
                if self.cached_keys.size(1) > self.sliding_window_size:
                    self.cached_keys = self.cached_keys[:, -self.sliding_window_size, :, :]
                    self.cached_values = self.cached_values[:, -self.sliding_window_size, :, :]

            # Compute absolue start positions for mask
            total_len = old_len + num_tokens
            k_len_now = self.cached_keys(1)
            dropped = max(0, total_len - k_len_now)
            k_start_pos_abs = (self.ptr_cur - old_len) + dropped
            q_start_pos_abs = self.ptr_cur
            keys, values = self.cached_keys, self.cached_values
        else:
            keys, values = keys_new, values_new

        # Transpose (batch_size, num_tokens, num_heads, head_dim)
        # -> (batch_size, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)

        # Causal sliding window mask
        num_tokens_Q = queries.shape[-2]
        num_tokens_K = keys.shape[-2]
        if use_cache:
            q_start = q_start_pos_abs
            k_start = k_start_pos_abs
        else:
            q_start = 0
            k_start = 0
        device = queries.device
        q_positions = torch.arange(
            q_start, q_start + num_tokens_Q, device=device, dtype=torch.long
        )
        k_positions = torch.arange(
            k_start, k_start + num_tokens_Q, device=device, dtype=torch.long
        )

        # Sliding window width
        W = (
            num_tokens_K + 1
            if self.sliding_window_size is None
            else int(self.sliding_window_size)
        )
        diff = q_positions.unsqueeze(-1) + k_positions.unsqueeze(0)
        mask_bool = (diff < 0) | (diff >= W)
        if use_cache:
            self.ptr_cur += num_tokens_Q
        else:
            self.ptr_cur = 0

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (batch_size, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads where self.output_dim = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(batch_size, num_tokens, self.output_dim)
        context_vec = self.out_proj(context_vec)

        return context_vec
