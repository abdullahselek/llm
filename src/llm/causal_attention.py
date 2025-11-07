"""Causal Attention Layer."""

import torch


class CausalAttention(torch.nn.Module):
    """Causal Attention Layer.

    It prevents tokens from attending to future tokens in a sequence,
    enabling autoregressive language modeling.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        context_length: int,
        dropout: float,
        qkv_bias: bool = False,
    ):
        """Initialize CausalAttention.

        Args:
            input_dim (int): The input dimension of the token embeddings.
            output_dim (int): The output dimension of the attention projections.
            context_length (int): The maximum sequence length that the attention can handle.
            dropout (float): The dropout probability for attention weights.
            qkv_bias (bool): Whether to include bias terms in the query, key, and value linear
                transformations.

        """
        super().__init__()
        self.W_query = torch.nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.W_key = torch.nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.W_value = torch.nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.dropout = torch.nn.Dropout(dropout)
        self.mask: torch.Tensor = torch.triu(
            torch.ones(context_length, context_length), diagonal=1
        ).bool()

    def forward(self, x: torch.Tensor):
        """Compute the causal attention mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, output_dim)
                representing the contextualized token representations with causal masking
                applied.

        """
        batch, num_tokens, input_dim = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(self.mask[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec
