"""Self Attention Layer."""

import torch


class SelfAttention(torch.nn.Module):
    """SelfAttention layer.

    It allows each token in a sequence to dynamically weigh the importance of every
    other token when generating representations, enabling the model to capture long
    range dependencies and contextual relationships regardless of their distance in
    the input sequence.
    """

    def __init__(self, input_dim: int, output_dim: int, qkv_bias: bool = False):
        """Initialize SelfAttention.

        Args:
            input_dim (int): The input dimension of the token embeddings.
            output_dim (int): The output dimension of the attention projections.
            qkv_bias (bool): Whether to include bias terms in the query, key
                and value linear transformations.

        """
        super().__init__()
        self.W_query = torch.nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.W_key = torch.nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.W_value = torch.nn.Linear(input_dim, output_dim, bias=qkv_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the self attention mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, dim_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, dim_out)
                         representing the contextualized token representations.

        """
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec
