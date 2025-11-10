"""Feed Forward."""

import torch
import torch.nn as nn

from llm.gelu import GELU


class FeedForward(nn.Module):
    """Feed Forward neural network layer for transformer architecture.

    This implementation follows the standard transformer feed-forward pattern with
    a bottleneck architecture that expands the input dimension by 4x before contracting
    it back.

    The architecture consists of:
    1. Linear projection from embedding dimension to 4× embedding dimension.
    2. GELU activation function.
    3. Linear projection back to original embedding dimension.

    This design allows the model to learn complex nonlinear transformations while maintaining
    computational efficiency through the bottleneck structure.
    """

    def __init__(self, embedding_dim: int):
        """Initialize FeedForward object.

        Args:
            embedding_dim (int): The dimension of the input embeddings.

        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            GELU(),
            nn.Linear(embedding_dim * 4, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed forward transformation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape
                (batch_size, sequence_length, embedding_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, embedding_dim).

        """
        return self.layers(x)
