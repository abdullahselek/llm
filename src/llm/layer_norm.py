"""Normalization Layer of LLM."""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Normalization layer for transformer architectures.

    It normalizes the inputs across the feature dimensions while
    maintaining learnable scale and shift parameters.

    Normalization layer computes the mean and variance across the last dimension(s)
    of the input tensor, then applies a learned scaling and shifting transformation.
    This helps stabilize training and improves convergence in deep neural networks.
    """

    def __init__(self, embedding_dim: int, eps: float = 1e-5):
        """Initialize Normalization layer.

        Args:
            embedding_dim (int): The dimension of the embeddings to normalize.
            eps (float): Small constant for numerical stability during normalization.
                Defaults to 1e-5.

        """
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(embedding_dim))
        self.shift = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape
                (batch_size, sequence_length, embedddings_dim)
                or (batch_size, embeddding_dim) for 2D inputs.

        Returns:
            torch.Tensor: Normalized tensor of the same shape as input.

        """
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(variance + self.eps)
        return self.scale * norm_x + self.shift
