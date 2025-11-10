"""Gaussian Error Linear Unit."""

import torch
import torch.nn as nn


class GELU(nn.Module):
    """Gaussian Error Linear Unit activation function.

    GELU is a smooth approximation of the rectifier and is defined as:
    GELU(x) = x * Φ(x) where Φ(x) is the cumulative distribution function of the
    standard normal distribution.

    This implementation uses the approximate form:
    GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

    The approximation is computationally efficient and provides good performance
    while maintaining the desirable properties of the exact GELU function.
    """

    def __init__(self):
        """Initialize a GELU object."""
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the GELU activation function to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape

        Returns:
            torch.Tensor: Output tensor with the same shape as input

        """
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )
