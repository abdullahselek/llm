"""Large Langauge Model."""

import torch
import torch.nn as nn

from llm.layer_norm import LayerNorm
from llm.transformer_block import TransformerBlock


class LLM(nn.Module):
    """Large Language Model implementing the transformer architecture for text generation.

    The model accepts an input tensor contains token indices and that the
    positional embeddings are indexed by position in the sequence. The output logits
    can be used directly for token prediction or with additional softmax for probability
    distributions.
    """

    def __init__(self, config: dict):
        """Initialize an LLM object.

        Args:
            config (dict): Configuration dictionary containing model hyperparameters:
                - vocab_size (int): Size of the vocabulary
                - context_length (int): Maximum sequence length the model can handle
                - embedding_dim (int): Dimension of token embeddings
                - n_heads (int): Number of attention heads in transformer blocks
                - drop_rate (float): Dropout probability for regularization
                - qkv_bias (bool): Whether to include bias terms in QKV projections

        """
        super().__init__()
        self.token_emb = nn.Embedding(config["vocab_size"], config["embedding_dim"])
        self.positional_emb = nn.Embedding(config["context_length"], config["embedding_dim"])
        self.dropout = nn.Dropout(config["drop_rate"])

        self.transfomer_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    context_length=config["context_length"],
                    embedding_dim=config["embedding_dim"],
                    n_heads=config["n_heads"],
                    drop_rate=config["drop_rate"],
                    qkv_bias=config["qkv_bias"],
                )
                for _ in range(config["n_layers"])
            ]
        )

        self.final_norm = LayerNorm(config["embedding_dim"])
        self.out_head = nn.Linear(config["embedding_dim"], config["vocab_size"], bias=False)

    def forward(self, in_x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the language model.

        Args:
            in_x (torch.Tensor): Input tensor of token indices with shape
                (batch_size, sequence_length).

        Returns:
            torch.Tensor: Logits tensor with shape (batch_size, sequence_length, vocab_size)

        """
        batch_size, seq_len = in_x.shape
        token_embeds = self.token_emb(in_x)
        pos_embeds = self.positional_emb(torch.arange(seq_len, device=in_x.device))
        x = token_embeds + pos_embeds  # (batch_size, num_tokens, emb_size)
        x = self.dropout(x)
        x = self.transfomer_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
