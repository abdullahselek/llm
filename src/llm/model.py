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

        self.transfomer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    context_length=config["context_length"],
                    embedding_dim=config["embedding_dim"],
                    n_heads=config["n_heads"],
                    drop_rate=config["drop_rate"],
                    qkv_bias=config["qkv_bias"],
                    window_size=config["kv_window_size"],
                )
                for _ in range(config["n_layers"])
            ]
        )

        self.final_norm = LayerNorm(config["embedding_dim"])
        self.out_head = nn.Linear(config["embedding_dim"], config["vocab_size"], bias=False)
        self.ptr_current_pos = 0

    def forward(self, in_x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        """Forward pass through the language model.

        Args:
            in_x (torch.Tensor): Input tensor of token indices with shape
                (batch_size, sequence_length).
            use_cache (bool): Enable using KV cache. Defaults to False.

        Returns:
            torch.Tensor: Logits tensor with shape (batch_size, sequence_length, vocab_size)

        """
        batch_size, seq_len = in_x.shape
        token_embeds = self.token_emb(in_x)

        if use_cache:
            pos_ids = torch.arange(
                self.ptr_current_pos,
                self.ptr_current_pos + seq_len,
                device=in_x.device,
                dtype=torch.long,
            )
            self.ptr_current_pos += seq_len
        else:
            pos_ids = torch.arange(0, seq_len, device=in_x.device, dtype=torch.long)

        pos_embeds = self.positional_emb(pos_ids).unsqueeze(0)
        x = token_embeds + pos_embeds  # (batch_size, num_tokens, emb_size)
        x = self.dropout(x)

        for trfmr_blk in self.transfomer_blocks:
            x = trfmr_blk(x, use_cache=use_cache)

        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def generate_text(
    model: LLM, idx: torch.Tensor, max_new_tokens: int, context_size: int
) -> torch.Tensor:
    """Generate text using greedy decoding with the given model.

    This function performs iterative text generation by repeatedly predicting the next token
    and appending it to the input sequence until the maximum number of tokens is generated.

    Args:
        model (LLM): The trained language model to use for generation.
        idx (torch.Tensor): Input tensor of token indices with shape
            (batch_size, sequence_length).
        max_new_tokens (int): Maximum number of new tokens to generate.
        context_size (int): Maximum context length to maintain during generation.

    Returns:
        torch.Tensor: Generated text tensor with shape
            (batch_size, original_length + max_new_tokens).

    """
    for _ in range(max_new_tokens):
        # Crop current context if it exceeds context size
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        # (batch, n_token, vocab_size) ->  (batch_size, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of vocablary with highest logits value
        # (batch, 1)
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # Append sampled index to the running sequence
        # (batch, n_tokens+1)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
