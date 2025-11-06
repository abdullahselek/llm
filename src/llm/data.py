"""Data module that creates dataset and dataloader."""

import logging
import sys

import torch

from llm.dataset import create_dataloader

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(name)s $(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("llm.data")


def get_input_embeddings(
    text: str,
    output_dim: int = 256,
    context_length: int = 1024,
    batch_size: int = 8,
    max_length: int = 4,
) -> list[torch.Tensor]:
    """Get input embeddings from given vocablary text.

    Args:
        text (str): Raw vocablary text.
        output_dim (int): Dimensionality of the embedding vectors, default 256.
        context_length (int): Maximum sequence length the model can handle, default 1024.
        batch_size (int): Batch size, default 8.
        max_length (int): Chunk length, default 4.

    Returns:
        List of sum of token and positional embeddings creates the final input representation.

    """
    vocab_size = 50257
    log.info(f"Vocablary size: {vocab_size}")

    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

    dataloader = create_dataloader(
        text,
        batch_size=batch_size,
        max_length=max_length,
        stride=max_length,
    )

    input_embeddings = []
    for batch in dataloader:
        x, y = batch
        token_embeddings = token_embedding_layer(x)
        pos_embeddings = pos_embedding_layer(torch.arange(max_length))

        input_embeddings.append(token_embeddings + pos_embeddings)

    return input_embeddings
