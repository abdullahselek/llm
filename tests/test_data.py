"""Test data module that generate input embeddings."""

from llm.data import get_input_embeddings


def test_get_input_embeddings_with_default_parameters(vocab_text: str):
    """Test input embeddings shape."""
    input_embeddings = get_input_embeddings(text=vocab_text)

    assert input_embeddings[0].shape == (8, 4, 256)


def test_get_input_embeddings_with_custom_parameters(vocab_text: str):
    """Test input embeddings shape when custom parameters are set."""
    input_embeddings = get_input_embeddings(
        text=vocab_text,
        output_dim=512,
        context_length=2048,
        batch_size=4,
        max_length=4,
    )

    assert input_embeddings[0].shape == (4, 4, 512)
