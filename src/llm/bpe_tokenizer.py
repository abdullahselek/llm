"""BytePair Encoder used by GPT-2."""

import tiktoken


class BPETokenizer:
    """BytePair Tokenizer. tiktoken package is preffered to get performance of Rust."""

    def __init__(self):
        """Initialize."""
        self.tokenizer = tiktoken.get_encoding("o200k_base")
        print(f"Vocabulary size: {self.tokenizer.n_vocab}")

    def encode(self, text: str) -> list[int]:
        """Encode given text and return tokens.

        Args:
            text (str): String to be encoded.

        Returns:
            List of ids.

        """
        ids = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode given ids.

        Args:
            ids (list[int]): List of ids.

        Returns:
            Decoded string.

        """
        return self.tokenizer.decode(ids)
