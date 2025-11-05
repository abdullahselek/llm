"""Simple Tokenizer that uses Regex."""

import re

ENCODE_PATTERN = r'([,.:;?_!"()\']|--|\s)'
DECODE_PATTERN = r'\s+([,.?!"()\'])'


class SimpleTokenizer:
    """SimpleTokenizer that uses regular experession."""

    def __init__(self, vocab: dict):
        """Initialize.

        Args:
            vocab (dict): Vocablary.

        """
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> list[int]:
        """Encode given text and return ids.

        Args:
            text (str): String value to be encoded.

        Returns:
            List of ids.

        """
        preprocessed = re.split(ENCODE_PATTERN, text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>" for item in preprocessed
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode given ids.

        Args:
            ids (list[int]): List of ids.

        Returns:
            Decoded string.

        """
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(DECODE_PATTERN, r"\1", text)
        return text
