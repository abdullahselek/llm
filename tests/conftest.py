"""PyTest Configurations."""

import pytest

from llm.utils import create_vocab


@pytest.fixture()
def vocab() -> dict:
    """Vocablary dict."""
    return create_vocab()
