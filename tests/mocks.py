"""Module that contains Mock classes."""


class MockHFDataset(list):
    """Simulates a HuggingFace dataset object (list like with select/shuffle)."""

    def select(self, indices: list[int]):
        """Select items of dataset by indices."""
        return MockHFDataset([self[i] for i in indices])

    def shuffle(self, seed: int | None = None):
        """Return self to keep order deterministic."""
        return self
