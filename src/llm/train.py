"""Model training module."""

import os

from datasets import Dataset, load_dataset
from dotenv import load_dotenv

load_dotenv()


def split_dataset(dataset: Dataset) -> tuple[Dataset, Dataset]:
    """Split dataset into training and validation sets.

    Args:
        dataset (Dataset): HF Dataset object.

    Returns:
        Training and validation sets.

    """
    total_size = len(dataset)
    split_point = int(0.8 * total_size)

    train_dataset = dataset.select(range(split_point))
    val_dataset = dataset.select(range(split_point, total_size))

    return train_dataset, val_dataset


def process_dataset(dataset: Dataset) -> list[dict[str, str]]:
    """Preproces HF codestart dataset.

    Args:
        dataset (Dataset): HF dataset object.

    Returns:
        List of dictionaries that contains code content.

    """
    code_contents = []
    for i in range(dataset.num_rows):
        code_contents.append({"text": dataset[i]["content"]})
    return code_contents


if __name__ == "__main__":
    dataset = load_dataset(
        "bigcode/starcoderdata",
        data_dir="python",
        split="train",
        token=os.getenv("HF_TOKEN"),
    )

    processed_data = process_dataset(dataset)
