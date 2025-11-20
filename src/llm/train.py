"""Model training module."""

import os
from pathlib import Path

from datasets import Dataset, load_dataset
from dotenv import load_dotenv

from llm.dataset import create_llm_dataloader
from llm.model import LLM
from llm.utils import read_config

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


def process_dataset(dataset: Dataset) -> list[str]:
    """Preproces HF codestart dataset.

    Args:
        dataset (Dataset): HF dataset object.

    Returns:
        List of code contents.

    """
    code_contents: list[str] = []
    for i in range(dataset.num_rows):
        code_contents.append(dataset[i]["content"])
    return code_contents


if __name__ == "__main__":
    dataset = load_dataset(
        "bigcode/starcoderdata",
        data_dir="python",
        split="train",
        token=os.getenv("HF_TOKEN"),
    )

    train_dataset, val_dataset = split_dataset(dataset)
    train_data, val_data = process_dataset(train_dataset), process_dataset(val_dataset)

    train_dataloader = create_llm_dataloader(
        texts=train_data,
        batch_size=4,
        max_length=256,
        stride=128,
        shuffle=True,
        drop_last=True,
        num_workers=4,
    )
    val_dataloader = create_llm_dataloader(
        texts=val_data,
        batch_size=4,
        max_length=256,
        stride=128,
        shuffle=True,
        drop_last=True,
        num_workers=4,
    )

    cfg_path = Path.cwd() / "src/llm/configs/llm_1.7b.yaml"
    config = read_config(cfg_path)

    model = LLM(config=config)
