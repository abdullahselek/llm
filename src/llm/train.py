"""Model training module."""

import logging
import os
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from llm.dataset import create_llm_dataloader
from llm.model import LLM
from llm.utils import read_config

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(name)s $(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("llm.train")


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


def train_one_epoch(
    model: LLM,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """Train model for one epoch.

    Args:
        model (LLM): Model object.
        dataloader (DataLoader): Torch dataloader object.
        optimizer (torch.optim.Optimizer): Torch optimizer.
        device (torch.device): Torch device.
        epoch (int): Epoch number.

    Returns:
        Loss value.

    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        # Forward pass
        outputs = model(input_ids, labels=target_ids)
        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 100 == 0:
            log.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    return total_loss / num_batches


def validate(model: LLM, dataloader: DataLoader, device: torch.device) -> float:
    """Validate model performance."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for input_ids, target_ids in dataloader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            outputs = model(input_ids, labels=target_ids)
            loss = outputs.loss

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


if __name__ == "__main__":
    log.info("Dataset is going to be dowmloaded...")
    start_time = time.perf_counter()
    dataset = load_dataset(
        "bigcode/starcoderdata",
        data_dir="python",
        split="train",
        token=os.getenv("HF_TOKEN"),
    )
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    log.info(f"Dataset loaded from HF in {elapsed_time:.2f} seconds.")

    log.info("Dataset is going to be splitted and processed...")
    start_time = time.perf_counter()
    train_dataset, val_dataset = split_dataset(dataset)
    train_data, val_data = process_dataset(train_dataset), process_dataset(val_dataset)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    log.info(
        """Dataset splitted and processed for training and validation in """
        f"""{elapsed_time:.2f} seconds."""
    )

    log.info("Train dataloader is going to be prepared...")
    start_time = time.perf_counter()
    train_dataloader = create_llm_dataloader(
        texts=train_data,
        batch_size=8,
        max_length=2048,
        stride=1024,
        shuffle=True,
        drop_last=True,
        num_workers=4,
    )
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    log.info(f"Train dataloader is loaded in {elapsed_time:.2f} seconds.")

    log.info("Validation dataloader is going to be prepared...")
    start_time = time.perf_counter()
    val_dataloader = create_llm_dataloader(
        texts=val_data,
        batch_size=8,
        max_length=2048,
        stride=1024,
        shuffle=True,
        drop_last=True,
        num_workers=4,
    )
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    log.info(f"Validation dataloader is loaded in {elapsed_time:.2f} seconds.")

    cfg_path = Path.cwd() / "src/llm/configs/llm_1.7b.yaml"
    config = read_config(cfg_path)

    model = LLM(config=config)
