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


def main():
    """Train and validate model."""
    log.info("Starting training process...")

    log.info("Dataset is going to be downloaded...")
    start_time = time.perf_counter()

    try:
        dataset = load_dataset(
            "bigcode/starcoderdata",
            data_dir="python",
            split="train",
            token=os.getenv("HF_TOKEN"),
        )
        log.info(f"Dataset loaded successfully with {len(dataset)} samples")
    except Exception as e:
        log.error(f"Failed to load dataset: {e}")
        return

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    log.info(f"Dataset loaded from HF in {elapsed_time:.2f} seconds.")

    log.info("Dataset is going to be processed...")
    start_time = time.perf_counter()

    try:
        train_dataset, val_dataset = split_dataset(dataset)
        train_data, val_data = process_dataset(train_dataset), process_dataset(val_dataset)
        log.info(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")
    except Exception as e:
        log.error(f"Failed to process dataset: {e}")
        return

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    log.info(f"Dataset processed for training and validation in {elapsed_time:.2f} seconds.")

    log.info("Creating data loaders...")
    start_time = time.perf_counter()

    try:
        train_dataloader = create_llm_dataloader(
            texts=train_data,
            batch_size=4,
            max_length=2048,
            stride=1024,
            shuffle=True,
            drop_last=True,
            num_workers=4,
        )

        val_dataloader = create_llm_dataloader(
            texts=val_data,
            batch_size=4,
            max_length=2048,
            stride=1024,
            shuffle=False,
            drop_last=True,
            num_workers=4,
        )

        log.info("Data loaders created successfully")
    except Exception as e:
        log.error(f"Failed to create data loaders: {e}")
        return

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    log.info(f"Data loaders loaded in {elapsed_time:.2f} seconds.")

    cfg_path = Path.cwd() / "src/llm/configs/llm_1.7b.yaml"

    try:
        config = read_config(cfg_path)
        model = LLM(config=config)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        log.info(f"Model moved to {device}")
    except Exception as e:
        log.error(f"Failed to initialize model: {e}")
        return

    try:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=5e-5,
            weight_decay=0.01,
        )

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

        log.info("Optimizer initialized successfully")
    except Exception as e:
        log.error(f"Failed to initialize optimizer: {e}")
        return

    num_epochs = 100
    best_val_loss = float("inf")

    log.info("Starting training...")
    for epoch in range(num_epochs):
        log.info(f"Starting epoch {epoch + 1}/{num_epochs}")

        train_loss = train_one_epoch(model, train_dataloader, optimizer, device, epoch + 1)
        log.info(f"Epoch {epoch + 1} - Training Loss: {train_loss:.4f}")

        val_loss = validate(model, val_dataloader, device)
        log.info(f"Epoch {epoch + 1} - Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save model checkpoint
            model_save_path = Path.cwd() / "models" / f"best_model_epoch_{epoch + 1}.pt"
            model_save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(model_save_path))
            log.info(f"Best model saved to {model_save_path}")

        scheduler.step()

    log.info("Training completed successfully!")


if __name__ == "__main__":
    main()
