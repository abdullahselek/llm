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
from tqdm import tqdm

from llm.dataset import create_llm_dataloader_from_dataset
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
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset["train"]
    val_dataset = dataset["test"]

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

    progress_bar = tqdm(dataloader, desc="Validation")
    with torch.no_grad():
        for input_ids, target_ids in dataloader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            outputs = model(input_ids, labels=target_ids)
            loss = outputs.loss

            total_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

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
        log.info(
            f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}"
        )
    except Exception as e:
        log.error(f"Failed to process dataset: {e}")
        return

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    log.info(f"Dataset processed for training and validation in {elapsed_time:.2f} seconds.")

    log.info("Creating data loaders...")
    start_time = time.perf_counter()

    try:
        max_length, stride = 128, 64
        batch_size, num_workers = 8, 2

        train_dataloader = create_llm_dataloader_from_dataset(
            dataset=train_dataset,
            batch_size=batch_size,
            max_length=max_length,
            stride=stride,
            shuffle=True,
            num_workers=num_workers,
        )

        val_dataloader = create_llm_dataloader_from_dataset(
            dataset=val_dataset,
            batch_size=batch_size,
            max_length=max_length,
            stride=stride,
            shuffle=False,
            num_workers=num_workers,
        )

        log.info("Data loaders created successfully")
    except Exception as e:
        log.error(f"Failed to create data loaders: {e}")
        return

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    log.info(f"Data loaders loaded in {elapsed_time:.2f} seconds.")

    cfg_path = Path.cwd() / "src/llm/configs/llm_3.55b.yaml"

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

    num_epochs = 3
    best_val_loss = float("inf")

    log.info("Starting training...")
    epoch_pbar = tqdm(range(num_epochs), desc="Training Epochs")
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
        epoch_pbar.set_postfix({"Val Loss": f"{val_loss:.4f}"})

    log.info("Training completed successfully!")


if __name__ == "__main__":
    main()
