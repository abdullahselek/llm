"""Model post training Supervised Fine Tuning module."""

import argparse
import logging
import sys
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
log = logging.getLogger("llm.sft")


def format_instruction(sample: dict) -> str:
    """Format a sample into an instruction prompt.

    Format:
    ### Instruction:
    {instruction}

    ### Input:
    {input} (optional)

    ### Response:
    {output}

    Args:
        sample (dict): An item from dataset.

    Returns:
        Formatted instruction prompt.

    """
    instruction = sample.get("instruction", "")
    input = sample.get("input", "")
    output = sample.get("output", "")

    if input:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input}\n\n"
            f"### Response:\n{output}<|endoftext|>"
        )
    else:
        return f"### Instruction:\n{instruction}\n\n### Response:\n{output}<|endoftext|>"


def process_sft_dataset(dataset: Dataset) -> Dataset:
    """Map the dataset to the instruction format.

    Args:
        dataset (Dataset): HuggingFace dataset object.

    Returns:
        Processed HuggingFace dataset object.

    """

    def _add_content_column(examples):
        return {"content": format_instruction(examples)}

    return dataset.map(_add_content_column)


def load_checkpoint(model: LLM, checkpoint_path: Path, device: torch.device):
    """Load pretrained weights into the model.

    Args:
        model (LLM): Intance of LLM.
        checkpoint_path (Path): Model checkpoint path.
        device (torch.device): Torch device.

    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    log.info(f"Loading checkpoint from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    log.info("Checkpoint loaded successfully.")


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

        outputs = model(input_ids, labels=target_ids)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 50 == 0:
            log.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    return total_loss / num_batches


def validate(model: LLM, dataloader: DataLoader, device: torch.device) -> float:
    """Validate model performance.

    Args:
        model (LLM): Instance of LLM.
        dataloader (DataLoader): Validation dataloader.
        device (torch.device): Torch device.

    """
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
    """Post train with SFT and validate model."""
    parser = argparse.ArgumentParser(description="Trigger LLM Post Training (SFT).")

    cfg_path = Path.cwd() / "src/llm/configs/llm_3.55b.yaml"

    parser.add_argument(
        "--config-path", type=Path, default=cfg_path, help="Path of LLM config"
    )
    parser.add_argument(
        "--pretrained-path", type=Path, required=True, help="Path to pre-trained .pt file"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="iamtarun/python_code_instructions_18k_alpaca",
        help="HF Dataset name",
    )

    args = parser.parse_args()

    log.info("Starting SFT process...")

    log.info(f"Downloading dataset {args.dataset_name}...")
    try:
        dataset = load_dataset(args.dataset_name, split="train")
        log.info(f"Dataset loaded: {len(dataset)} instructions.")
    except Exception as e:
        log.error(f"Failed to load dataset: {e}")
        return

    log.info("Formatting dataset to instruction template...")
    try:
        dataset = dataset.map(lambda x: {"content": format_instruction(x)})
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = dataset["train"]
        val_dataset = dataset["test"]
    except Exception as e:
        log.error(f"Failed to process dataset: {e}")
        return

    log.info("Creating data loaders...")
    try:
        max_length, stride = 512, 128
        batch_size, num_workers = 4, 2

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
    except Exception as e:
        log.error(f"Failed to create data loaders: {e}")
        return

    try:
        config = read_config(args.config_path)
        model = LLM(config=config)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        load_checkpoint(model, args.pretrained_path, device)

        model.to(device)
        log.info(f"Model loaded and moved to {device}")
    except Exception as e:
        log.error(f"Failed to initialize model: {e}")
        return

    try:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-5,
            weight_decay=0.01,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    except Exception as e:
        log.error(f"Failed to initialize optimizer: {e}")
        return

    num_epochs = 2
    best_val_loss = float("inf")

    epoch_pbar = tqdm(range(num_epochs), desc="SFT Epochs")
    for epoch in range(num_epochs):
        log.info(f"Starting epoch {epoch + 1}/{num_epochs}")

        train_loss = train_one_epoch(model, train_dataloader, optimizer, device, epoch + 1)
        log.info(f"Epoch {epoch + 1} - Training Loss: {train_loss:.4f}")

        val_loss = validate(model, val_dataloader, device)
        log.info(f"Epoch {epoch + 1} - Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            model_save_path = Path.cwd() / "models" / f"sft_llm_epoch_{epoch + 1}.pt"
            model_save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(model_save_path))
            log.info(f"Best SFT model saved to {model_save_path}")

        scheduler.step()
        epoch_pbar.set_postfix({"Val Loss": f"{val_loss:.4f}"})

    log.info("Post training completed successfully!")


if __name__ == "__main__":
    main()
