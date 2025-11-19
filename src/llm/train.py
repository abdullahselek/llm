"""Model training module."""

import os

from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

dataset = load_dataset(
    "bigcode/starcoderdata",
    data_dir="python",
    split="train",
    token=os.getenv("HF_TOKEN"),
)
