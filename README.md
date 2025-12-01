# LLM Code

Designing and building a Large Language Model from scratch specifically for code generation.

## Requirements

- Python 3.13
- [uv](https://docs.astral.sh/uv/)

## Setting up development environment

After installing uv package and project manager, run the command below to create virtual environment and install all dependencies.

```bash
uv sync
```

## Developer Tasks Automation

This project uses [Nox](https://nox.thea.codes/en/stable/) for implementing and running dev automation tasks such as linting, formatting code and running tests etc. Current tasks are available in [noxfile.py](https://github.com/abdullahselek/llm/blob/main/noxfile.py).

Task that runs the unit tests:

```bash
uv run nox -s run_tests
```

## Training the model

The LLM uses an open source [dataset](https://huggingface.co/datasets/bigcode/starcoderdata) from HuggingFace, for now training script only loads Python specific code from the dataset.

Model configs are available at `src/llm/configs`. To start a new training with 3.55b model config simply run

```bash
uv run train-llm --config-path ./src/llm/configs/llm_3.55b.yaml --data-percentage 10
```
