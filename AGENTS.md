# AGENTS.md

## Project Overview
CS336 Spring 2025 Assignment 1: Transformer language model implementation with BPE tokenizer training. Students implement core components in `cs336_basics/` and connect them via `tests/adapters.py`.

## Key Commands

### Environment & Dependencies
```bash
uv run <python_file>  # Auto-solves and activates environment
uv sync               # Install/update dependencies
```

### Testing
```bash
uv run pytest                    # Run all tests
uv run pytest -v ./tests         # Verbose test output
uv run pytest tests/test_tokenizer.py  # Single test file
uv run pytest -k "test_name"     # Run specific test by name
```

### Submission
```bash
./make_submission.sh  # Creates zip submission file
```

## Architecture Notes

### Module Structure
- `cs336_basics/modules.py`: Linear, Embedding, RMSNorm, SwiGLU, RoPE, attention modules
- `cs336_basics/tokenizer.py`: BPE tokenizer with training and encoding/decoding
- `cs336_basics/loss.py`: Cross-entropy loss implementation
- `cs336_basics/pretokenization_example.py`: Chunk boundary finding for parallel pretokenization

### Test Adapter Pattern
All implementations must be connected through `tests/adapters.py`. This file imports from `cs336_basics` and provides wrapper functions that tests call. Students must implement the `NotImplementedError` functions in this file.

### Snapshot Testing
Tests use snapshot testing (`tests/conftest.py`):
- `numpy_snapshot` fixture for NumPy arrays (`.npz` files in `tests/_snapshots/`)
- `snapshot` fixture for arbitrary data (`.pkl` files)
- Update snapshots with `--update-snapshots` flag (solution only)

## Important Conventions

### Tensor Type Annotations
Use `jaxtyping` annotations:
```python
from jaxtyping import Float, Int, Bool
from torch import Tensor

def func(x: Float[Tensor, "batch seq d_model"]) -> Float[Tensor, "batch seq d_model"]:
```

### Dependencies
- `einops`: Tensor operations (`rearrange`, `einsum`, `reduce`, `repeat`)
- `jaxtyping`: Type annotations for tensors
- `sortedcontainers`: For BPE tokenizer
- `regex`: Enhanced regex for pretokenization patterns

### Testing Quirks
- Tests import from `cs336_basics` modules directly
- Some tests use fixtures from `tests/fixtures/` (GPT-2 vocab/merges)
- Memory-limited tests exist for tokenizer (Linux only)
- Snapshot tests compare against reference implementations

## Data Setup
Download required data before running experiments (check if data exists first to avoid unnecessary downloads):
```bash
mkdir -p data && cd data

# Check and download TinyStories data
if [ ! -f "TinyStoriesV2-GPT4-train.txt" ]; then
    wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
fi
if [ ! -f "TinyStoriesV2-GPT4-valid.txt" ]; then
    wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
fi

# Check and download OpenWebText data
if [ ! -f "owt_train.txt" ]; then
    wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
    gunzip owt_train.txt.gz
fi
if [ ! -f "owt_valid.txt" ]; then
    wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
    gunzip owt_valid.txt.gz
fi

cd ..
```

## Common Pitfalls
1. **Adapter functions**: Must implement all `NotImplementedError` functions in `tests/adapters.py`
2. **State dict keys**: Match exactly what tests expect (see adapter function docstrings)
3. **RoPE dimensions**: Use `d_model // num_heads` for attention head dimension
4. **Numerical stability**: Use float32 for RMSNorm calculations, then cast back
5. **BPE training**: Follow GPT-2 pretokenization pattern (`PAT` in tokenizer.py)

## Code Style
- Line length: 120 characters (ruff)
- Use `einops` for tensor operations where possible
- Follow existing patterns in `modules.py` for new components
- Type annotations required for all public functions