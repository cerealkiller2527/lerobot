# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Testing
```bash
# Run end-to-end tests (Makefile based)
make test-end-to-end DEVICE=cpu  # or DEVICE=cuda

# Run specific policy tests
make test-act-ete-train DEVICE=cpu
make test-diffusion-ete-train DEVICE=cpu
make test-tdmpc-ete-train DEVICE=cpu

# Run tests with pytest
pytest tests/                     # Run all tests
pytest tests/ --timeout=300       # With timeout
pytest tests/ --cov=lerobot       # With coverage
pytest tests/test_specific.py     # Run specific test file
```

### Linting and Code Quality
```bash
# Format code with Ruff
ruff format src/lerobot tests/

# Check and fix linting issues
ruff check --fix src/lerobot tests/

# Run pre-commit hooks
pre-commit run --all-files

# Security checks
bandit -c pyproject.toml -r src/

# Check for typos
typos src/lerobot
```

### Training and Evaluation
```bash
# Train a policy
lerobot-train --policy.type=act --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human

# Evaluate a trained model
lerobot-eval --policy.path=outputs/train/*/checkpoints/last/pretrained_model

# Visualize dataset
python -m lerobot.scripts.visualize_dataset --repo-id lerobot/pusht --episode-index 0
```

## High-Level Architecture

### Core Components

**Policies** (`src/lerobot/policies/`)
- Modular policy implementations (ACT, Diffusion, TDMPC, SAC, Pi0, SmolVLA, VQ-BeT)
- Each policy has configuration and modeling modules
- Factory pattern for policy instantiation via `factory.py`
- Policies inherit from `PreTrainedPolicy` base class

**Datasets** (`src/lerobot/datasets/`)
- `LeRobotDataset` format for robot data with temporal frame support
- HuggingFace datasets integration (Arrow/Parquet backed)
- Video compression support (mp4) for efficient storage
- Statistics computation and transforms for normalization

**Robots** (`src/lerobot/robots/`)
- Support for various robot platforms (SO-100/101, HopeJR, Koch, LeKiwi, Stretch3, ViperX)
- Motor control abstractions (Dynamixel, Feetech)
- Calibration and teleoperation support

**Environments** (`src/lerobot/envs/`)
- Gymnasium-based simulation environments
- Support for ALOHA, PushT, XArm environments
- Environment factory for standardized creation

**Training Infrastructure** (`src/lerobot/scripts/`)
- Training script with distributed training support
- Evaluation and benchmarking utilities
- Wandb integration for experiment tracking
- Checkpoint management and model serialization

### Key Design Patterns

1. **Configuration-First**: Each policy has a configuration class defining all hyperparameters
2. **HuggingFace Integration**: Uses HF Hub for model/dataset hosting and HF datasets for data handling
3. **Modular Policies**: Policies are self-contained with standardized interfaces
4. **Temporal Data Handling**: Dataset supports multi-frame retrieval via `delta_timestamps`
5. **Video Encoding**: Efficient storage using mp4 compression for camera observations

### Important Paths
- Models saved to: `outputs/train/{date}/{time}_{task}/checkpoints/`
- Datasets cached in: `~/.cache/huggingface/lerobot/`
- Videos stored in: `{dataset_path}/videos/`

### Key Entry Points
- `lerobot-train`: Main training script
- `lerobot-eval`: Policy evaluation
- `lerobot-record`: Data collection from real robots
- `lerobot-teleoperate`: Robot teleoperation