# Installation

This guide walks you through installing SPIDER and its dependencies.

## Prerequisites

- **Python**: 3.12 or higher
- **CUDA**: 11.8 or higher (for GPU acceleration)
- **Git**: For cloning repositories
- **Git LFS**: For large file storage (datasets)

## Option 1: Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer that we recommend for development.

### Install uv

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Install SPIDER

```bash
# Clone the repository
git clone https://github.com/jc-bao/spider.git
cd spider

# Install Python 3.12 if not already available
uv python install 3.12

# Sync dependencies and install SPIDER
uv sync --python 3.12
pip install --ignore-requires-python --no-deps -e .
```

### Clone Example Datasets

```bash
# Install git-lfs
sudo apt install git-lfs  # Ubuntu/Debian
# brew install git-lfs    # macOS

git lfs install
git clone https://huggingface.co/datasets/retarget/retarget_example example_datasets
```

## Option 2: Using Conda

If you prefer conda for environment management:

```bash
# Clone the repository
git clone https://github.com/jc-bao/spider.git
cd spider

# Create conda environment
conda create -n spider python=3.12
conda activate spider

# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

## Verify Installation

Test your installation by running a simple example:

```bash
# With uv
uv run examples/run_mjwp.py

# With conda
python examples/run_mjwp.py
```

If everything is installed correctly, you should see the MuJoCo viewer open with a retargeting simulation.

## Optional: Install Additional Backends

### DexMachina (Genesis Simulator)

For RL training with Genesis simulator:

```bash
# Follow DexMachina installation instructions
# https://mandizhao.github.io/dexmachina-docs/0_install.html

conda activate dexmachina
pip install --ignore-requires-python --no-deps -e .
```

### HDMI (Humanoid RL)

For humanoid robot RL training:

```bash
# Clone and install HDMI
git clone https://github.com/lecar-lab/hdmi.git
cd hdmi

# Install SPIDER in HDMI environment
uv pip install --no-deps -e ../spider
```

### IDE Setup

For VSCode, recommended extensions:
- Python
- Pylance
- Ruff

You can load the `.vscode/settings.json` file to your VSCode to get the recommended settings.
