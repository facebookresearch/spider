# Workflow: DexMachina (Genesis)

The DexMachina workflow integrates SPIDER with [Genesis](https://genesis.github.io/) simulator and [DexMachina](https://github.com/MandiZhao/dexmachina) framework for dexterous manipulation with downstream RL training.

## Prerequisites

### Install DexMachina Environment

Follow the [official DexMachina installation guide](https://mandizhao.github.io/dexmachina-docs/0_install.html):

```bash
# Create DexMachina conda environment
conda create -n dexmachina python=3.12
conda activate dexmachina

# Install Genesis
git clone https://github.com/Genesis-Embodied-AI/Genesis.git
cd Genesis
pip install -e .

# Install DexMachina
git clone https://github.com/MandiZhao/dexmachina.git
cd dexmachina
pip install -e .
```

### Install SPIDER (Minimal)

Install SPIDER without MuJoCo Warp (only need optimization components):

```bash
cd /path/to/spider
conda activate dexmachina

# Install SPIDER without dependencies (DexMachina has its own)
pip install --ignore-requires-python --no-deps -e .
```

## Running DexMachina Workflow

### Basic Example

```bash
conda activate dexmachina
cd /path/to/spider

# Run with default config (Allegro hand, cube repose task)
python examples/run_dexmachina.py
```

This will:
1. Initialize Genesis environment with DexMachina task
2. Load reference motion from task
3. Optimize trajectory with Sampling
4. Save optimized trajectory and video
