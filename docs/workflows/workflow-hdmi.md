# Workflow: HDMI (Humanoid-Object Interaction)

The HDMI workflow integrates SPIDER with [HDMI (Humanoid Manipulation)](https://github.com/lecar-lab/hdmi) framework for humanoid robot manipulation with downstream sim2real RL training.

## Prerequisites

### Install HDMI Environment

Follow the [official HDMI installation guide](https://github.com/lecar-lab/hdmi):

```bash
# Clone HDMI repository
git clone https://github.com/lecar-lab/hdmi.git
cd hdmi

# Install with uv (recommended)
uv sync

# Install SPIDER into HDMI environment
uv pip install --no-deps -e /path/to/spider
```

## Running HDMI Workflow

### Basic Example

```bash
cd /path/to/spider

# Run with default config (G1 humanoid, suitcase task)
python examples/run_hdmi.py
# OR refer to
./examples/run_hdmi.sh
```

### Configuration Options

You can enable `mjlab` native viewer by setting `viewer` to `hdmi`.

## Train RL

Write the motion as one HDMI reference motion by running the following command:

```bash
python examples/postprocess/read_to_hdmi.py
```
