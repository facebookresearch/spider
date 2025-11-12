# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SPIDER (Scalable Physics-Informed DExterous Retargeting) is a physics-based retargeting framework that converts human motion to diverse robot embodiments (dexterous hands and humanoid robots). The codebase implements a three-phase pipeline: data preprocessing, physics-based trajectory optimization, and multi-backend simulation.

## Development Setup

### Environment Setup

**Option 1: Using uv (recommended)**
```bash
# Install Python 3.12 interpreter
uv python install 3.12

# Sync environment
uv sync --python 3.12
pip install --ignore-requires-python --no-deps -e .
```

**Option 2: Using conda**
```bash
conda create -n spider python=3.12
conda activate spider
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

### Running Tests

To test individual simulator backends:
```bash
# Test MJWP (Mujoco Warp) simulator
uv run spider/simulators/mjwp_test.py

# Test DexMachina simulator
uv run spider/simulators/dexmachina_test.py

# Test HDMI simulator
uv run spider/simulators/hdmi_test.py
```

### Linting and Formatting

The project uses ruff for linting and formatting:
```bash
# Run linter
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code
ruff format .
```

Configuration in [pyproject.toml](pyproject.toml):
- Line length: 88 characters
- Target: Python 3.12+
- Style: Google-style docstrings
- Enabled checks: pycodestyle, Pyflakes, isort, pep8-naming, pydocstyle, pyupgrade, flake8-bugbear

## Architecture Overview

### Core Pipeline Phases

1. **Dataset Processing** ([spider/process_datasets/](spider/process_datasets/))
   - Converts raw human motion data from various datasets (GigaHand, Hot3D, OakInk, etc.)
   - Outputs standardized NPZ with wrist/finger/object poses

2. **Preprocessing** ([spider/preprocess/](spider/preprocess/))
   - 5-stage pipeline to prepare robot-compatible trajectories:
     - `decompose_fast.py`: Convex decomposition of object meshes
     - `detect_contact.py`: Detects hand-object contact points
     - `generate_xml.py`: Builds MuJoCo scene XML
     - `ik.py`: Inverse kinematics to convert human poses to robot joint angles
     - Output: `trajectory_kinematic.npz` (qpos, qvel, ctrl, contact info)

3. **Physics Optimization** ([examples/run_mjwp.py](examples/run_mjwp.py))
   - Sampling-based MPC using DIAL algorithm
   - Refines kinematic trajectory with physics constraints

### Module Organization

- **[spider/config.py](spider/config.py)**: Central configuration system. All parameters flow through the `Config` dataclass. Key sections:
  - Task/dataset configuration (robot_type, embodiment_type, task, data_id)
  - Simulator timing (sim_dt, ctrl_dt, horizon, knot_dt)
  - Optimizer parameters (num_samples, temperature, noise schedules)
  - Reward scaling (position/rotation/joint/velocity weights)

- **[spider/simulators/](spider/simulators/)**: Physics backends implementing common interface
  - `mjwp.py`: Primary backend using MuJoCo Warp (batched GPU simulation)
  - `dexmachina.py`: Genesis simulator for RL training on dexterous hands
  - `hdmi.py`: MjLab-based humanoid robot simulation
  - Common API: `setup_env()`, `step_env()`, `get_reward()`, `get_terminate()`, `save_state()`, `load_state()`

- **[spider/optimizers/sampling.py](spider/optimizers/sampling.py)**: Sampling-based MPC implementation
  - `sample_ctrls()`: Generates noisy action samples
  - `make_rollout_fn()`: Batched trajectory evaluation with early termination
  - `make_optimize_once_fn()`: Single iteration of weighted sampling
  - `make_optimize_fn()`: Full optimization loop with annealing

- **[spider/viewers/](spider/viewers/)**: Visualization systems
  - `mjwp_viewer.py`: Real-time MuJoCo rendering
  - `rerun_viewer.py`: Remote logging with timeline scrubbing

- **[spider/io.py](spider/io.py)**: Data loading/saving with interpolation utilities

- **[spider/math.py](spider/math.py)**: Quaternion math for 3D rotations

## Common Workflows

### Full MJWP Workflow (Dexterous Hand)

```bash
# Set task parameters
TASK=pick_spoon_bowl
HAND_TYPE=bimanual
DATA_ID=0
ROBOT_TYPE=allegro
DATASET_NAME=oakink

# Process raw dataset
uv run spider/process_datasets/oakink.py --task=${TASK} --embodiment-type=${HAND_TYPE} --data-id=${DATA_ID}

# Decompose object mesh
uv run spider/preprocess/decompose_fast.py --task=${TASK} --dataset-name=${DATASET_NAME} --data-id=${DATA_ID} --embodiment-type=${HAND_TYPE}

# Detect contacts (optional)
uv run spider/preprocess/detect_contact.py --task=${TASK} --dataset-name=${DATASET_NAME} --data-id=${DATA_ID} --embodiment-type=${HAND_TYPE}

# Generate scene XML
uv run spider/preprocess/generate_xml.py --task=${TASK} --dataset-name=${DATASET_NAME} --data-id=${DATA_ID} --embodiment-type=${HAND_TYPE} --robot-type=${ROBOT_TYPE}

# Run IK
uv run spider/preprocess/ik.py --task=${TASK} --dataset-name=${DATASET_NAME} --data-id=${DATA_ID} --embodiment-type=${HAND_TYPE} --robot-type=${ROBOT_TYPE} --open-hand

# Physics-based retargeting
uv run examples/run_mjwp.py +override=${DATASET_NAME} task=${TASK} data_id=${DATA_ID} robot_type=${ROBOT_TYPE} embodiment_type=${HAND_TYPE}
```

### Quick Test with Preprocessed Data

```bash
# If example_datasets is already cloned
uv run examples/run_mjwp.py
```

### DexMachina Workflow (Genesis + RL)

```bash
# Install DexMachina environment separately
conda activate dexmachina
pip install --ignore-requires-python --no-deps -e .

# Run retargeting
python examples/run_dexmachina.py
```

### HDMI Workflow (Humanoid + RL)

```bash
# Install in HDMI uv environment
cd ../hdmi
uv pip install --no-deps -e ../spider
```

### Remote Development

```bash
# Start rerun server on remote machine
uv run rerun --serve-web --port 9876

# Run with rerun-only viewer
uv run examples/run_mjwp.py viewer="rerun"
```

## Configuration System

SPIDER uses Hydra for configuration management. Config files are in [examples/config/](examples/config/):

- `default.yaml`: Base configuration
- `mjwp.yaml`, `dexmachina.yaml`, `hdmi.yaml`: Backend-specific overrides
- `override/`: Task-specific configurations (humanoid, fair_mon, etc.)

### Overriding Parameters

```bash
# Override via command line
uv run examples/run_mjwp.py robot_type=inspire task=pick_cup num_samples=2048

# Use preset override
uv run examples/run_mjwp.py +override=humanoid

# Combine multiple overrides
uv run examples/run_mjwp.py +override=gigahand robot_type=xhand viewer=rerun
```

### Key Configuration Parameters

**Timing** (all values must be divisible by `sim_dt`):
- `sim_dt`: Simulation timestep (default: 0.01s)
- `ctrl_dt`: Control frequency (default: 0.4s)
- `horizon`: Planning horizon (default: 1.6s)
- `knot_dt`: Knot point spacing for trajectory parameterization (default: 0.4s)

**Optimizer**:
- `num_samples`: Parallel trajectory samples (default: 1024)
- `temperature`: Softmax temperature for weighting (default: 0.1)
- `max_num_iterations`: Optimization iterations per control step (default: 32)
- Noise scales: `joint_noise_scale` (0.3), `pos_noise_scale` (0.01), `rot_noise_scale` (0.03)

**Reward Weights**:
- `pos_rew_scale`: End-effector position tracking (default: 1.0)
- `rot_rew_scale`: Rotation tracking (default: 0.3)
- `joint_rew_scale`: Joint angle tracking (default: 0.003)
- `vel_rew_scale`: Velocity regularization (default: 0.0001)

## Embodiment Types

The codebase supports multiple embodiment configurations via `embodiment_type`:

- **`bimanual`**: Two hands + two objects (e.g., allegro, inspire, xhand)
  - qpos structure: [palm_R(7) + fingers_R + palm_L(7) + fingers_L + obj_R(7) + obj_L(7)]

- **`left`** / **`right`**: Single hand + one object
  - qpos structure: [palm(7) + fingers + obj(7)]

- **`CMU`**: Humanoid body (G1, H1, T1 robots)
  - qpos structure: [base_pos(3) + base_quat(4) + joint_angles + obj(7)]

## Data Directory Structure

Processed data follows this convention:
```
{dataset_dir}/processed/{dataset_name}/{robot_type}/{embodiment_type}/{task}/{data_id}/
├── trajectory_keypoints.npz       # From dataset processor
├── trajectory_kinematic.npz       # From IK stage
├── trajectory_mjwp.npz            # Optimized trajectory
├── visualization_mjwp.mp4         # Output video
└── metrics.json                   # Success rate, tracking error
```

Scene XML is shared across robots:
```
{dataset_dir}/processed/{dataset_name}/{embodiment_type}/{task}/{data_id}/
├── scene.xml                      # MuJoCo scene
├── scene_eq.xml                   # With equality constraints (for num_dyn > 1)
└── task_info.json                 # Metadata (ref_dt, etc.)
```

## Key Design Patterns

### Batched Simulation

All simulators use batched execution across `num_samples` parallel worlds:
```python
# Shapes during optimization
ctrls_samples: [num_samples, horizon_steps, nu]
env.data_wp.qpos: [num_samples, nq]  # Batched state
rewards: [num_samples]  # Per-sample cumulative reward
```

### Embodiment-Specific Logic

Reward computation and noise scheduling adapt based on `config.embodiment_type`:
```python
# In config.py get_noise_scale()
if embodiment_type == "bimanual":
    noise_scale[:, :, :3] *= pos_noise_scale  # Right palm position
    noise_scale[:, :, 3:6] *= rot_noise_scale  # Right palm rotation
    half_dof = nu // 2
    noise_scale[:, :, half_dof:half_dof+3] *= pos_noise_scale  # Left palm
```

### Annealing and Exploitation

Noise schedule balances exploration/exploitation:
- First sample: Zero noise (deterministic baseline)
- Most samples: Logspace from `first_ctrl_noise_scale` to `last_ctrl_noise_scale`
- Last 1% samples: `exploit_noise_scale` for fine-tuning
- Noise multiplied by `beta_traj` each iteration

### Domain Randomization

When `num_dr > 1`, optimizer evaluates across randomized contact margins and object offsets, using worst-case reward for robustness.

## Adding New Robots

1. Add robot MJCF to assets (see [spider/simulators/mjwp.py:setup_env()](spider/simulators/mjwp.py) for loading logic)
2. Update embodiment type mapping in [spider/config.py:process_config()](spider/config.py)
3. Adjust reward weights in config for robot-specific tracking priorities
4. Test with IK: `uv run spider/preprocess/ik.py --robot-type=new_robot`

## Adding New Datasets

1. Create processor in [spider/process_datasets/](spider/process_datasets/) following existing examples
2. Output format: NPZ with `qpos_wrist_*`, `qpos_finger_*`, `qpos_obj_*`, `contact`, `contact_pos`
3. Add Hydra override config in `examples/config/override/{dataset_name}.yaml`
4. Update ref_dt in `task_info.json` if dataset framerate differs from default

## Performance Notes

- MJWP uses CUDA graph capture for fast batched simulation (~10-20x faster than sequential MuJoCo)
- `torch.compile` can be enabled via `use_torch_compile=True` (experimental)
- Viewer overhead: MuJoCo viewer adds ~10ms per frame; Rerun is asynchronous
- Bottleneck is typically in optimization sampling, not physics simulation

## Debugging

### Visualization

- Use `viewer="mujoco-rerun"` for dual visualization (real-time + logged)
- Traces show top-k sample trajectories (configurable via `num_trace_topk_samples`)
- Reference trajectory shown as transparent ghost in MuJoCo viewer

### Common Issues

- **"horizon must be divisible by sim_dt"**: Ensure all timing parameters are multiples of `sim_dt`
- **Contact instabilities**: Increase `nconmax_per_env` or reduce `pair_margin_range`
- **Poor tracking**: Adjust reward scales (`pos_rew_scale`, `rot_rew_scale`, `joint_rew_scale`)
- **Optimization not converging**: Increase `num_samples` or `max_num_iterations`
- **Out of memory**: Reduce `num_samples` or `horizon_steps`

### Logging

- Set `save_info=True` to save trajectory NPZ with qpos/qvel/ctrl
- Set `save_metrics=True` for success rate and tracking error statistics
- Use `ipdb` for interactive debugging (already in dependencies)
