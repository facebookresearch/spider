# Quick Start

This guide will help you run your first SPIDER retargeting example in minutes.

## Prerequisites

Make sure you have completed the [installation](/guide/installation) and cloned the example datasets.

## Run Preprocessed Example

The quickest way to see SPIDER in action is to run a preprocessed example:

```bash
# Run with default config (Xhand bimanual, OakInk dataset)
uv run examples/run_mjwp.py
```

This will:
1. Load preprocessed kinematic trajectory
2. Optimize trajectory with physics constraints
3. Display real-time visualization in MuJoCo viewer
4. Save optimized trajectory and video

## Full Workflow Example

To process a task from scratch, follow these steps:

### 1. Set Task Parameters

```bash
export TASK=p36-tea
export HAND_TYPE=bimanual
export DATA_ID=10
export ROBOT_TYPE=xhand
export DATASET_NAME=gigahand
```

### 2. Process Dataset

Convert raw human motion data to standardized format:

```bash
uv run spider/process_datasets/oakink.py \
  --task=${TASK} \
  --embodiment-type=${HAND_TYPE} \
  --data-id=${DATA_ID}
```

### 3. Decompose Object Mesh

Create convex decomposition for physics simulation:

```bash
uv run spider/preprocess/decompose_fast.py \
  --task=${TASK} \
  --dataset-name=${DATASET_NAME} \
  --data-id=${DATA_ID} \
  --embodiment-type=${HAND_TYPE}
```

### 4. Detect Contacts (Optional)

Identify hand-object contact points:

```bash
uv run spider/preprocess/detect_contact.py \
  --task=${TASK} \
  --dataset-name=${DATASET_NAME} \
  --data-id=${DATA_ID} \
  --embodiment-type=${HAND_TYPE}
```

### 5. Generate Scene

Create MuJoCo scene XML:

```bash
uv run spider/preprocess/generate_xml.py \
  --task=${TASK} \
  --dataset-name=${DATASET_NAME} \
  --data-id=${DATA_ID} \
  --embodiment-type=${HAND_TYPE} \
  --robot-type=${ROBOT_TYPE}
```

### 6. Run Inverse Kinematics

Convert human poses to robot joint angles:

```bash
uv run spider/preprocess/ik.py \
  --task=${TASK} \
  --dataset-name=${DATASET_NAME} \
  --data-id=${DATA_ID} \
  --embodiment-type=${HAND_TYPE} \
  --robot-type=${ROBOT_TYPE} \
  --open-hand
```

### 7. Physics-Based Retargeting

Optimize trajectory with physics constraints:

```bash
uv run examples/run_mjwp.py \
  +override=${DATASET_NAME} \
  task=${TASK} \
  data_id=${DATA_ID} \
  robot_type=${ROBOT_TYPE} \
  embodiment_type=${HAND_TYPE}
```

## Understanding the Output

After running, you'll find these files in the output directory:

```
example_datasets/processed/oakink/allegro/bimanual/pick_spoon_bowl/0/
├── trajectory_kinematic.npz     # IK result
├── trajectory_mjwp.npz          # Optimized trajectory
├── visualization_mjwp.mp4       # Video output
└── metrics.json                 # Success metrics
```

### Trajectory Data Format

The NPZ files contain:

```python
import numpy as np

data = np.load('trajectory_mjwp.npz')

# Available keys:
# - qpos: [T, nq] joint positions
# - qvel: [T, nv] joint velocities
# - ctrl: [T, nu] control commands
# - time: [T] timestamps
```

## Customize Parameters

Override configuration parameters from command line:

```bash
# Increase optimization samples
uv run examples/run_mjwp.py num_samples=2048

# Change planning horizon
uv run examples/run_mjwp.py horizon=2.0

# Use different viewer
uv run examples/run_mjwp.py viewer=rerun

# Combine multiple overrides
uv run examples/run_mjwp.py \
  robot_type=inspire \
  num_samples=2048 \
  horizon=2.0 \
  viewer=mujoco-rerun
```

## Remote Development

For headless servers, use Rerun viewer:

```bash
# On remote server, start Rerun server
uv run rerun --serve-web --port 9876

# Run SPIDER with Rerun viewer
uv run examples/run_mjwp.py viewer=rerun

# On local machine, open browser to:
# http://your-server:9876
```

## Common Tasks

### Run Different Robot

```bash
# Allegro hand
uv run examples/run_mjwp.py robot_type=allegro

# Inspire hand
uv run examples/run_mjwp.py robot_type=inspire

# Xhand
uv run examples/run_mjwp.py robot_type=xhand
```

### Run Different Dataset

```bash
# GigaHand
uv run examples/run_mjwp.py +override=gigahand

# Hot3D
uv run examples/run_mjwp.py +override=hot3d

# Humanoid task
uv run examples/run_mjwp.py +override=humanoid
```

### Single Hand Configuration

```bash
# Right hand only
uv run examples/run_mjwp.py embodiment_type=right

# Left hand only
uv run examples/run_mjwp.py embodiment_type=left
```
