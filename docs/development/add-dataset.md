# Add New Dataset

This guide explains how to add a new dataset to SPIDER for retargeting human motion data to robots.

## Overview

Adding a new dataset involves:
1. Preparing raw data in standardized format
2. Creating a dataset processor script
3. Extracting hand kinematics and object poses
4. Converting meshes to MuJoCo-compatible format
5. Testing the integration

## Dataset Requirements

Your dataset should include:
- **Hand motion data**: Either MANO parameters or joint angles
- **Object information**: 3D meshes and 6D poses (position + orientation)
- **Temporal alignment**: Synchronized hand and object trajectories

Supported formats:
- MANO parameters (shape, pose)
- Direct joint angles
- Wrist poses (position + rotation)
- Object 6D poses

## Data File Structure

SPIDER uses a standardized directory structure:

```
example_datasets/
├── raw/                          # Raw data from original dataset
│   └── my_dataset/
│       ├── task_name_01.pkl      # Raw motion capture data
│       ├── task_name_02.pkl
│       └── meshes/               # Object meshes
│           ├── cup.obj
│           └── spoon.obj
│
└── processed/                    # Processed data for SPIDER
    └── my_dataset/
        ├── dataset_summary.json  # Dataset metadata
        ├── assets/               # Shared assets
        │   ├── objects/          # Object meshes
        │   │   └── cup/
        │   │       ├── convex/   # Convex decomposition
        │   │       │   ├── 0.obj
        │   │       │   ├── 1.obj
        │   │       │   └── ...
        │   │       └── visual.obj
        │   └── robots/           # Robot models
        │       └── allegro/
        │           ├── left.xml
        │           └── right.xml
        └── mano/                 # Processed MANO data
            └── bimanual/
                └── task_name/
                    └── 0/
                        ├── trajectory_keypoint.npz
                        └── info.json
```

## Step 1: Prepare Raw Data

Place your raw data in the appropriate directory:

```bash
mkdir -p example_datasets/raw/my_dataset
# Copy your raw data files here
```

### Raw Data Format

Your raw data file (`.pkl` or `.npz`) should contain:

```python
{
    # Hand data (one of the following):
    'mano_pose': [...],           # [T, 48] MANO pose parameters
    'mano_shape': [...],          # [10] or [T, 10] MANO shape parameters
    # OR
    'qpos_finger_left': [...],    # [T, n_joints] Left finger joints
    'qpos_finger_right': [...],   # [T, n_joints] Right finger joints
    'qpos_wrist_left': [...],     # [T, 7] Left wrist pose (xyz + quat)
    'qpos_wrist_right': [...],    # [T, 7] Right wrist pose (xyz + quat)

    # Object data:
    'object_pose_left': [...],    # [T, 7] Object pose (xyz + quat) for left hand
    'object_pose_right': [...],   # [T, 7] Object pose (xyz + quat) for right hand
    'object_name_left': 'cup',    # Object identifier
    'object_name_right': 'spoon',

    # Metadata:
    'fps': 30.0,                  # Frame rate
    'task_name': 'pick_cup',      # Task identifier
}
```

## Step 2: Create Dataset Processor

Create a processor script at `spider/process_datasets/my_dataset.py`:

## Step 3: Handle Object Meshes

Convert Meshes to OBJ Format

Ensure meshes are in MuJoCo-compatible OBJ format:

```python
def convert_mesh_to_obj(input_mesh: str, output_mesh: str):
    """
    Convert mesh to MuJoCo-compatible OBJ format.

    Supports various input formats (STL, PLY, glb, etc.)
    """
    import trimesh

    # Load mesh
    mesh = trimesh.load(input_mesh)

    # Ensure single mesh (merge if needed)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    # Export as OBJ
    mesh.export(output_mesh)

    print(f"Converted {input_mesh} to {output_mesh}")
```

Place Meshes in Assets

Place meshes in assets folder, for example, for cup:

```bash
mkdir -p example_datasets/processed/my_dataset/assets/objects/cup
cp example_datasets/raw/my_dataset/meshes/cup.obj example_datasets/processed/my_dataset/assets/objects/cup/visual.obj
```

## Step 4: Test Your Dataset Processor

Run Processor

```bash
# Process a single sample
uv run spider/process_datasets/my_dataset.py \
  --task=pick_cup \
  --embodiment-type=bimanual \
  --data-id=0
```

### Verify Output

Check that the output files exist and have correct format.

```python
import numpy as np

# Load processed data
data = np.load('example_datasets/processed/my_dataset/mano/bimanual/pick_cup/0/trajectory_keypoint.npz')

print("Keys:", list(data.keys()))
print("qpos_finger shape:", data['qpos_finger'].shape)
print("qpos_wrist shape:", data['qpos_wrist'].shape)
```

## Resources

- [OakInk Dataset](https://oakink.net/)
- [GigaHand Dataset](https://gigahands.github.io/)
- [Hot3D Dataset](https://github.com/facebookresearch/hot3d)
- [MANO Hand Model](https://mano.is.tue.mpg.de/)
