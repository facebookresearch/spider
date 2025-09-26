# Preprocessing Pipeline Implementation

This document describes the completed preprocessing pipeline in `retarget/preprocess/pipeline.py`.

## Overview

The pipeline automates the complete preprocessing workflow for different datasets (gigahand, oakink, fair_mon, fair_fre) through the following steps:

1. **Dataset Processing**: Convert raw data (.pkl) to standardized kinematic trajectories (.npz)
2. **Object Decomposition**: Generate convex decomposition for object meshes
3. **XML Generation**: Create MuJoCo scene files with robot and object models
4. **IK Retargeting**: Generate robot-specific joint trajectories from human demonstrations

## Key Functions

### `get_all_tasks_with_data_id(config: Config) -> Dict[str, List[int]]`

Discovers all available tasks and their data IDs from the raw dataset directory.

- **gigahand**: Scans `object_poses/` directory for participant-scene-sequence patterns
- **oakink**: Scans for `{task}_{hand_type}.pkl` files
- **fair_mon/fair_fre**: Scans task directories for numbered `.pkl` files

### `get_task_config(dataset_dir, dataset_name, robot_type, hand_type, task, data_id) -> Config`

Creates a `Config` object for a specific task with the provided parameters.

### `process_single(config: Config)`

Processes a single task through the complete pipeline with intelligent skipping:

1. **Step 1 - Dataset Processing**: 
   - Checks if `trajectory_kinematic.npz` exists in MANO output directory
   - Calls appropriate dataset processor (gigahand.py, oakink.py, fair_mon.py)
   - Skips if already processed

2. **Step 2 - Object Decomposition**:
   - Checks if convex decomposition directories exist in task_info.json
   - Calls `decompose.py` to generate convex hulls
   - Skips if already decomposed

3. **Step 3 - XML Generation**:
   - Checks if `scene.xml` exists in robot output directory
   - Calls `generate_xml.py` to create MuJoCo scene
   - Skips if already generated

4. **Step 4 - IK Retargeting**:
   - Checks if robot `trajectory_kinematic.npz` exists
   - Calls `ik.py` to generate robot joint trajectories
   - Skips if already processed

### `main(cfg: PipelineConfig)`

Main entry point that:
- Discovers all tasks for the specified dataset/robot/hand configuration
- Optionally filters to specific tasks if provided
- Processes each task through the pipeline
- Provides detailed logging and error handling
- Reports success/failure statistics

## Configuration

The pipeline uses Hydra configuration from `examples/config/preprocess.yaml`:

```yaml
dataset_dir: "../../example_datasets"
dataset_name: "oakink"  # oakink, gigahand, fair_mon, fair_fre
hand_type: "bimanual"   # right, left, bimanual
robot_type: "allegro"   # allegro, metahand, inspire, etc.
all_tasks: null         # Optional list to filter specific tasks
```

## Usage

### Command Line (Hydra)
```bash
cd retarget
python retarget/preprocess/pipeline.py dataset_name=oakink robot_type=allegro hand_type=bimanual
```

### Programmatic
```python
from retarget.preprocess.pipeline import main, PipelineConfig

cfg = PipelineConfig(
    dataset_dir="./example_datasets",
    dataset_name="oakink",
    robot_type="allegro", 
    hand_type="bimanual"
)
main(cfg)
```

## Features

- **Intelligent Skipping**: Automatically detects existing processed files and skips unnecessary steps
- **Multi-Dataset Support**: Handles different dataset formats and structures
- **Error Resilience**: Continues processing other tasks if one fails
- **Comprehensive Logging**: Detailed progress reporting with loguru
- **Flexible Filtering**: Process all tasks or filter to specific ones

## Output Structure

The pipeline creates the following directory structure:

```
{dataset_dir}/processed/{dataset_name}/
├── mano/{hand_type}/{task}/{data_id}/
│   └── trajectory_kinematic.npz     # Human kinematic data
├── {robot_type}/{hand_type}/{task}/
│   ├── scene.xml                    # MuJoCo scene file
│   ├── scene_eq.xml                 # Scene with equality constraints
│   ├── task_info.json               # Task metadata
│   └── {data_id}/
│       └── trajectory_kinematic.npz # Robot kinematic data
└── assets/
    ├── objects/{object_name}/
    │   ├── visual.obj               # Object mesh
    │   └── convex/                  # Convex decomposition
    └── robots/{robot_type}/         # Robot assets
```

## Testing

A test script is provided at `test_pipeline.py` to validate:
- Task discovery across different datasets
- Config generation
- File existence checking
- Pipeline component integration

## Error Handling

The pipeline includes robust error handling:
- Individual task failures don't stop the entire pipeline
- Detailed error logging with stack traces
- Summary reporting of successful vs failed tasks
- Graceful handling of missing files or directories
