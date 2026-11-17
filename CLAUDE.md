# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SPIDER (Scalable Physics-Informed DExterous Retargeting) is a physics-based retargeting framework that converts human motion (from video or mocap) into robot actions for dexterous hands and humanoid robots. Developed by Meta/FAIR. Python 3.12+, licensed CC BY-NC 4.0.

## Common Commands

### Environment Setup
```bash
uv sync                    # Install dependencies (preferred)
pip install --no-deps -e . # Editable install without pulling deps
```

### Linting & Formatting
```bash
ruff check .          # Lint
ruff check --fix .    # Auto-fix
ruff format .         # Format
```

### Running Tests
```bash
uv run spider/simulators/mjwp_test.py        # MuJoCo Warp simulator
uv run spider/simulators/dexmachina_test.py  # DexMachina simulator
uv run spider/simulators/hdmi_test.py        # HDMI simulator
```

### Running Retargeting (MJWP workflow)
```bash
# Single task
uv run examples/run_mjwp.py +override=gigahand task=p36-tea data_id=0 robot_type=xhand embodiment_type=bimanual

# With remote viewer
uv run examples/run_mjwp.py viewer="rerun"
```

### Running Retargeting on the CPU (default workflow, `simulator=cpu`)
`spider/simulators/cpu.py` has MJWP's simulator interface on MuJoCo's CPU batch API; `examples/run_cpu.py` follows
`run_mjwp.py`'s protocol. Writes `trajectory_cpu.npz`, `metrics.json` and `visualization_cpu.mp4`.
```bash
# Needs MuJoCo >= 3.17, which SPIDER's own environment does not have: use the small environment of
# docs/workflows/workflow-cpu.md. Scene: example_datasets/processed/chord/sharpa/bimanual/<task>/<data_id>/
# {scene.xml,scene.npz} (ARCTIC scenes in the example dataset: task = <subject>-<object>-<use|grab>,
# data_id = take; default s01-box-use/1)
python examples/run_cpu.py                                       # Viser at http://localhost:8080: pick a task, press Start
python examples/run_cpu.py task=s01-laptop-use data_id=2 autostart=true   # start that clip at once, with the viewer
python examples/run_cpu.py show_viewer=false num_threads=16      # headless; about 9 s of planning per second of motion

# Score a saved run again
python -m spider.postprocess.evaluate_cpu OUT

# Tests
pytest spider/simulators/cpu_test.py spider/postprocess/evaluate_cpu_test.py spider/viewers/viser_viewer_test.py \
    spider/simulators/chord_export_test.py
```
The recipe (128 samples, 4 iterations per control frame, going back when the object slips) is in
`examples/config/cpu.yaml`; the reasons are in `docs/workflows/workflow-chord.md`.

### Running Retargeting in FlashCHORD's GPU environment
`examples/run_chord.py`: the GPU twin of `run_cpu.py` (same loop, recovery and viewer) on `spider/simulators/chord.py`;
`examples/config/chord.yaml` starts from `cpu.yaml`. Writes `trajectory_chord.npz`, `metrics.json`, `visualization_chord.mp4`.
```bash
# Needs flash_chord (SPIDER installed into its environment with --no-deps). With the viewer a run waits for
# Start in the browser: add autostart=true, or show_viewer=false for batch runs.
python examples/run_chord.py "data_path='REFERENCE_DIR'" output_dir=OUT
# export a scene for the CPU workflow
python -m spider.simulators.chord_export --parquet REFERENCE_DIR --out example_datasets/processed/chord/sharpa/bimanual/TASK/ID
CHORD_TEST_PARQUET=REFERENCE_DIR pytest spider/simulators/chord_test.py
```
The gains and solver sizes of the recipe are constants at the top of `spider/simulators/chord.py`.

### Full Pipeline (preprocess → retarget → postprocess)
```bash
uv run spider/process_datasets/gigahand.py --task=TASK --embodiment-type=TYPE --data-id=ID
uv run spider/preprocess/decompose_fast.py --task=TASK --dataset-name=NAME --data-id=ID --embodiment-type=TYPE
uv run spider/preprocess/detect_contact.py --task=TASK --dataset-name=NAME --data-id=ID --embodiment-type=TYPE
uv run spider/preprocess/generate_xml.py --task=TASK --dataset-name=NAME --data-id=ID --embodiment-type=TYPE --robot-type=ROBOT
uv run spider/preprocess/ik.py --task=TASK --dataset-name=NAME --data-id=ID --embodiment-type=TYPE --robot-type=ROBOT --open-hand
uv run examples/run_mjwp.py +override=NAME task=TASK data_id=ID robot_type=ROBOT embodiment_type=TYPE
```

## Code Style

- **Ruff** for linting and formatting: 88-char line length, Google-style docstrings, Python 3.12+ target
- See `[tool.ruff]` in `pyproject.toml` for full lint rule configuration

## Architecture

### Configuration System
`spider/config.py` defines a single `@dataclass Config` (~150 fields) that drives the entire pipeline. Configs are loaded via Hydra/OmegaConf from YAML files in `examples/config/`. Dataset-specific overrides live in `examples/config/override/`. The `+override=DATASET` Hydra syntax selects a dataset config overlay.

### Data Flow
1. **Raw data** (`.pkl`) → `spider/process_datasets/*.py` converts dataset-specific formats to standard NPZ
2. **Preprocessing** (`spider/preprocess/`) → mesh decomposition, contact detection, scene XML generation, inverse kinematics
3. **Physics optimization** (`examples/run_mjwp.py`) → sampling-based MPC retargeting using simulator
4. **Postprocessing** (`spider/postprocess/`) → success metrics, robot deployment prep

### Simulators (`spider/simulators/`)
- **MJWP** (`mjwp.py`, ~1400 lines) — primary backend, GPU-accelerated MuJoCo + Warp with batched environments
- **MJWP-EQ** (`mjwp_eq.py`) — variant with equality constraints
- **DexMachina** (`dexmachina.py`) — Genesis simulator for RL training
- **HDMI** (`hdmi.py`) — humanoid RL workflow
- **Isaac** (`isaac.py`) — NVIDIA Isaac integration
- **CPU** (`cpu.py`) — the default (`simulator=cpu`): MJWP's interface on MuJoCo's CPU batch API in double precision; self-contained (MuJoCo >= 3.17); runner `examples/run_cpu.py`
- **CHORD** (`chord.py`, `chord_export.py`) — sampling MPC directly on FlashCHORD's GPU environment (its physics and control law; needs `flash_chord`), and the export of its scene for the CPU workflow; runner `examples/run_chord.py`

### Optimizer (`spider/optimizers/sampling.py`)
Sampling-based MPC using cross-entropy method. Key functions: `make_rollout_fn`, `make_optimize_fn`, `make_optimize_once_fn`. Supports `torch.compile` for speedup, noise scheduling with temperature annealing, and contact guidance.

### Viewers (`spider/viewers/`)
Multiple visualization backends (MuJoCo, Rerun, Viser) with a common interface. Selected via `viewer=` config parameter. Supports simultaneous multi-viewer (e.g., `viewer="mujoco-rerun"`).

### Dataset Processors (`spider/process_datasets/`)
Each supported dataset (GigaHands, OakInk, Hot3D, GMR, FAIR-MON, FAIR-FRE) has a dedicated processor that converts raw data to standard format. New datasets require a new processor here outputting NPZ files.

### Adding New Robots
Requires: MJCF assets in `spider/assets/`, embodiment mappings in `spider/config.py`, and reward weight tuning.

### Key Utility Modules
- `spider/io.py` — data loading/saving, path resolution
- `spider/math.py` — quaternion/rotation math utilities
- `spider/interp.py` — trajectory interpolation
- `spider/mujoco_utils.py` — MuJoCo model helpers
