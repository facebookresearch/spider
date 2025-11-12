<h1 align="center">🕸️ SPIDER: Scalable Physics-Informed DExterous Retargeting</h1>

<p align="center">
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-3.12+-blue.svg" alt="Python 3.12+">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch">
  </a>
  <a href="https://arxiv.org/abs/xxxx">
    <img src="https://img.shields.io/badge/arXiv-2406.12345-b31b1b.svg" alt="arXiv">
  </a>
  <a href="https://jc-bao.github.io/spider/">
    <img src="https://img.shields.io/badge/docs-latest-blue.svg" alt="Documentation">
  </a>
  <a href="https://jc-bao.github.io/spider-project/">
    <img src="https://img.shields.io/badge/website-project-blue.svg" alt="Project Website">
  </a>
</p>

![logo](figs/teaser.png)

## Overview

Scalable Physics-Informed DExterous Retargeting (SPIDER) is a general framework for physics-based retargeting from human to diverse robot embodiments, including both dexterous hand and humanoid robot.
It is designed to be a minimum, flexible and extendable framework for human2robot retargeting.
This code base provides the following pipeline from human video to robot actions:

![pipeline](figs/pipeline_animation.gif)


## Gallery

### Simulation results:

| Inspire Pick Tea Pot (Gigahands Dataset) | Xhand Play Glass (Hot3D dataset) | Schunk Pick Board (Oakink dataset) | Allegro Pick Cat Toy (Reconstructed from single RGB video)
| ------- | ------- | ------- | ------- |
| ![](figs/sim/inspire_pick_pot.gif) | ![](figs/sim/xhand_glass.gif) | ![](figs/sim/schunk_move_board.gif) | ![](figs/sim/allegro_pick_cat.gif) |


| G1 Pick | G1 Run | H1 Kick | T1 skip |
| ------- | ------- | ------- | ------- |
| ![](figs/sim/g1_pick.gif) | ![](figs/sim/g1_run.gif) | ![](figs/sim/h1_kick.gif) | ![](figs/sim/t1_skip.gif) |


### Multiple viewer support:
| Mujoco | Rerun |
| ------- | ------- |
| ![](figs/viewers/mujoco_viewer.gif) | ![](figs/viewers/rerun_viewer.gif) |


### Multiple simulators support:

| Genesis | Mujoco Warp |
| ------- | ------- |
| ![](figs/sim/dexmachina.gif) | ![](figs/sim/mjwarp.gif) |

### Deployment to real-world robots:

| Pick Cup | Rotate Bulb | Unplug Charger | Pick Duck |
| ------- | ------- | ------- | ------- |
| ![](figs/real/pick_cup_real.gif) | ![](figs/real/rotate_bulb_real.gif) | ![](figs/real/unplug_real.gif) | ![](figs/real/pick_duck_real.gif) |


## Features

- First general **physics-based** retargeting pipeline for both dexterous hand and humanoid robot.
- Supports 9+ robots and 6+ datasets out of the box.
- Seemless integration with RL training and data augmentation for BC pipeline.
- Native support for multiple simulators (Mujoco Wrap, Genesis) and multiple downstream training pipelines (HDMI, DexMachina).
- Sim2real ready.

![](figs/embodiment_support.png)

## Quickstart

Clone example datasets:

```bash
sudo apt install git-lfs
git lfs install
git clone https://huggingface.co/datasets/retarget/retarget_example example_datasets
```

### (Option 1) Quickstart with uv:

Create env and install (make sure `uv` uses Python 3.12, which is what the project targets):

```bash
# one-time install of the interpreter if you don't already have it
uv python install 3.12

# sync the .venv with that interpreter
uv sync --python 3.12
pip install --ignore-requires-python --no-deps -e .
```

If you already have the example datasets cloned, you can skip the preprocessing step where we convert the human data to robot kinematic trajectories.
Run SPIDER on a processed trial:

```bash
uv run examples/run_mjwp.py
```

For full workflow, please refer to the [Workflow](#workflow) section.

### (Option 2) Quickstart with conda:

```bash
conda create -n spider python=3.12
conda activate spider
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

Run MJWP on a processed trial:

```bash
python examples/run_mjwp.py
```

## Workflow

SPIDER is designed to support multiple workflows depending on your simulator of choice and downstream tasks.
- Native Mujoco Wrap (MJWP) is the default workflow and supports dexterous hand and humanoid robot retargeting.
- We also support [Genesis](https://genesis.github.io/) simulator with [DexMachina](https://github.com/MandiZhao/dexmachina), workflow is useful for further training a policy with RL for dexterous hand.
- [HDMI](https://github.com/lecar-lab/hdmi) workflow supports humanoid robot retargeting + RL workflow with humanoid-object interaction tasks. It use [MjLab](https://github.com/mujocolab/mjlab) as its backend simulator.

### Native Mujoco Wrap Workflow

- supports dexterous hand and humanoid robot retargeting

```bash
TASK=p36-tea
HAND_TYPE=bimanual
DATA_ID=10
ROBOT_TYPE=xhand
DATASET_NAME=gigahand

# put your raw data under folder raw/{dataset_name/ in your dataset folder

# read data from self collected dataset
uv run spider/process_datasets/gigahand.py --task=${TASK} --embodiment-type=${HAND_TYPE} --data-id=${DATA_ID}

# decompose object
uv run spider/preprocess/decompose_fast.py --task=${TASK} --dataset-name=${DATASET_NAME} --data-id=${DATA_ID} --embodiment-type=${HAND_TYPE}

# detect contact (optional)
uv run spider/preprocess/detect_contact.py --task=${TASK} --dataset-name=${DATASET_NAME} --data-id=${DATA_ID} --embodiment-type=${HAND_TYPE}

# generate scene
uv run spider/preprocess/generate_xml.py --task=${TASK} --dataset-name=${DATASET_NAME} --data-id=${DATA_ID} --embodiment-type=${HAND_TYPE} --robot-type=${ROBOT_TYPE}

# kinematic retargeting
uv run spider/preprocess/ik.py --task=${TASK} --dataset-name=${DATASET_NAME} --data-id=${DATA_ID} --embodiment-type=${HAND_TYPE} --robot-type=${ROBOT_TYPE} --open-hand

# retargeting
uv run examples/run_mjwp.py +override=${DATASET_NAME} task=${TASK} data_id=${DATA_ID} robot_type=${ROBOT_TYPE} embodiment_type=${HAND_TYPE}
```

### DexMachina Workflow

```bash
# install dexmachina conda environment following their official instructions: https://mandizhao.github.io/dexmachina-docs/0_install.html
conda activate dexmachina
# note: install spider only without mujoco warp since we only use the optimization part
pip install --ignore-requires-python --no-deps -e .
# run retargeting
python examples/run_dexmachina.py
```

### HDMI Workflow

```bash
# install HDMI uv environment following their official instructions:
# go to hdmi folder, install SPIDER with
uv pip install --no-deps -e ../spider
```


## Remote Development

```bash
# start rerun server
uv run rerun --serve-web --port 9876

# run SPIDER only with rerun viewer
uv run examples/run_mjwp.py viewer="rerun"
```

## Acknowledgments

- Thanks Mandi Zhao for the help with the [DexMachina workflow](https://github.com/MandiZhao/dexmachina) for SPIDER + Genesis.
- Thanks Taylor Howell for the help in the early stages of integrating [Mujoco Wrap](https://github.com/google-deepmind/mujoco_warp) for SPIDER + MJWP.
- Thanks Haoyang Weng for the help with the [HDMI workflow](https://github.com/lecar-lab/hdmi) for SPIDER + Sim2real RL.
- Inverse kinematics design is ported from [GMR](https://github.com/YanjieZe/GMR) and [LocoMujoco](https://github.com/robfiras/loco-mujoco).
- Dataset processing is ported from [Hot3D](https://github.com/facebookresearch/hot3d), [Oakinkv2](https://github.com/oakink/OakInk2), [Maniptrans](https://github.com/ManipTrans/ManipTrans), [Gigahands](https://github.com/Gigahands/Gigahands).
- Visualization inspired by other good sampling repo like [Hydrax](https://github.com/vincekurtz/hydrax) and [Judo](https://github.com/bdaiinstitute/judo).


## Citation

```bibtex
```
