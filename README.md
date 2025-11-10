# SPIDER: Scalable Physics-Informed DExterous Retargeting

## Quickstart (uv):

Create env and install:

```bash
uv sync
pip install --ignore-requires-python --no-deps -e .
```

Clone example datasets:

```bash
sudo apt install git-lfs
git lfs install
git clone https://huggingface.co/datasets/retarget/retarget_example example_datasets
```

Run MJWP on a processed trial:

```bash
uv run examples/run_mjwp.py
```

## Native Workflow

- supports dexterous hand and humanoid robot retargeting

## DexMachina Workflow

```bash
conda activate dexmachina
# note: install spider only without mujoco warp since we only use the optimization part
pip install --ignore-requires-python --no-deps -e .
# run retargeting
python examples/run_dexmachina.py
```

## HDMI Workflow

```bash
# go to hdmi folder, install spider with
uv pip install --no-deps -e ../spider
```

## FAIR Internal Workflows

Montereal:

```bash
# put data in example_datasets/raw/fair_mon/{task}_{embodiment_type}/{data_id}.pkl
# e.g. example_datasets/raw/fair_mon/cat_right/0.pkl
TASK=coke
HAND_TYPE=right
DATA_ID=0
ROBOT_TYPE=allegro
DATASET_NAME=fair_mon

# read data
uv run spider/process_datasets/fair_mon.py --task=${TASK} --right-object-name=${RIGHT_OBJECT_NAME} --embodiment-type=${HAND_TYPE} --data-id=${DATA_ID}

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

## Remote Development

```bash
# start rerun server
uv run rerun --serve-web --port 9876
```

## Notes

1. IK is important. try to rerun ik if the retargeting is not good.
