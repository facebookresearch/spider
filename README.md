# SPIDER: Scalable Physics-Informed DExterous Retargeting

## Quickstart (uv):

Create env and install:

```bash
uv sync
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

## FAIR Internal Workflows

Montereal:

```bash
# put data in example_datasets/raw/fair_mon/{task}_{hand_type}/{data_id}.pkl
# e.g. example_datasets/raw/fair_mon/cat_right/0.pkl
TASK=cat
HAND_TYPE=right
DATA_ID=0

# read data
cd spider/process_datasets
uv run fair_mon.py --task=${TASK} --right-object-name=cat --hand-type=${HAND_TYPE} --data-id=${DATA_ID}

# decompose object
cd spider/preprocess
uv run decompose.py --task=${TASK} --dataset-name=fair_mon --data-id=${DATA_ID} --hand-type=${HAND_TYPE}

# detect contact (optional)
cd spider/preprocess
uv run detect_contact.py --task=${TASK} --dataset-name=fair_mon --data-id=${DATA_ID} --hand-type=${HAND_TYPE}

# generate scene
cd spider/preprocess
uv run generate_xml.py --task=${TASK} --dataset-name=fair_mon --data-id=${DATA_ID} --hand-type=${HAND_TYPE} --robot-type=metahand

# kinematic retargeting
cd spider/preprocess
uv run ik.py --task=${TASK} --dataset-name=fair_mon --data-id=${DATA_ID} --hand-type=${HAND_TYPE} --robot-type=metahand --open-hand

# retargeting
cd spider/examples
uv run run_mjwp.py +override=fair_mon
```
