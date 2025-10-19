#! /bin/zsh
export DISPLAY=:1
HDMI_DIR="/home/pcy/Research/code/HDMI"

# source HDMI env
source ${HDMI_DIR}/.venv/bin/activate

# Install required packages
# uv pip install opencv-python rerun-sdk loguru hydra-core omegaconf

# run SPIDER
# viewer options:
# - "mjlab" for mjlab viewer only
# - "mujoco" for mujoco viewer only
# - "rerun" for rerun viewer only
# - "mujoco-rerun" for both mujoco and rerun (recommended for SPIDER)
# - "mjlab-mujoco-rerun" for all three viewers (mjlab may conflict)
python examples/run_hdmi.py viewer="mjlab-rerun" rerun_spawn=true #num_samples=1024 ctrl_dt=0.02 knot_dt=0.02 horizon=0.02 joint_noise_scale=0.0001 max_num_iterations=1
