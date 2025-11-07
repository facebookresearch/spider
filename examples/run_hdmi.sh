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
# python examples/run_hdmi.py task=move_largebox viewer="mujoco-rerun" rerun_spawn=true horizon=0.8 knot_dt=0.2 ctrl_dt=0.04 num_samples=1024  temperature=0.1 max_num_iterations=32 joint_noise_scale=0.2 first_ctrl_noise_scale=1.0 improvement_check_steps=1 improvement_threshold=0.02 max_sim_steps=100 #ctrl_dt=0.02 knot_dt=0.02 horizon=0.02 joint_noise_scale=0.0001 max_num_iterations=1

python examples/run_hdmi.py task=move_suitcase joint_noise_scale=0.2 knot_dt=0.2 ctrl_dt=0.04 horizon=0.8  +data_id=1 viewer="mujoco-rerun" rerun_spawn=true +save_rerun=true +save_metrics=false max_sim_steps=-1
