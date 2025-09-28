#!/bin/bash

# Example script showing how to run with Hydra configuration
# This replaces the original tyro command:
# uv run run_mjwp.py --task=${TASK} --dataset-name=fair_mon --robot-type=metahand --hand-type=${HAND_TYPE} --viewer=rerun-mujoco --rerun-spawn --ref-dt=0.05 --max-sim-steps=100 --horizon=1.0 --ctrl-dt=1.0 --knot-dt=0.5 --num-dr=2 --perturb-torque=0.01 --pos-noise-scale=0.1 --rot-noise-scale=0.01 --joint-noise-scale=0.3 --improvement-threshold=0.001 --max-num-iterations=16 --temperature=0.1

# Set environment variables if needed
export TASK=${TASK:-"pick_spoon_bowl"}
export HAND_TYPE=${HAND_TYPE:-"bimanual"}

# Run with Hydra using the fair_mon override configuration
cd examples
python run_mjwp.py \
    --config-path=config/overrides \
    --config-name=fair_mon \
    task=${TASK} \
    hand_type=${HAND_TYPE}

# Alternative: Run with command line overrides (if you don't want to use the override file)
# python run_mjwp.py \
#     task=${TASK} \
#     dataset_name=fair_mon \
#     robot_type=metahand \
#     hand_type=${HAND_TYPE} \
#     viewer=rerun-mujoco \
#     rerun_spawn=true \
#     ref_dt=0.05 \
#     max_sim_steps=100 \
#     horizon=1.0 \
#     ctrl_dt=1.0 \
#     knot_dt=0.5 \
#     num_dr=2 \
#     perturb_torque=0.01 \
#     pos_noise_scale=0.1 \
#     rot_noise_scale=0.01 \
#     joint_noise_scale=0.3 \
#     improvement_threshold=0.001 \
#     max_num_iterations=16 \
#     temperature=0.1
