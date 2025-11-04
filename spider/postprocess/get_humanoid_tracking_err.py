"""Get tracking error of generated humanoid trajectories.

Input: trajectory_mjwp.npz from generated datasets
Output: tracking error of position, orientation, and joint angles
"""

import os

import numpy as np
import torch
import tyro

from spider import ROOT
from spider.io import get_processed_data_dir
from spider.math import quat_sub


def main(
    dataset_dir: str = f"{ROOT}/../example_datasets",
    dataset_name: str = "amass",
    robot_type: str = "unitree_g1",
    embodiment_type: str = "humanoid",
    task: str = "sprint",
    data_id: int = 0,
):
    dataset_dir = os.path.abspath(dataset_dir)
    processed_dir = get_processed_data_dir(
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        robot_type=robot_type,
        embodiment_type=embodiment_type,
        task=task,
        data_id=data_id,
    )
    print(f"Processed directory: {processed_dir}")
    # load trajectory data
    trajectory_mjwp_path = (
        f"{processed_dir}/trajectory_hdmi.npz"
        if dataset_name == "hdmi"
        else f"{processed_dir}/trajectory_mjwp.npz"
    )
    trajectory_kinematic_path = f"{processed_dir}/trajectory_kinematic.npz"
    if not os.path.exists(trajectory_mjwp_path):
        raise FileNotFoundError(
            f"trajectory_mjwp.npz not found at {trajectory_mjwp_path}"
        )
    if not os.path.exists(trajectory_kinematic_path):
        raise FileNotFoundError(
            f"trajectory_kinematic.npz not found at {trajectory_kinematic_path}"
        )
    trajectory_mjwp_data = np.load(trajectory_mjwp_path)
    qpos_mjwp = trajectory_mjwp_data["qpos"]
    qpos_mjwp = qpos_mjwp.reshape(-1, qpos_mjwp.shape[-1])
    trajectory_kinematic_data = np.load(trajectory_kinematic_path)
    qpos_kinematic = trajectory_kinematic_data["qpos"]
    # compute substeps
    num_substeps = int(np.round(qpos_mjwp.shape[0] / qpos_kinematic.shape[0]))
    if num_substeps > 1:
        H = int(num_substeps * qpos_kinematic.shape[0])
        qpos_mjwp = qpos_mjwp[:H][::num_substeps]
    else:
        H = min(qpos_mjwp.shape[0], qpos_kinematic.shape[0])
        qpos_mjwp = qpos_mjwp[:H]
        qpos_kinematic = qpos_kinematic[:H]
    # compute tracking error
    pos_err = np.linalg.norm(qpos_mjwp[:, :3] - qpos_kinematic[:, :3], axis=1).mean()
    quat_err_torch = quat_sub(
        torch.from_numpy(qpos_mjwp[:, 3:7]),
        torch.from_numpy(qpos_kinematic[:, 3:7]),
    )
    quat_err = np.linalg.norm(quat_err_torch.numpy(), axis=1).mean()
    if dataset_name == "hdmi":
        qpos_mjwp_joint = qpos_mjwp[:, 7:-7]
        qpos_kinematic_joint = qpos_kinematic[:, 7:-7]
        qpos_obj = qpos_mjwp[:, -7:]
        qpos_obj_kinematic = qpos_kinematic[:, -7:]
        joint_err = np.linalg.norm(
            qpos_mjwp_joint - qpos_kinematic_joint, axis=1
        ).mean()
        obj_err = np.linalg.norm(qpos_obj - qpos_obj_kinematic, axis=1).mean()
        obj_quat_err_torch = quat_sub(
            torch.from_numpy(qpos_obj[:, 3:7]),
            torch.from_numpy(qpos_obj_kinematic[:, 3:7]),
        )
        obj_quat_err = np.linalg.norm(obj_quat_err_torch.numpy(), axis=1).mean()
        print(
            f"Tracking error: \npos_err: {pos_err:.2f}\nquat_err: {quat_err:.2f}\njoint_err: {joint_err:.2f}\nobj_err: {obj_err:.2f}\nobj_quat_err: {obj_quat_err:.2f}"
        )
    else:
        qpos_mjwp_joint = qpos_mjwp[:, 7:]
        qpos_kinematic_joint = qpos_kinematic[:, 7:]
        joint_err = np.linalg.norm(
            qpos_mjwp_joint - qpos_kinematic_joint, axis=1
        ).mean()
        print(
            f"Tracking error: \npos_err: {pos_err:.2f}\nquat_err: {quat_err:.2f}\njoint_err: {joint_err:.2f}"
        )


if __name__ == "__main__":
    tyro.cli(main)
