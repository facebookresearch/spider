"""Convert Spider retarget outputs to the real-world data format used by Allegro."""

from __future__ import annotations

import os
from typing import Optional

import loop_rate_limiters
import mujoco
import mujoco.viewer
import numpy as np

from spider.io import get_processed_data_dir


def main(
    dataset_dir: str,
    dataset_name: str,
    robot_type: str,
    hand_type: str,
    task: str,
    data_type: str = "mjwp",
    data_id: int = 0,
    model_type: str = "standard",
    output_dir: Optional[str] = None,
):
    """Load Spider trajectory and export wrist/object poses for Allegro replay."""
    dataset_dir = os.path.abspath(dataset_dir)
    processed_dir_robot = get_processed_data_dir(
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        robot_type=robot_type,
        hand_type=hand_type,
        task=task,
        data_id=data_id,
    )

    xml_name = "scene.xml" if model_type == "standard" else "scene_eq.xml"
    model_path = f"{processed_dir_robot}/../{xml_name}"
    data_path = f"{processed_dir_robot}/trajectory_{data_type}.npz"

    output_root = output_dir if output_dir is not None else dataset_dir
    output_root = os.path.abspath(output_root)
    os.makedirs(output_root, exist_ok=True)
    output_path = (
        f"{output_root}/{dataset_name}/{robot_type}_{hand_type}_{task}_{data_id}.npz"
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    traj_data = np.load(data_path)
    qpos_list = traj_data["qpos"].reshape(-1, traj_data["qpos"].shape[-1])
    if data_type != "kinematic" and "ctrl" in traj_data:
        ctrl_list = traj_data["ctrl"].reshape(-1, traj_data["ctrl"].shape[-1])
    else:
        ctrl_list = None
    total_steps = qpos_list.shape[0]

    real_data = {
        "right_wrist_pos": np.zeros((total_steps, 3)),
        "right_wrist_xmat": np.zeros((total_steps, 3, 3)),
        "right_joint_pos": np.zeros((total_steps, 16)),
        "left_wrist_pos": np.zeros((total_steps, 3)),
        "left_wrist_xmat": np.zeros((total_steps, 3, 3)),
        "left_joint_pos": np.zeros((total_steps, 16)),
        "left_object_pose": np.zeros((total_steps, 7)),
        "right_object_pose": np.zeros((total_steps, 7)),
        "model_path": model_path,
    }

    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_model.opt.disableflags = mujoco.mjtDisableBit.mjDSBL_CONTACT
    mj_model.opt.timestep = 0.001
    mj_data = mujoco.MjData(mj_model)

    right_palm_site_id = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_SITE, "right_palm"
    )
    left_palm_site_id = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_SITE, "left_palm"
    )
    right_object_site_id = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_SITE, "right_object"
    )
    left_object_site_id = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_SITE, "left_object"
    )

    rate_limiter = loop_rate_limiters.RateLimiter(100)
    saved = False
    with mujoco.viewer.launch_passive(mj_model, mj_data) as gui:
        step = 0
        while gui.is_running():
            mj_data.qpos[:] = qpos_list[step]
            if ctrl_list is not None:
                mj_data.ctrl[:] = ctrl_list[step]
            mujoco.mj_step(mj_model, mj_data)

            real_data["right_wrist_pos"][step] = mj_data.site_xpos[right_palm_site_id]
            real_data["right_wrist_xmat"][step] = mj_data.site_xmat[
                right_palm_site_id
            ].reshape(3, 3)
            real_data["right_joint_pos"][step] = mj_data.qpos[6:22]

            real_data["left_wrist_pos"][step] = mj_data.site_xpos[left_palm_site_id]
            real_data["left_wrist_xmat"][step] = mj_data.site_xmat[
                left_palm_site_id
            ].reshape(3, 3)
            real_data["left_joint_pos"][step] = mj_data.qpos[28:44]

            right_object_pos = mj_data.site_xpos[right_object_site_id]
            right_object_xmat = mj_data.site_xmat[right_object_site_id].reshape(3, 3)
            left_object_pos = mj_data.site_xpos[left_object_site_id]
            left_object_xmat = mj_data.site_xmat[left_object_site_id].reshape(3, 3)

            right_quat = np.zeros(4)
            mujoco.mju_mat2Quat(right_quat, right_object_xmat.flatten())
            real_data["right_object_pose"][step, :3] = right_object_pos
            real_data["right_object_pose"][step, 3:] = right_quat

            left_quat = np.zeros(4)
            mujoco.mju_mat2Quat(left_quat, left_object_xmat.flatten())
            real_data["left_object_pose"][step, :3] = left_object_pos
            real_data["left_object_pose"][step, 3:] = left_quat

            step += 1
            gui.sync()
            if step >= total_steps:
                if not saved:
                    np.savez(output_path, **real_data)
                    saved = True
                    return real_data
                step = 0

            rate_limiter.sleep()

    if not saved:
        np.savez(output_path, **real_data)
    return real_data
