"""
Preprocess data from loco mujoco.

Process:
1. move z-axis to make sure initial frame the foot is on the ground

Input: npz file from loco mujoco initial retargeting, `qpos` is used.

Output: npz file containing improved qpos, qvel, ctrl, contact, contact_pos.

Author: Chaoyi Pan
Date: 2025-07-18
"""

import os
import glob
import numpy as np
import loguru
from loop_rate_limiters import RateLimiter
import mujoco
import mujoco.viewer
import tyro
from contextlib import contextmanager
from matplotlib import pyplot as plt


def main(
    robot_type: str = "g1",
    task: str = "CMU/CMU/01/01_01_poses",
    show_viewer: bool = True,
    max_sim_steps: int = -1,
):
    # load data
    robot_name_locomujoco = {
        "g1": "UnitreeG1",
    }[robot_type]
    data_dir = (
        f"{os.path.expanduser('~')}/.loco-mujoco-caches/AMASS/{robot_name_locomujoco}"
    )
    data_path = f"{data_dir}/{task}.npz"
    data = np.load(data_path)
    qpos_list = data["qpos"]
    qvel_list = data["qvel"]
    max_sim_steps = qpos_list.shape[0] if max_sim_steps == -1 else max_sim_steps
    qpos_list = qpos_list[:max_sim_steps]
    qvel_list = qvel_list[:max_sim_steps]

    # load mujoco model
    hand_type = task.split("/")[0]
    task_id = task.split("/")[-1]
    model_path = f"../assets/{robot_type}/scene_{hand_type}_{task_id}.xml"
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos[:] = qpos_list[0, :]
    mujoco.mj_step(mj_model, mj_data)

    foot_site_ids = []
    for side in ["left", "right"]:
        for i in range(1, 5):
            idx = mujoco.mj_name2id(
                mj_model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_foot_{i}"
            )
            foot_site_ids.append(idx)
    foot_site_ids = np.array(foot_site_ids)
    foot_poses = mj_data.site_xpos[foot_site_ids, :]
    foot_height_avg = np.mean(foot_poses[:, 2])
    # offset qpos in z-axis
    qpos_list[:, 2] -= foot_height_avg

    # detect contact, record important site positions
    body_site_names = [
        "pelvis",
        "torso",
        "left_wrist",
        "right_wrist",
    ]
    body_site_ids = []
    for name in body_site_names:
        idx = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, name)
        body_site_ids.append(idx)
    body_site_ids = np.array(body_site_ids)
    contact_seq = np.zeros((qpos_list.shape[0], 8))
    contact_pos = np.zeros((qpos_list.shape[0], 8, 3))
    body_pos = np.zeros((qpos_list.shape[0], len(body_site_names), 3))
    body_quat = np.zeros((qpos_list.shape[0], len(body_site_names), 4))
    for i in range(qpos_list.shape[0]):
        mj_data.qpos[:] = qpos_list[i, :]
        mujoco.mj_step(mj_model, mj_data)
        contact_pos[i, :] = mj_data.site_xpos[foot_site_ids, :]
        contact_seq[i, :] = contact_pos[i, :, 2] < 0.001
        body_pos[i, :] = mj_data.site_xpos[body_site_ids, :]
        xmat = mj_data.site_xmat[body_site_ids, :]
        # convert xmat to quat
        for j in range(len(body_site_names)):
            mujoco.mju_mat2Quat(body_quat[i, j], xmat[j])

    # smooth contact sequence
    contact_seq_smooth = np.zeros_like(contact_seq)
    for i in range(qpos_list.shape[0]):
        past_contact_seq = contact_seq[np.maximum(0, i - 2) : i, :]
        past_contact = np.any(past_contact_seq, axis=0, keepdims=True)
        future_contact_seq = contact_seq[
            i + 0 : np.minimum(qpos_list.shape[0], i + 2), :
        ]
        future_contact = np.any(future_contact_seq, axis=0, keepdims=True)
        contact_seq_smooth[i, :] = np.any(
            np.concatenate([past_contact, future_contact], axis=0),
            axis=0,
        )
    contact_seq = contact_seq_smooth

    # merge contact seq (left and right share the same contact seq)
    contact_seq_left = contact_seq[:, :4].sum(axis=1, keepdims=True) >= 1
    contact_seq_right = contact_seq[:, 4:].sum(axis=1, keepdims=True) >= 1
    contact_seq = np.concatenate(
        [contact_seq_left] * 4 + [contact_seq_right] * 4, axis=1
    )

    # smooth contact pos
    for i in range(8):
        # average out the contact pos in contact phase
        # find continuous contact sequences (consecutive 1s)
        if np.any(contact_seq[:, i]):
            # find transitions from 0 to 1 (start of contact) and 1 to 0 (end of contact)
            diff = np.diff(np.concatenate(([0], contact_seq[:, i], [0])).astype(int))
            start_indices = np.where(diff == 1)[0]  # start of contact sequences
            end_indices = np.where(diff == -1)[0]  # end of contact sequences

            # for each continuous contact sequence, average the positions
            for start_idx, end_idx in zip(start_indices, end_indices):
                if start_idx < end_idx:  # valid contact sequence
                    # compute mean position during this contact period
                    mean_pos = np.mean(contact_pos[start_idx:end_idx, i, :], axis=0)
                    # set z pos to 0 to make sure stable contact
                    mean_pos[2] = 0.005
                    # set all positions in this period to the mean
                    contact_pos[start_idx:end_idx, i, :] = mean_pos

    # plot contact sequence
    if show_viewer:
        plt.close("all")
        fig, axs = plt.subplots(4, 4, figsize=(10, 10))
        for j in range(2):
            for i in range(4):
                axs[j, i].plot(contact_seq[:, j * 4 + i])
        for j in range(2):
            for i in range(4):
                # plot contact pos in x-z plane with color, label with contact as red and without contact as green
                Npoints = 400
                axs[j + 2, i].scatter(
                    contact_pos[:Npoints, j * 4 + i, 0],
                    contact_pos[:Npoints, j * 4 + i, 2],
                    c=contact_seq[:Npoints, j * 4 + i],
                    cmap="RdYlGn",
                )
        plt.show()

    # regenerate motion with constraints in foot contact
    # create new mujoco model with equality constraints in foot contact
    mj_spec = mujoco.MjSpec.from_file(model_path)
    # add mocap body for body sites
    for name in body_site_names:
        b = mj_spec.worldbody.add_body(
            name=f"{name}_mocap",
            mocap=True,
        )
        b.add_site(
            name=f"{name}_mocap",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.01, 0.01, 0.01],
            rgba=[0.0, 1.0, 0.0, 0.5],
            group=1,
        )
        e = mj_spec.add_equality(
            name=f"{name}_mocap",
            type=mujoco.mjtEq.mjEQ_WELD,
            name1=f"{name}_mocap",
            name2=name,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
        )
        e.solref = [0.02, 1.0]
        e.solimp = [0.9, 0.95, 0.02, 0.5, 2.0]

    for side in ["left", "right"]:
        for i in range(1, 5):
            e = mj_spec.add_equality(
                name=f"{side}_foot_{i}",
                type=mujoco.mjtEq.mjEQ_CONNECT,
                name1=f"{side}_foot_{i}",
                name2=f"{side}_foot_{i}_mocap",
                objtype=mujoco.mjtObj.mjOBJ_SITE,
                data=np.zeros(11),
            )
            e.solref = [0.02, 1.0]
            e.solimp = [0.9, 0.95, 0.005, 0.5, 2.0]

    # simulated again with filtered state
    mj_model = mj_spec.compile()
    foot_mocap_ids = []
    for side in ["left", "right"]:
        for i in range(1, 5):
            body_id = mujoco.mj_name2id(
                mj_model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_foot_{i}"
            )
            mocap_id = mj_model.body_mocapid[body_id]
            foot_mocap_ids.append(mocap_id)
    body_mocap_ids = []
    for name in body_site_names:
        body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, f"{name}_mocap")
        mocap_id = mj_model.body_mocapid[body_id]
        body_mocap_ids.append(mocap_id)
    foot_mocap_ids = np.array(foot_mocap_ids)
    mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    mj_model.opt.gravity = [0, 0, 0]
    mj_data = mujoco.MjData(mj_model)

    # process all states
    for i in range(qpos_list.shape[0]):
        for _ in range(16):
            mj_data.qvel[:] = 0.0
            mj_data.ctrl[:] = qpos_list[i, 7:]
            mj_data.mocap_pos[foot_mocap_ids] = contact_pos[i, :]
            mj_data.mocap_pos[body_mocap_ids] = body_pos[i, :]
            mj_data.mocap_quat[body_mocap_ids] = body_quat[i, :]
            mujoco.mj_step(mj_model, mj_data)
        qpos_list[i, :] = mj_data.qpos.copy()
        contact_pos[i, :] = mj_data.mocap_pos[foot_mocap_ids].copy()

    # scan qpos_list, find large jump between two frames, replace it with the average of the two frames
    for i in range(1, qpos_list.shape[0] - 1):
        if np.linalg.norm(qpos_list[i, :] - qpos_list[i - 1, :]) > 0.1:
            qpos_list[i, :] = (qpos_list[i + 1, :] + qpos_list[i - 1, :]) / 2

    # compute qvel from qpos
    ref_dt = 0.02
    for i in range(qpos_list.shape[0] - 1):
        mujoco.mj_differentiatePos(
            mj_model,
            qvel_list[i, :].astype(np.float64),
            ref_dt,
            qpos_list[i, :].astype(np.float64),
            qpos_list[i + 1, :].astype(np.float64),
        )
    qpos_list = qpos_list[:-1]
    qvel_list = qvel_list[:-1]

    rate_limiter = RateLimiter(1.0 / ref_dt)
    if show_viewer:
        run_viewer = lambda: mujoco.viewer.launch_passive(mj_model, mj_data)
    else:

        @contextmanager
        def run_viewer():
            yield type(
                "DummyViewer",
                (),
                {"is_running": lambda: True, "sync": lambda: None},
            )

    # visualize the motion
    with run_viewer() as gui:
        cnt = 0
        while gui.is_running():
            mj_data.qpos[:] = qpos_list[cnt, :]
            mj_data.mocap_pos[foot_mocap_ids] = contact_pos[cnt, :]
            mj_data.mocap_quat[body_mocap_ids] = body_quat[cnt, :]
            cnt = cnt + 1
            if cnt >= qpos_list.shape[0]:
                break
            mujoco.mj_step(mj_model, mj_data)
            if show_viewer:
                gui.sync()
                rate_limiter.sleep()

    # save data
    new_task_name = task.split("/")[-1]
    new_mode = task.split("/")[0]
    # new_data_path = f"../../datasets/g1/{new_mode}_{new_task_name}.npz"
    new_data_dir = f"../../outputs/{robot_type}/{new_mode}/ik/{new_task_name}"
    os.makedirs(new_data_dir, exist_ok=True)
    new_data_path = f"{new_data_dir}/0.npz"
    np.savez(
        new_data_path,
        qpos=qpos_list,
        qvel=qvel_list,
        ctrl=qpos_list[:, 7:],
        contact=contact_seq,
        contact_pos=contact_pos,
    )
    loguru.logger.info(f"Saved data to {new_data_path}")

    # optional: for gradient-based method use, save to binary file for cpp use
    # binary_dir = "/home/pcy/Research/code/mujoco_mpc-fork/mjpc/tasks/g1"
    # os.makedirs(binary_dir, exist_ok=True)

    # # Convert to float32 and save binary files
    # qpos_binary = qpos_list.astype(np.float32)
    # qvel_binary = qvel_list.astype(np.float32)

    # qpos_binary_path = f"{binary_dir}/{new_mode}_{new_task_name}_qpos.bin"
    # qvel_binary_path = f"{binary_dir}/{new_mode}_{new_task_name}_qvel.bin"

    # qpos_binary.tofile(qpos_binary_path)
    # qvel_binary.tofile(qvel_binary_path)

    # loguru.logger.info(f"Saved binary qpos data to {qpos_binary_path}")
    # loguru.logger.info(f"Saved binary qvel data to {qvel_binary_path}")
    # loguru.logger.info(
    #     f"qpos shape: {qpos_binary.shape}, qvel shape: {qvel_binary.shape}"
    # )


if __name__ == "__main__":
    tyro.cli(main)
