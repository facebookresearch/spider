"""Process real world data from FAIR Fremont Campus.

1. Convert mesh to obj
2. Convert original data to finger tip, object and wrist position and rotation
3. Visualize the data in mujoco viewer

File path example: {dataset_dir}/raw/fair_fre/pickcup_right/{data_id}.pkl

Input: pickle file from Changhao.
Output: npz file containing:
    qpos_wrist_left, qpos_finger_left, qpos_obj_left, qpos_wrist_right, qpos_finger_right, qpos_obj_right

Author: Chaoyi Pan
Date: 2025-07-07
"""

import json
import os
import pickle
from contextlib import contextmanager

import imageio
import loguru
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
import pymeshlab
import torch
import tyro
from loop_rate_limiters import RateLimiter
from scipy.spatial.transform import Rotation as R

from spider.io import get_mesh_dir, get_processed_data_dir


def main(
    dataset_dir: str = "../../example_datasets",
    embodiment_type: str = "right",
    data_id: int = 0,
    task: str = "pickcup",
    right_object_name: str = "pickcup",
    left_object_name: str | None = None,
    show_viewer: bool = True,
    max_steps: int = -1,
    save_video: bool = False,
    filter_sudden_changes: bool = True,
    z_offset: float = 0.14,
):
    dataset_dir = os.path.abspath(dataset_dir)
    output_dir = get_processed_data_dir(
        dataset_dir=dataset_dir,
        dataset_name="fair_fre",
        robot_type="mano",
        embodiment_type=embodiment_type,
        task=task,
        data_id=data_id,
    )
    os.makedirs(output_dir, exist_ok=True)
    file_path = f"{dataset_dir}/raw/fair_fre/{task}_{embodiment_type}/{data_id}.pkl"
    with open(file_path, "rb") as f:
        data_raw = pickle.load(f)
    data_raw = dict(data_raw)

    # read mesh info for this task/hand from info.json
    info_path = f"{dataset_dir}/raw/fair_fre/{task}_{embodiment_type}/info.json"
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"File {info_path} not found")
    with open(info_path) as f:
        info = json.load(f)
    right_mesh_filename = info.get("right_object_mesh", None)
    left_mesh_filename = info.get("left_object_mesh", None)

    # get object initial pose
    obj_initial_pose = data_raw["object_trajectory"][0].copy()
    if task == "pickbox":
        # manually resolve object initial pose ambiguity
        mat = np.eye(4)
        mat[:3, :3] = R.from_euler("xyz", [-np.pi / 2, -np.pi / 2, 0]).as_matrix()
        obj_initial_pose = obj_initial_pose @ mat
        obj_initial_pose[2, 3] += z_offset
    elif task == "bottle":
        obj_initial_pose[2, 3] += z_offset
    elif task == "wipe":
        # manually resolve object initial pose ambiguity
        mat = np.eye(4)
        # mat[:3, :3] = R.from_euler("xyz", [np.pi / 2, 0, 0]).as_matrix()
        obj_initial_pose[:3, :3] = mat[:3, :3]
        obj_initial_pose[2, 3] -= z_offset
    else:
        mat = np.eye(4)
        obj_initial_pose[:3, :3] = mat[:3, :3]
        obj_initial_pose[2, 3] -= z_offset
    data_right_hand = data_raw["right_hand"]
    data_right_hand["object_trajectory"] = data_raw["object_trajectory"].copy()
    data_left_hand = data_raw["left_hand"]
    if data_left_hand is not None:
        data_left_hand["object_trajectory"] = data_raw["object_trajectory"].copy()
    data = {}
    for side, data_hand in [("right", data_right_hand), ("left", data_left_hand)]:
        if data_hand is None:
            data[side] = {}
            continue
        # convert everything to object frame
        for i in range(len(data_hand["wrist_poses"])):
            data_hand["wrist_poses"][i] = (
                np.linalg.inv(obj_initial_pose) @ data_hand["wrist_poses"][i]
            )
            # r = R.from_euler("y", -np.pi / 2)
            # mat = np.eye(4)
            # mat[:3, :3] = r.as_matrix()
            data_hand["wrist_poses"][i] = data_hand["wrist_poses"][i]  # @ mat
            # rotate wrist_poses around x axis by np.pi
            # r = R.from_euler("x", np.pi / 2)
            # mat = np.eye(4)
            # mat[:3, :3] = r.as_matrix()
            # data_raw["wrist_poses"][i] = data_raw["wrist_poses"][i] @ mat
            finger_tip_pos = np.zeros((5, 3))
            for j in range(5):
                tip_pos = data_hand["fingertip_keypoints"][i, j, :3]
                tip_pos_expand = np.concatenate([tip_pos, [1]])
                tip_pos_transform = np.linalg.inv(obj_initial_pose) @ tip_pos_expand
                finger_tip_pos[j] = tip_pos_transform[:3]
                # finger_tip_pos[j, 2] = np.maximum(finger_tip_pos[j, 2], 0.01)
                # if i > 100:
                #     finger_tip_pos[j, 2] = 0.01
            data_hand["fingertip_keypoints"][i] = finger_tip_pos
            data_hand["object_trajectory"][i] = (
                np.linalg.inv(obj_initial_pose) @ data_hand["object_trajectory"][i]
            )
            # fix object z
            # data_hand["object_trajectory"][i, 2, 3] = 0.02

        # filter out sudden changes in wrist_poses
        if filter_sudden_changes:
            for i in range(1, len(data_hand["wrist_poses"])):
                if (
                    np.linalg.norm(
                        data_hand["wrist_poses"][i][:3, 3]
                        - data_hand["wrist_poses"][i - 1][:3, 3]
                    )
                    > 0.1
                ):
                    data_hand["wrist_poses"][i] = data_hand["wrist_poses"][i - 1]
                    data_hand["fingertip_keypoints"][i] = data_hand[
                        "fingertip_keypoints"
                    ][i - 1]
                    data_hand["object_trajectory"][i] = data_hand["object_trajectory"][
                        i - 1
                    ]

        # convert data_raw to data
        wrist_rot = []
        for i in range(len(data_hand["wrist_poses"])):
            r = R.from_matrix(data_hand["wrist_poses"][i][:3, :3])
            wrist_rot.append(r.as_rotvec())
        wrist_rot = np.array(wrist_rot)
        object_name = right_object_name if side == "right" else left_object_name
        object_mesh_path = []
        data[side] = {
            "wrist_pos": torch.from_numpy(data_hand["wrist_poses"][:, :3, 3]),
            "wrist_rot": torch.from_numpy(wrist_rot),
            "mano_joints": torch.from_numpy(data_hand["fingertip_keypoints"]),
            "obj_trajectory": torch.from_numpy(data_hand["object_trajectory"]),
            "obj_mesh_path": object_mesh_path,
        }

    data_right = data["right"]
    data_left = data["left"]
    try:
        N = len(data_right["wrist_pos"])
    except:
        N = len(data_left["wrist_pos"])

    # read right hand data
    if embodiment_type in ["right", "bimanual"]:
        right_wrist_pos = data_right["wrist_pos"].cpu().numpy()
        right_wrist_rot = data_right["wrist_rot"].cpu().numpy()
        right_mano_joints = data_right["mano_joints"].cpu().numpy()
        if len(right_mano_joints.shape) == 2:
            right_mano_joints = right_mano_joints.reshape(-1, 5, 3)
    else:
        right_wrist_pos = np.zeros((N, 3))
        right_wrist_rot = np.zeros((N, 3))
        right_mano_joints = np.zeros((N, 5, 3))

    # read left hand data
    if embodiment_type in ["left", "bimanual"]:
        left_wrist_pos = data_left["wrist_pos"].cpu().numpy()
        left_wrist_rot = data_left["wrist_rot"].cpu().numpy()
        left_mano_joints = data_left["mano_joints"].cpu().numpy()
        if len(left_mano_joints.shape) == 2:
            left_mano_joints = left_mano_joints.reshape(-1, 5, 3)
    else:
        left_wrist_pos = np.zeros((N, 3))
        left_wrist_rot = np.zeros((N, 3))
        left_mano_joints = np.zeros((N, 5, 3))

    # Prepare task info and copy meshes into standardized mesh dirs
    task_info = {
        "task": task,
        "dataset_name": "fair_fre",
        "robot_type": "mano",
        "embodiment_type": embodiment_type,
        "data_id": data_id,
        "right_object_mesh_dir": None,
        "left_object_mesh_dir": None,
    }

    # read right object data (mesh copy)
    if embodiment_type in ["right", "bimanual"] and right_mesh_filename is not None:
        right_obj_mesh_path = f"{dataset_dir}/raw/fair_fre/meshes/{right_mesh_filename}"
        if not os.path.exists(right_obj_mesh_path):
            raise FileNotFoundError(f"File {right_obj_mesh_path} not found")
        right_object_name = os.path.splitext(os.path.basename(right_mesh_filename))[0]
        mesh_dir = get_mesh_dir(
            dataset_dir=dataset_dir,
            dataset_name="fair_fre",
            object_name=right_object_name,
        )
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(right_obj_mesh_path)
        if not os.path.exists(mesh_dir):
            os.makedirs(mesh_dir)
        ms.save_current_mesh(f"{mesh_dir}/visual.obj")
        loguru.logger.info(f"Saved {right_obj_mesh_path} mesh to {mesh_dir}/visual.obj")
        task_info["right_object_mesh_dir"] = mesh_dir

    # read left object data (mesh copy)
    if embodiment_type in ["left", "bimanual"] and left_mesh_filename is not None:
        left_obj_mesh_path = f"{dataset_dir}/raw/fair_fre/meshes/{left_mesh_filename}"
        if not os.path.exists(left_obj_mesh_path):
            raise FileNotFoundError(f"File {left_obj_mesh_path} not found")
        left_object_name = os.path.splitext(os.path.basename(left_mesh_filename))[0]
        if (
            task_info["right_object_mesh_dir"] is not None
            and left_object_name
            == os.path.basename(task_info["right_object_mesh_dir"]).split("/")[-1]
        ):
            loguru.logger.info(
                "Left and right object mesh names are the same; skipping left mesh copy."
            )
            task_info["left_object_mesh_dir"] = None
        else:
            mesh_dir = get_mesh_dir(
                dataset_dir=dataset_dir,
                dataset_name="fair_fre",
                object_name=left_object_name,
            )
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(left_obj_mesh_path)
            if not os.path.exists(mesh_dir):
                os.makedirs(mesh_dir)
            ms.save_current_mesh(f"{mesh_dir}/visual.obj")
            loguru.logger.info(
                f"Saved {left_obj_mesh_path} mesh to {mesh_dir}/visual.obj"
            )
            task_info["left_object_mesh_dir"] = mesh_dir

    # persist task info
    task_info_path = f"{output_dir}/../task_info.json"
    with open(task_info_path, "w") as f:
        json.dump(task_info, f, indent=2)
    loguru.logger.info(f"Saved task_info to {task_info_path}")

    # read right object trajectory
    if embodiment_type in ["right", "bimanual"]:
        right_obj_trajectory = data_right["obj_trajectory"].cpu().numpy()
    else:
        right_obj_trajectory = np.tile(np.eye(4), (N, 1, 1))

    # read left object trajectory
    if embodiment_type in ["left", "bimanual"]:
        if embodiment_type == "bimanual" and left_mesh_filename is None:
            # reset left object trajectory to be the same as right object trajectory
            left_obj_trajectory = np.tile(np.eye(4), (N, 1, 1))
        else:
            left_obj_trajectory = data_left["obj_trajectory"].cpu().numpy()
    else:
        left_obj_trajectory = np.tile(np.eye(4), (N, 1, 1))

    # global rotation
    # r_global = R.from_euler("xyz", [+np.pi / 2, 0, 0])
    r_global = R.from_euler("xyz", [0, 0, 0])

    # get qpos
    if max_steps > 0:
        N = min(N, max_steps)
        right_wrist_pos = right_wrist_pos[:N]
        right_wrist_rot = right_wrist_rot[:N]
        right_mano_joints = right_mano_joints[:N]
        left_wrist_pos = left_wrist_pos[:N]
        left_wrist_rot = left_wrist_rot[:N]
        left_mano_joints = left_mano_joints[:N]
        right_obj_trajectory = right_obj_trajectory[:N]
        left_obj_trajectory = left_obj_trajectory[:N]
    else:
        N = len(right_wrist_pos)
    unit_quat = np.array([1, 0, 0, 0])
    qpos_wrist_right = np.zeros((N, 7))
    qpos_finger_right = np.zeros((N, 5, 7))
    qpos_obj_right = np.zeros((N, 7))
    qpos_wrist_left = np.zeros((N, 7))
    qpos_finger_left = np.zeros((N, 5, 7))
    qpos_obj_left = np.zeros((N, 7))
    r_wrist_offset = R.from_euler("xyz", [0, 0, 0])
    for i in range(N):
        # right hand
        qpos_wrist_right[i, :3] = r_global.apply(right_wrist_pos[i])
        # convert from axis-angle to quaternion
        r = r_global * R.from_rotvec(right_wrist_rot[i]) * r_wrist_offset
        xyzw = r.as_quat()
        wxyz = np.concatenate([xyzw[3:], xyzw[:3]])
        qpos_wrist_right[i, 3:] = wxyz
        # left hand
        qpos_wrist_left[i, :3] = r_global.apply(left_wrist_pos[i])
        # convert from axis-angle to quaternion
        r = r_global * R.from_rotvec(left_wrist_rot[i]) * r_wrist_offset
        xyzw = r.as_quat()
        wxyz = np.concatenate([xyzw[3:], xyzw[:3]])
        qpos_wrist_left[i, 3:] = wxyz
    for i in range(N):
        for j in range(5):
            # right hand
            qpos_finger_right[i, j, :3] = r_global.apply(right_mano_joints[i, j])
            qpos_finger_right[i, j, 3:] = unit_quat
            # left hand
            qpos_finger_left[i, j, :3] = r_global.apply(left_mano_joints[i, j])
            qpos_finger_left[i, j, 3:] = unit_quat
    for i in range(N):
        # right object
        T_obj = right_obj_trajectory[i]
        # get position from translation matrix
        qpos_obj_right[i, :3] = r_global.apply(T_obj[:3, 3])
        # get rotation from rotation matrix
        r = r_global * R.from_matrix(T_obj[:3, :3])
        xyzw = r.as_quat()
        wxyz = np.concatenate([xyzw[3:], xyzw[:3]])
        qpos_obj_right[i, 3:] = wxyz
        # left object
        T_obj = left_obj_trajectory[i]
        # get position from translation matrix
        qpos_obj_left[i, :3] = r_global.apply(T_obj[:3, 3])
        # get rotation from rotation matrix
        r = r_global * R.from_matrix(T_obj[:3, :3])
        xyzw = r.as_quat()
        wxyz = np.concatenate([xyzw[3:], xyzw[:3]])
        qpos_obj_left[i, 3:] = wxyz

    # save qpos
    output_dir = get_processed_data_dir(
        dataset_dir=dataset_dir,
        dataset_name="fair_fre",
        robot_type="mano",
        embodiment_type=embodiment_type,
        task=task,
        data_id=data_id,
    )
    os.makedirs(output_dir, exist_ok=True)

    # hardcoding: process hand data
    # for i in range(32, 280):
    #     qpos_obj_right[i, :3] = qpos_finger_right[i, :, :3].mean(axis=0)
    #     qpos_obj_right[i, 3:] = qpos_obj_right[31, 3:]

    np.savez(
        f"{output_dir}/trajectory_kinematic.npz",
        qpos_wrist_right=qpos_wrist_right,
        qpos_finger_right=qpos_finger_right,
        qpos_obj_right=qpos_obj_right,
        qpos_wrist_left=qpos_wrist_left,
        qpos_finger_left=qpos_finger_left,
        qpos_obj_left=qpos_obj_left,
    )
    loguru.logger.info(f"Saved qpos to {output_dir}/trajectory_kinematic.npz")

    qpos_list = np.concatenate(
        [
            qpos_wrist_right[:, None],
            qpos_finger_right,
            qpos_wrist_left[:, None],
            qpos_finger_left,
            qpos_obj_right[:, None],
            qpos_obj_left[:, None],
        ],
        axis=1,
    )

    # visualize the data
    # mj_model = mujoco.MjModel.from_xml_path("../assets/mano/empty_scene.xml")
    mj_spec = mujoco.MjSpec.from_file("../assets/mano/empty_scene.xml")

    # add right object to body "right_object"
    object_right_handle = mj_spec.worldbody.add_body(
        name="right_object",
        mocap=True,
    )
    collision_right_handle = mj_spec.worldbody.add_body(
        name="right_object_collision",
        mocap=False,
    )
    collision_right_handle.add_joint(
        name="right_object_collision_joint",
        type=mujoco.mjtJoint.mjJNT_FREE,
        armature=0.0001,
        frictionloss=0.0001,
    )
    object_right_handle.add_site(
        name="right_object",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.01, 0.02, 0.03],
        rgba=[1, 0, 0, 1],
        group=0,
    )
    if (
        embodiment_type in ["right", "bimanual"]
        and task_info["right_object_mesh_dir"] is not None
    ):
        mj_spec.add_mesh(
            name="right_object",
            file=f"{task_info['right_object_mesh_dir']}/visual.obj",
        )
        object_right_handle.add_geom(
            name="right_object",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="right_object",
            pos=[0, 0, 0],
            quat=[1, 0, 0, 0],
            group=0,
            condim=1,
        )
        collision_right_handle.add_geom(
            name="right_collision",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="right_object",
            pos=[0, 0, 0],
            quat=[1, 0, 0, 0],
            group=0,
            condim=1,
            contype=0,
            conaffinity=0,
        )

    # add left object to body "left_object"
    object_left_handle = mj_spec.worldbody.add_body(
        name="left_object",
        mocap=True,
    )
    object_left_handle.add_site(
        name="left_object",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.01, 0.02, 0.03],
        rgba=[0, 1, 0, 1],
        group=0,
    )
    collision_left_handle = mj_spec.worldbody.add_body(
        name="left_object_collision",
        mocap=False,
    )
    collision_left_handle.add_joint(
        name="left_object_collision_joint",
        type=mujoco.mjtJoint.mjJNT_FREE,
        armature=0.0001,
        frictionloss=0.0001,
    )
    bimanual_single_object = (
        embodiment_type == "bimanual" and task_info["left_object_mesh_dir"] is None
    )
    if (
        embodiment_type in ["left", "bimanual"]
        and not bimanual_single_object
        and task_info["left_object_mesh_dir"] is not None
    ):
        # add left object to body "left_object"
        mj_spec.add_mesh(
            name="left_object",
            file=f"{task_info['left_object_mesh_dir']}/visual.obj",
        )
        object_left_handle.add_geom(
            name="left_object",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="left_object",
            pos=[0, 0, 0],
            quat=[1, 0, 0, 0],
            group=0,
            condim=1,
            contype=0,
            conaffinity=0,
        )
        collision_left_handle.add_geom(
            name="left_collision",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="left_object",
            pos=[0, 0, 0],
            quat=[1, 0, 0, 0],
            group=0,
            condim=1,
            contype=1,
            conaffinity=1,
        )
    else:
        # add placeholder to avoid simulation error
        collision_left_handle.add_geom(
            name="left_collision",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.01, 0.02, 0.03],
            pos=[0, 0, 0],
            quat=[1, 0, 0, 0],
            group=0,
        )

    # manually specify contact pairs
    fingertips_right = [
        "right_thumb_tip",
        "right_index_tip",
        "right_middle_tip",
        "right_ring_tip",
        "right_pinky_tip",
    ]
    fingertips_left = [
        "left_thumb_tip",
        "left_index_tip",
        "left_middle_tip",
        "left_ring_tip",
        "left_pinky_tip",
    ]
    if embodiment_type == "right":
        fingertips = fingertips_right
        objects = ["right_collision"]
    elif embodiment_type == "left":
        fingertips = fingertips_left
        objects = ["left_collision"]
    elif embodiment_type == "bimanual":
        fingertips = fingertips_right + fingertips_left
        objects = ["right_collision", "left_collision"]
    else:
        raise ValueError(f"Invalid hand type: {embodiment_type}")

    for fingertip in fingertips:
        for object_name in objects:
            mj_spec.add_pair(
                name=f"{fingertip}_{object_name}",
                geomname1=fingertip,
                geomname2=object_name,
                condim=1,
            )

    mj_model = mj_spec.compile()
    # ---------------------------------------------------
    # Video recording setup (mimic implementation in ik.py)
    # ---------------------------------------------------
    if save_video:
        mj_model.vis.global_.offwidth = 720
        mj_model.vis.global_.offheight = 480
        renderer = mujoco.Renderer(mj_model, height=480, width=720)
        images = []

    mj_data = mujoco.MjData(mj_model)
    rate_limiter = RateLimiter(30.0)
    if show_viewer:
        run_viewer = lambda: mujoco.viewer.launch_passive(mj_model, mj_data)
    else:

        @contextmanager
        def run_viewer():
            yield type(
                "DummyViewer",
                (),
                {
                    "is_running": lambda: True,
                    "sync": lambda: None,
                    "cam": mujoco.MjvCamera(),
                },
            )

    with run_viewer() as gui:
        cnt = 0
        contact_seq = np.zeros((N, 10))
        while gui.is_running():
            mj_data.qpos[:7] = qpos_obj_right[cnt]
            mj_data.qpos[7:14] = qpos_obj_left[cnt]
            mj_data.mocap_pos[:] = qpos_list[cnt, :, :3]
            mj_data.mocap_quat[:] = qpos_list[cnt, :, 3:]
            mujoco.mj_step(mj_model, mj_data)
            # capture frame for video
            if save_video:
                renderer.update_scene(mj_data, gui.cam)
                images.append(renderer.render())
            # get sensor data
            contact = mj_data.sensordata > 0
            contact_seq[cnt, :] = contact
            cnt = (cnt + 1) % N
            if cnt == (N - 1):
                # do a sliding window to make the contact read more consistent
                contact_seq_smooth = np.zeros_like(contact_seq)
                for i in range(N):
                    contact_seq_smooth[i, :] = np.any(
                        contact_seq[np.maximum(0, i - 40) : np.minimum(N, i + 40), :],
                        axis=0,
                    )
                # save contact_seq_smooth to npz
                np.savez(
                    f"{output_dir}/trajectory_kinematic.npz",
                    qpos_wrist_right=qpos_wrist_right,
                    qpos_finger_right=qpos_finger_right,
                    qpos_obj_right=qpos_obj_right,
                    qpos_wrist_left=qpos_wrist_left,
                    qpos_finger_left=qpos_finger_left,
                    qpos_obj_left=qpos_obj_left,
                    contact=contact_seq_smooth,
                )
                loguru.logger.info(
                    f"Saved data to {output_dir}/trajectory_kinematic.npz"
                )
                if show_viewer:
                    # save video after simulation finishes
                    if save_video:
                        imageio.mimsave(
                            f"{output_dir}/visualization.mp4", images, fps=30
                        )
                        loguru.logger.info(
                            f"Saved video to {output_dir}/visualization.mp4"
                        )
                    fig, axs = plt.subplots(2, 5, figsize=(10, 10))
                    for i in range(5):
                        axs[0, i].plot(contact_seq_smooth[:, i])
                        axs[1, i].plot(contact_seq_smooth[:, i + 5])
                    plt.show()
                else:
                    break
            if show_viewer:
                gui.sync()
                rate_limiter.sleep()


if __name__ == "__main__":
    tyro.cli(main)
