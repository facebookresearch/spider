"""Get contact information from mujoco simulation. Replay the trajectory and locate contact points.

Author: Chaoyi Pan
Date: 2025-07-07
"""

import json
import os
from contextlib import contextmanager

import loguru
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
import tyro
from loop_rate_limiters import RateLimiter

import spider
from spider.io import get_processed_data_dir


def main(
    dataset_dir: str = f"{spider.ROOT}/../example_datasets",
    dataset_name: str = "oakink",
    embodiment_type: str = "bimanual",
    task: str = "pick_spoon_bowl",
    data_id: int = 0,
    show_viewer: bool = True,
    save_video: bool = False,
    plot_contact: bool = False,
    ref_dt: float = 0.02,
):
    dataset_dir = os.path.abspath(dataset_dir)
    robot_type = "mano"
    processed_dir = get_processed_data_dir(
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        robot_type=robot_type,
        embodiment_type=embodiment_type,
        task=task,
        data_id=data_id,
    )
    os.makedirs(processed_dir, exist_ok=True)
    # load qpos
    qpos_path = f"{processed_dir}/trajectory_keypoints.npz"
    if not os.path.exists(qpos_path):
        raise FileNotFoundError(
            f"trajectory_keypoints.npz not found at {qpos_path}. Run dataset preprocessing first."
        )
    data = np.load(qpos_path)
    qpos_wrist_right = data["qpos_wrist_right"]
    qpos_finger_right = data["qpos_finger_right"]
    qpos_obj_right = data["qpos_obj_right"]
    qpos_wrist_left = data["qpos_wrist_left"]
    qpos_finger_left = data["qpos_finger_left"]
    qpos_obj_left = data["qpos_obj_left"]
    # check if contact and contact_pos is in the data
    if (
        "contact_left" in data
        and "contact_pos_left" in data
        and "contact_right" in data
        and "contact_pos_right" in data
    ):
        add_contact_site = True
        loguru.logger.info("Contact and contact_pos already in the data")
    else:
        add_contact_site = False

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
    loguru.logger.info(f"Loaded qpos from {qpos_path}")

    # visualize the data
    mj_spec = mujoco.MjSpec.from_file(f"{spider.ROOT}/assets/mano/contact_scene.xml")

    # add right object to body "right_object"
    right_object_handle = mj_spec.worldbody.add_body(
        name="right_object",
        mocap=False,
    )
    right_object_handle.add_joint(
        name="right_object_collision_joint",
        type=mujoco.mjtJoint.mjJNT_FREE,
        armature=0.0001,
        frictionloss=0.0001,
    )
    # add left object to body "left_object"
    left_object_handle = mj_spec.worldbody.add_body(
        name="left_object",
        mocap=False,
    )
    left_object_handle.add_joint(
        name="left_object_collision_joint",
        type=mujoco.mjtJoint.mjJNT_FREE,
        armature=0.0001,
        frictionloss=0.0001,
    )

    # load task_info for object mesh paths
    task_info_path = f"{processed_dir}/../task_info.json"
    if not os.path.exists(task_info_path):
        loguru.logger.warning(
            f"task_info.json not found at {task_info_path}; proceeding without object meshes."
        )
        right_object_files, left_object_files = [], []
    else:
        with open(task_info_path) as f:
            task_info = json.load(f)
        right_convex_dir = task_info.get("right_object_convex_dir")
        right_convex_dir = f"{dataset_dir}/{right_convex_dir}"
        left_convex_dir = task_info.get("left_object_convex_dir")
        left_convex_dir = f"{dataset_dir}/{left_convex_dir}"

        if (
            embodiment_type in ["right", "bimanual"]
            and right_convex_dir
            and os.path.isdir(right_convex_dir)
        ):
            right_object_files = [
                f
                for f in os.listdir(right_convex_dir)
                if f.endswith(".obj") and f.split(".")[0].isdigit()
            ]
            for f in right_object_files:
                suffix = f.split(".")[0]
                file_path = f"{right_convex_dir}/{f}"
                mj_spec.add_mesh(
                    name=f"right_{suffix}",
                    file=file_path,
                )
        else:
            right_object_files = []

        if (
            embodiment_type in ["left", "bimanual"]
            and left_convex_dir
            and os.path.isdir(left_convex_dir)
        ):
            left_object_files = [
                f
                for f in os.listdir(left_convex_dir)
                if f.endswith(".obj") and f.split(".")[0].isdigit()
            ]
            for f in left_object_files:
                suffix = f.split(".")[0]
                file_path = f"{left_convex_dir}/{f}"
                mj_spec.add_mesh(
                    name=f"left_{suffix}",
                    file=file_path,
                )
        else:
            left_object_files = []

    right_object_collision_names = []
    if embodiment_type in ["right", "bimanual"]:
        for f in right_object_files:
            suffix = f.split(".")[0]
            right_object_handle.add_geom(
                name=f"right_object_{suffix}",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=f"right_{suffix}",
                pos=[0, 0, 0],
                quat=[1, 0, 0, 0],
                group=0,
                condim=1,
                contype=0,
                conaffinity=0,
            )
            right_object_collision_names.append(f"right_object_{suffix}")
    else:
        right_object_handle.add_geom(
            name="right_object_mass",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=[1.0, 1.0, 1.0],
            size=[0.01, 0.01, 0.01],
            group=0,
            condim=1,
            contype=0,
            conaffinity=0,
        )

    left_object_collision_names = []
    if embodiment_type in ["left", "bimanual"]:
        for f in left_object_files:
            suffix = f.split(".")[0]
            left_object_handle.add_geom(
                name=f"left_object_{suffix}",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=f"left_{suffix}",
                pos=[0, 0, 0],
                quat=[1, 0, 0, 0],
                group=0,
                condim=1,
                contype=0,
                conaffinity=0,
            )
            left_object_collision_names.append(f"left_object_{suffix}")
        if len(left_object_collision_names) == 0:
            left_object_handle.add_geom(
                name="left_object_mass",
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                pos=[1.0, 1.0, 1.0],
                size=[0.01, 0.01, 0.01],
                group=0,
                condim=1,
                contype=0,
                conaffinity=0,
            )
    else:
        left_object_handle.add_geom(
            name="left_object_mass",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=[1.0, 1.0, 1.0],
            size=[0.01, 0.01, 0.01],
            group=0,
            condim=1,
            contype=0,
            conaffinity=0,
        )

    object_collision_names = right_object_collision_names + left_object_collision_names

    # add contact pairs
    hand_collision_names = []
    right_hand_collision_names = [
        "right_thumb_tip",
        "right_index_tip",
        "right_middle_tip",
        "right_ring_tip",
        "right_pinky_tip",
    ]
    left_hand_collision_names = [
        "left_thumb_tip",
        "left_index_tip",
        "left_middle_tip",
        "left_ring_tip",
        "left_pinky_tip",
    ]
    if embodiment_type in ["right", "bimanual"]:
        hand_collision_names.extend(right_hand_collision_names)
    if embodiment_type in ["left", "bimanual"]:
        hand_collision_names.extend(left_hand_collision_names)

    for object_collision_name in object_collision_names:
        for hand_collision_name in hand_collision_names:
            mj_spec.add_pair(
                name=f"{hand_collision_name}_{object_collision_name}",
                geomname1=hand_collision_name,
                geomname2=object_collision_name,
            )

    # add finger tip relative position sensor
    if len(left_object_collision_names) == 0:
        object_list = ["right_object", "right_object"]
    else:
        object_list = ["right_object", "left_object"]
    for finger_name in right_hand_collision_names:
        mj_spec.add_sensor(
            type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
            name=f"{finger_name}_p_{object_list[0]}",
            objtype=mujoco.mjtObj.mjOBJ_XBODY,
            objname=finger_name,
            reftype=mujoco.mjtObj.mjOBJ_XBODY,
            refname=object_list[0],
        )
    for finger_name in left_hand_collision_names:
        mj_spec.add_sensor(
            type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
            name=f"{finger_name}_p_{object_list[1]}",
            objtype=mujoco.mjtObj.mjOBJ_XBODY,
            objname=finger_name,
            reftype=mujoco.mjtObj.mjOBJ_XBODY,
            refname=object_list[1],
        )

    if add_contact_site:
        for i, finger_name in enumerate(right_hand_collision_names):
            if "right" in object_list[0]:
                handle = right_object_handle
                contact_pos = data["contact_pos_right"][i]
            else:
                handle = left_object_handle
                contact_pos = data["contact_pos_left"][i]
            handle.add_site(
                name=f"{finger_name}_{object_list[0]}_contact_site",
                pos=contact_pos,
                size=[0.01, 0.01, 0.01],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                rgba=[0, 1, 0, 1],
                group=4,
            )
        for i, finger_name in enumerate(left_hand_collision_names):
            if "left" in object_list[1]:
                handle = left_object_handle
                contact_pos = data["contact_pos_left"][i]
            else:
                handle = right_object_handle
                contact_pos = data["contact_pos_right"][i]
            handle.add_site(
                name=f"{finger_name}_{object_list[1]}_contact_site",
                pos=contact_pos,
                size=[0.01, 0.01, 0.01],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                rgba=[0, 1, 0, 1],
                group=4,
            )

    mj_model = mj_spec.compile()
    mj_data = mujoco.MjData(mj_model)
    rate_limiter = RateLimiter(1.0 / ref_dt)

    if save_video:
        import imageio

        mj_model.vis.global_.offwidth = 720
        mj_model.vis.global_.offheight = 480
        renderer = mujoco.Renderer(mj_model, height=480, width=720)

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
        images = []
        N = len(qpos_wrist_right)
        Ncontact = 10  # if embodiment_type == "bimanual" else 5
        contact_seq = np.zeros((N, Ncontact))
        contact_pos = np.zeros((N, Ncontact, 3))
        while gui.is_running():
            mj_data.qpos[:] = qpos_list[cnt].flatten()
            mj_data.qvel[:] = 0
            mujoco.mj_step(mj_model, mj_data)
            if save_video:
                renderer.update_scene(mj_data, gui.cam)
                images.append(renderer.render())
            # get sensor data
            contact = mj_data.sensordata[:Ncontact] > 0
            # get fingertip local position
            fingertip_local_pos = mj_data.sensordata[Ncontact:].reshape(
                -1, 3
            )  # bimanual: (10, 3) right: (5, 3) left: (5, 3)
            contact_seq[cnt, :] = contact
            contact_pos[cnt, :, :] = fingertip_local_pos
            cnt = (cnt + 1) % N
            if cnt == (N - 1):
                # post process contact_seq by sliding window
                contact_seq_smooth = np.zeros_like(contact_seq)
                for i in range(N):
                    contact_seq_smooth[i, :] = np.any(
                        contact_seq[np.maximum(0, i - 1) : np.minimum(N, i + 1), :],
                        axis=0,
                    )
                # get contact position
                avg_contact_pos = np.zeros((Ncontact, 3))
                std_contact_pos = np.zeros((Ncontact, 3))
                for i in range(Ncontact):
                    cnt = 0
                    contact_pos_list = []
                    for j in range(N):
                        if contact_seq[j, i]:
                            avg_contact_pos[i, :] += contact_pos[j, i, :]
                            contact_pos_list.append(contact_pos[j, i, :])
                            cnt += 1
                    if cnt > 0:
                        avg_contact_pos[i, :] /= cnt
                        contact_pos_list = np.array(contact_pos_list)
                        std_contact_pos[i, :] = np.std(contact_pos_list, axis=0)
                for i in range(Ncontact):
                    if std_contact_pos[i, :].max() > 0.1:
                        loguru.logger.warning(
                            "contact point moving in a large range, disable contact point"
                        )
                        contact_seq_smooth[:, i] = False
                        avg_contact_pos[i, :] = np.zeros(3)

                if embodiment_type == "bimanual":
                    contact_right = contact_seq_smooth[:, :5]
                    contact_left = contact_seq_smooth[:, 5:]
                    contact_pos_right = avg_contact_pos[:5, :]
                    contact_pos_left = avg_contact_pos[5:, :]
                elif embodiment_type == "right":
                    contact_right = contact_seq_smooth
                    contact_pos_right = avg_contact_pos
                    contact_left = np.zeros_like(contact_right)
                    contact_pos_left = np.zeros_like(contact_pos_right)
                elif embodiment_type == "left":
                    contact_left = contact_seq_smooth
                    contact_pos_left = avg_contact_pos
                    contact_right = np.zeros_like(contact_left)
                    contact_pos_right = np.zeros_like(contact_pos_left)
                else:
                    raise ValueError(f"Invalid hand type: {embodiment_type}")
                # save contact_seq_smooth to npz
                # save contact info to processed dir for future reference
                contact_out_path = f"{processed_dir}/trajectory_keypoints.npz"
                np.savez(
                    contact_out_path,
                    qpos_wrist_right=qpos_wrist_right,
                    qpos_finger_right=qpos_finger_right,
                    qpos_obj_right=qpos_obj_right,
                    qpos_wrist_left=qpos_wrist_left,
                    qpos_finger_left=qpos_finger_left,
                    qpos_obj_left=qpos_obj_left,
                    contact_left=contact_left,
                    contact_pos_left=contact_pos_left,
                    contact_right=contact_right,
                    contact_pos_right=contact_pos_right,
                )
                loguru.logger.info(f"Saved contact data to {contact_out_path}")
                if plot_contact:
                    fig, axs = plt.subplots(2, 5, figsize=(10, 10))
                    for i in range(5):
                        axs[0, i].plot(contact_seq_smooth[:, i])
                        axs[1, i].plot(contact_seq_smooth[:, i + 5])
                    plt.show()
                if not show_viewer:
                    break
            if show_viewer:
                gui.sync()
                rate_limiter.sleep()

    if save_video:
        file_dir = f"../../outputs/mano/{embodiment_type}/{task}"
        os.makedirs(file_dir, exist_ok=True)
        imageio.mimsave(f"{file_dir}/contact.mp4", images, fps=30)
        loguru.logger.info(f"Saved video to {file_dir}/contact.mp4")


if __name__ == "__main__":
    tyro.cli(main)
