"""Process Oakink data (preprocessed by maniptrans).

Process:
1. Convert mesh to obj
2. Convert original data to finger tip, object and wrist position and rotation
3. Visualize the data in mujoco viewer

Input: pickle file from Changhao.

Output: npz file containing:
    qpos_wrist_left, qpos_finger_left, qpos_obj_left, qpos_wrist_right, qpos_finger_right, qpos_obj_right

Author: Chaoyi Pan
Date: 2025-07-07
"""

import io
import json
import os
import pickle
from contextlib import contextmanager

import loguru
import mujoco
import mujoco.viewer
import numpy as np
import pymeshlab
import torch
import tyro
from loop_rate_limiters import RateLimiter
from scipy.spatial.transform import Rotation as R

from spider.io import get_mesh_dir, get_processed_data_dir

os.environ["CUDA_VISIBLE_DEVICES"] = ""


class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


def main(
    dataset_dir: str = "../../example_datasets",
    embodiment_type: str = "bimanual",
    task: str = "pick_spoon_bowl",
    show_viewer: bool = True,
    save_video: bool = False,
    start_idx: int = 0,
):
    # path related
    dataset_dir = os.path.abspath(dataset_dir)
    file_path = f"{dataset_dir}/raw/oakink/{task}_{embodiment_type}.pkl"
    output_dir = get_processed_data_dir(
        dataset_dir=dataset_dir,
        dataset_name="oakink",
        robot_type="mano",
        embodiment_type=embodiment_type,
        task=task,
        data_id=0,
    )
    os.makedirs(output_dir, exist_ok=True)

    # task info
    task_info = {
        "task": task,
        "dataset_name": "oakink",
        "robot_type": "mano",
        "embodiment_type": embodiment_type,
        "data_id": 0,
        "right_object_mesh_dir": None,
        "left_object_mesh_dir": None,
        "ref_dt": 0.02,
    }

    with open(file_path, "rb") as f:
        data = CPUUnpickler(f).load()
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

    # read right object data
    if embodiment_type in ["right", "bimanual"]:
        right_obj_mesh_path = data_right["obj_mesh_path"][0]
        right_obj_mesh_name = right_obj_mesh_path.split("align_ds/")[1]
        right_obj_mesh_path = f"{dataset_dir}/raw/oakink/meshes/{right_obj_mesh_name}"
        right_obj_mesh_name = right_obj_mesh_name.split("/")[0]
        # check if the file exists
        if not os.path.exists(right_obj_mesh_path):
            raise FileNotFoundError(f"File {right_obj_mesh_path} not found")

        # convert mesh to obj and save to mesh_dir/{file_name}/visual.obj
        mesh_dir = get_mesh_dir(
            dataset_dir=dataset_dir,
            dataset_name="oakink",
            object_name=right_obj_mesh_name,
        )
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(right_obj_mesh_path)
        if not os.path.exists(mesh_dir):
            os.makedirs(mesh_dir)
        ms.save_current_mesh(f"{mesh_dir}/visual.obj")
        task_info["right_object_mesh_dir"] = mesh_dir

    # read left object data
    if embodiment_type in ["left", "bimanual"]:
        left_obj_mesh_path = data_left["obj_mesh_path"][0]
        left_obj_mesh_name = left_obj_mesh_path.split("align_ds/")[1]
        left_obj_mesh_path = f"{dataset_dir}/raw/oakink/meshes/{left_obj_mesh_name}"
        left_obj_mesh_name = left_obj_mesh_name.split("/")[0]
        if left_obj_mesh_name == right_obj_mesh_name:
            loguru.logger.info(
                "Left and right object mesh paths are the same; setting left object mesh directory to None."
            )
            task_info["left_object_mesh_dir"] = None
        else:
            # check if the file exists
            if not os.path.exists(left_obj_mesh_path):
                raise FileNotFoundError(f"File {left_obj_mesh_path} not found")
            # convert mesh to obj and save to mesh_dir/{file_name}/visual.obj
            mesh_dir = get_mesh_dir(
                dataset_dir=dataset_dir,
                dataset_name="oakink",
                object_name=left_obj_mesh_name,
            )
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(left_obj_mesh_path)
            if not os.path.exists(mesh_dir):
                os.makedirs(mesh_dir)
            ms.save_current_mesh(f"{mesh_dir}/visual.obj")
            task_info["left_object_mesh_dir"] = mesh_dir

    # persist task info for downstream tools (e.g., decomposition)
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
        if embodiment_type == "bimanual" and left_obj_mesh_path == right_obj_mesh_path:
            # reset left object trajectory to be the same as right object trajectory
            left_obj_trajectory = np.tile(np.eye(4), (N, 1, 1))
        else:
            left_obj_trajectory = data_left["obj_trajectory"].cpu().numpy()
    else:
        left_obj_trajectory = np.tile(np.eye(4), (N, 1, 1))

    # global rotation
    r_global = R.from_euler("xyz", [np.pi / 2, 0, 0])

    # get qpos
    N = len(right_wrist_pos)
    unit_quat = np.array([1, 0, 0, 0])
    qpos_wrist_right = np.zeros((N, 7))
    qpos_finger_right = np.zeros((N, 5, 7))
    qpos_obj_right = np.zeros((N, 7))
    qpos_wrist_left = np.zeros((N, 7))
    qpos_finger_left = np.zeros((N, 5, 7))
    qpos_obj_left = np.zeros((N, 7))
    r_right_wrist_offset = R.from_euler("xyz", [np.pi / 2, 0, 0]) * R.from_euler(
        "xyz", [0, 0, np.pi]
    )
    r_left_wrist_offset = R.from_euler("xyz", [np.pi / 2, 0, 0]) * R.from_euler(
        "xyz", [0, 0, np.pi]
    )
    for i in range(N):
        # right hand
        qpos_wrist_right[i, :3] = r_global.apply(right_wrist_pos[i])
        # convert from axis-angle to quaternion
        r = r_global * R.from_rotvec(right_wrist_rot[i]) * r_right_wrist_offset
        xyzw = r.as_quat()
        wxyz = np.concatenate([xyzw[3:], xyzw[:3]])
        qpos_wrist_right[i, 3:] = wxyz
        # left hand
        qpos_wrist_left[i, :3] = r_global.apply(left_wrist_pos[i])
        # convert from axis-angle to quaternion
        r = r_global * R.from_rotvec(left_wrist_rot[i]) * r_left_wrist_offset
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
    np.savez(
        f"{output_dir}/trajectory_keypoints.npz",
        qpos_wrist_right=qpos_wrist_right[start_idx:],
        qpos_finger_right=qpos_finger_right[start_idx:],
        qpos_obj_right=qpos_obj_right[start_idx:],
        qpos_wrist_left=qpos_wrist_left[start_idx:],
        qpos_finger_left=qpos_finger_left[start_idx:],
        qpos_obj_left=qpos_obj_left[start_idx:],
    )
    loguru.logger.info(f"Saved qpos to {output_dir}/trajectory_keypoints.npz")

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
    mj_spec = mujoco.MjSpec.from_file(f"{spider.ROOT}/assets/mano/empty_scene.xml")

    # add right object to body "right_object"
    object_right_handle = mj_spec.worldbody.add_body(
        name="right_object",
        mocap=True,
    )
    object_right_handle.add_site(
        name="right_object",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.01, 0.02, 0.03],
        rgba=[1, 0, 0, 1],
        group=0,
    )

    if embodiment_type in ["right", "bimanual"]:
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
    if (
        embodiment_type in ["left", "bimanual"]
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

    mj_model = mj_spec.compile()
    mj_data = mujoco.MjData(mj_model)
    rate_limiter = RateLimiter(120.0)
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

    if save_video:
        import imageio

        mj_model.vis.global_.offwidth = 720
        mj_model.vis.global_.offheight = 480
        renderer = mujoco.Renderer(mj_model, height=480, width=720)
        images = []
    with run_viewer() as gui:
        cnt = 0
        contact_seq = np.zeros((N, 10))
        while gui.is_running():
            mj_data.mocap_pos[:] = qpos_list[cnt, :, :3]
            mj_data.mocap_quat[:] = qpos_list[cnt, :, 3:]
            mujoco.mj_step(mj_model, mj_data)
            cnt = (cnt + 1) % N
            if save_video:
                renderer.update_scene(mj_data, gui.cam)
                img = renderer.render()
                images.append(img)
            if cnt == (N - 1):
                if save_video:
                    imageio.mimsave(f"{output_dir}/visualization.mp4", images, fps=120)
                    loguru.logger.info(f"Saved video to {output_dir}/visualization.mp4")
                if not show_viewer:
                    break
            if show_viewer:
                gui.sync()
                rate_limiter.sleep()


if __name__ == "__main__":
    tyro.cli(main)
