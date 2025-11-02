"""This is converter for kinematic data from GMR

Please generate the pkl file from GMR first with command like:

For simpx
```bash
python scripts/smplx_to_robot.py --smplx_file "/home/pcy/Research/code/GMR/data/ACCAD/Male2MartialArtsExtended_c3d/Extended_1_stageii.npz" --robot "unitree_g1" --save_path "/home/pcy/Research/code/spider/example_datasets/processed/amass/g1/humanoid/martial_arts/0/trajectory_gmr.pkl"
```

For lafan1:
```bash
python scripts/bvh_to_robot.py --bvh_file "/home/pcy/Research/code/GMR/data/lafan1/dance1_subject1.bvh" --robot "unitree_g1" --save_path "/home/pcy/Research/code/spider/example_datasets/processed/lafan/g1/humanoid/dance/0/trajectory_gmr.pkl" --rate_limit --format "lafan1" --motion_fps 30
```

Input: pkl file from GMR
Output:
1. npz file containing:
    qpos, qvel, ctrl, contact
2. scene file including robot and object (for AMASS, no object is needed)

"""

import json
import os
import pickle
import shutil

import imageio
import mujoco
import numpy as np
import tyro
from loop_rate_limiters import RateLimiter

from spider import ROOT
from spider.io import get_processed_data_dir
from spider.mujoco_utils import get_viewer


def main(
    dataset_dir: str = f"{ROOT}/../example_datasets",
    dataset_name: str = "amass",
    robot_type: str = "unitree_g1",
    embodiment_type: str = "humanoid",
    task: str = "sprint",
    data_id: int = 0,
    show_viewer: bool = True,
    save_video: bool = True,
    overwrite: bool = True,
    enable_rate_limiter: bool = False,
    start_frame: int = 0,
    end_frame: int = -1,
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
    # load gmr pkl data
    gmr_pkl_path = f"{processed_dir}/trajectory_gmr.pkl"
    if not os.path.exists(gmr_pkl_path):
        raise FileNotFoundError(f"GMR pkl file not found at {gmr_pkl_path}")
    with open(gmr_pkl_path, "rb") as f:
        gmr_data = pickle.load(f)
    fps = gmr_data["fps"]
    print(f"fps: {fps}")
    root_pos = gmr_data["root_pos"]
    root_quat = gmr_data["root_rot"][:, [3, 0, 1, 2]]  # from xyzw to wxyz
    dof_pos = gmr_data["dof_pos"]
    qpos = np.concatenate([root_pos, root_quat, dof_pos], axis=-1)
    qpos = qpos[start_frame:end_frame]
    print(f"qpos shape: {qpos.shape}")

    # prepare scene file by copying from SPIDER assets
    # in SPIDER, we need 2 files: one for robot, another for scene
    # robot file
    src_robot_dir = f"{ROOT}/assets/robots/{robot_type}"
    tgt_robot_dir = f"{dataset_dir}/processed/{dataset_name}/assets/robots/{robot_type}"
    if not os.path.exists(tgt_robot_dir) or overwrite:
        shutil.copytree(src_robot_dir, tgt_robot_dir, dirs_exist_ok=True)
        print(f"copy from {src_robot_dir} to {tgt_robot_dir}")

    # create a scene file, which only includes the robot
    scene_dir = f"{processed_dir}/.."
    tgt_scene_file = f"{scene_dir}/scene.xml"
    src_scene_file = f"{tgt_robot_dir}/scene.xml"
    # copy
    shutil.copy(src_scene_file, tgt_scene_file)
    print(f"copy from {src_scene_file} to {tgt_scene_file}")

    # create task info file
    task_info_file = f"{scene_dir}/task_info.json"
    with open(task_info_file, "w") as f:
        json.dump({"ref_dt": 1.0 / fps}, f, indent=2)
    print(f"Saved task info to {task_info_file}")

    # run mujoco
    mj_model = mujoco.MjModel.from_xml_path(tgt_scene_file)
    mj_data = mujoco.MjData(mj_model)
    run_viewer = get_viewer(show_viewer, mj_model, mj_data)
    rate_limiter = RateLimiter(fps)
    # log info
    info_list = []
    # log video
    if save_video:
        images = []
        mj_model.vis.global_.offwidth = 720
        mj_model.vis.global_.offheight = 480
        renderer = mujoco.Renderer(mj_model, height=480, width=720)
    with run_viewer() as gui:
        for i in range(qpos.shape[0]):
            mj_data.qpos[:] = qpos[i]
            # compute qvel
            if i > 0:
                mujoco.mj_differentiatePos(
                    mj_model, mj_data.qvel, 1.0 / fps, qpos[i - 1], qpos[i]
                )
            else:
                mj_data.qvel[:] = 0.0
            # compute contact (currently it is a placeholder, will be implemented later)
            contact = np.zeros(1)
            # compute ctrl
            mj_data.ctrl[:] = qpos[i][7:]
            mujoco.mj_forward(mj_model, mj_data)
            # log
            info = {
                "qpos": mj_data.qpos.copy(),
                "qvel": mj_data.qvel.copy(),
                "ctrl": mj_data.ctrl.copy(),
                "contact": contact,
            }
            info_list.append(info)
            # render
            if save_video:
                renderer.update_scene(mj_data, "track")
                images.append(renderer.render())
            if show_viewer:
                gui.sync()
            if enable_rate_limiter:
                rate_limiter.sleep()
    info_aggregated = {}
    for key in info_list[0].keys():
        info_aggregated[key] = np.stack([info[key] for info in info_list], axis=0)
    np.savez(f"{processed_dir}/trajectory_kinematic.npz", **info_aggregated)
    print(f"Saved trajectory to {processed_dir}/trajectory_kinematic.npz")
    if save_video:
        imageio.mimsave(f"{processed_dir}/visualization_ik.mp4", images, fps=fps)
        print(f"Saved video to {processed_dir}/visualization_ik.mp4")


if __name__ == "__main__":
    tyro.cli(main)
