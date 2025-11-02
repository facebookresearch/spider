import os
from contextlib import contextmanager

import loguru
import mujoco
import mujoco.viewer
import numpy as np
import tyro
from loop_rate_limiters import RateLimiter

from spider.io import get_processed_data_dir


def main(
    dataset_dir: str = "../../example_datasets",
    dataset_name: str = "oakink",
    robot_type: str = "allegro",
    embodiment_type: str = "bimanual",
    task: str = "pick_spoon_bowl",
    data_type: str = "kinematic",
    data_id: int = 0,
    model_type: str = "standard",  # "standard" | "eq"
    replay_speed: float = 1,
    sim_dt: float = 0.01,
    ref_dt: float = 0.02,
    ctrl_dt: float = 0.2,
    save_video: bool = False,
    show_viewer: bool = True,
    motion_length: int = -1,  # -1 means use all the motion
    noise_scale: float = 1.0,
    margin: float = 0.00,  # margin for the object
    ctrl_mode: str = "retarget",
    save_info: bool = False,
    enable_xfrc: bool = False,
):
    # resolve processed paths like in ik.py
    dataset_dir = os.path.abspath(dataset_dir)
    processed_dir_robot = get_processed_data_dir(
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        robot_type=robot_type,
        embodiment_type=embodiment_type,
        task=task,
        data_id=data_id,
    )
    xml_name = "scene.xml" if model_type == "standard" else "scene_eq.xml"
    model_path = f"{processed_dir_robot}/../{xml_name}"
    file_path = f"{processed_dir_robot}/trajectory_{data_type}.npz"

    # load model
    model = mujoco.MjModel.from_xml_path(model_path)
    model.opt.timestep = sim_dt
    model.opt.iterations = 20
    model.opt.ls_iterations = 50
    # enable override flag
    if margin > 0.0:
        model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_OVERRIDE
        model.opt.o_margin = margin
        model.opt.o_solimp = [0.9, 0.95, 0.001, 0.5, 2]
        model.opt.o_solref = [0.02, 1.0]
        model.opt.o_friction = [1.0, 1.0, 0.1, 0.0, 0.0]
    data = mujoco.MjData(model)

    traj_data = np.load(file_path)
    qpos_list = traj_data["qpos"].reshape(-1, model.nq)
    qvel_list = traj_data["qvel"].reshape(-1, model.nv)
    if "ctrl" in traj_data:
        ctrl_list = traj_data["ctrl"].reshape(-1, model.nu)
    else:
        if robot_type == "g1":
            ctrl_list = qpos_list[:, 7:]
        else:
            if embodiment_type == "bimanual":
                ctrl_list = qpos_list[:, :-14]
            else:
                ctrl_list = qpos_list[:, :-7]

    data.qpos[:] = qpos_list[0]
    data.qvel[:] = qvel_list[0]
    data.ctrl[:] = ctrl_list[0]
    mujoco.mj_forward(model, data)

    # save video
    if save_video:
        import imageio

        model.vis.global_.offwidth = 720
        model.vis.global_.offheight = 480
        renderer = mujoco.Renderer(model, height=480, width=720)
        images = []

    # create reference
    ref_data = mujoco.MjData(model)
    ref_data.qpos[:] = qpos_list[0]
    mujoco.mj_forward(model, ref_data)
    vopt = mujoco.MjvOption()
    vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True  # Transparent.
    pert = mujoco.MjvPerturb()
    catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC  # only show dynamic bodies

    rate_limiter = RateLimiter(1 / sim_dt * replay_speed)
    info_list = []
    # generate control sequence
    ctrl_list = ctrl_list.copy()
    ctrl_list_repeat = ctrl_list[:, None, :].repeat(int(ref_dt / sim_dt), axis=1)
    default_ctrl_list = ctrl_list_repeat.reshape(-1, ctrl_list.shape[1])
    qpos_list = qpos_list.copy()
    qpos_list_repeat = qpos_list[:, None, :].repeat(int(ref_dt / sim_dt), axis=1)
    qpos_list = qpos_list_repeat.reshape(-1, qpos_list.shape[1])
    qvel_list = qvel_list.copy()
    qvel_list_repeat = qvel_list[:, None, :].repeat(int(ref_dt / sim_dt), axis=1)
    qvel_list = qvel_list_repeat.reshape(-1, qvel_list.shape[1])
    noise_scale = np.ones(model.nu) * 0.2 * noise_scale
    total_time = default_ctrl_list.shape[0] * sim_dt
    if embodiment_type in ["left", "right"]:
        noise_scale[:3] = 0.01
        noise_scale[3:6] = 0.03
    elif embodiment_type == "bimanual":
        half_dof = model.nu // 2
        noise_scale[:3] = 0.02
        noise_scale[3:6] = 0.03
        noise_scale[half_dof : half_dof + 3] = 0.02
        noise_scale[half_dof + 3 : half_dof + 6] = 0.03

    if ctrl_mode in ["retarget", "ik"]:

        def get_ctrl_list():
            return default_ctrl_list.copy()

    elif ctrl_mode == "noise":

        def get_ctrl_list():
            num_noise = int(total_time / ctrl_dt)
            noise_list = np.random.randn(num_noise, model.nu) * noise_scale
            noise_list_repeat = noise_list[:, None, :].repeat(
                int(ctrl_dt / sim_dt), axis=1
            )
            noise_list = noise_list_repeat.reshape(-1, noise_list.shape[1])
            # append last noise to make sure the total time is correct
            for _ in range(default_ctrl_list.shape[0] - noise_list.shape[0]):
                noise_list = np.concatenate([noise_list, noise_list[-1:]], axis=0)
            return default_ctrl_list + noise_list

    elif ctrl_mode == "rect":

        def get_ctrl_list():
            num_rect = int(total_time / ctrl_dt)
            rect_list = np.zeros((num_rect, model.nu))
            for i in range(num_rect):
                rect_list[i, :] = 2.0 * (i % 2) - 1.0
            rect_list *= noise_scale
            rect_list_repeat = rect_list[:, None, :].repeat(
                int(ctrl_dt / sim_dt), axis=1
            )
            rect_list = rect_list_repeat.reshape(-1, rect_list.shape[1])
            for _ in range(default_ctrl_list.shape[0] - rect_list.shape[0]):
                rect_list = np.concatenate([rect_list, rect_list[-1:]], axis=0)
            return default_ctrl_list + rect_list

    elif ctrl_mode == "debug":
        log_info = np.load("../../outputs/info.npz")
        ctrls = log_info["ctrl"]
        ctrls_flat = ctrls.reshape(-1, model.nu)

        # append last ctrl to make sure the total time is correct
        for _ in range(default_ctrl_list.shape[0] - ctrls_flat.shape[0]):
            ctrls_flat = np.concatenate([ctrls_flat, ctrls_flat[-1:]], axis=0)

        def get_ctrl_list():
            return ctrls_flat

    else:
        raise ValueError(f"Invalid ctrl_mode: {ctrl_mode}")

    if motion_length == -1:
        motion_length = qpos_list.shape[0]
    qpos_list = qpos_list[:motion_length]
    qvel_list = qvel_list[:motion_length]
    ctrl_list = ctrl_list[:motion_length]

    # Prepare viewer context (real or dummy)
    if show_viewer:
        run_viewer = lambda: mujoco.viewer.launch_passive(model, data)
    else:
        cam = mujoco.MjvCamera()
        cam.type = 2
        cam.fixedcamid = 0

        @contextmanager
        def run_viewer():
            yield type(
                "DummyViewer",
                (),
                {
                    "is_running": lambda: True,
                    "sync": lambda: None,
                    "cam": cam,
                },
            )

    with run_viewer() as gui:
        cnt = 0
        ctrl_list = get_ctrl_list()[:motion_length]
        while gui.is_running():
            data.ctrl[:] = ctrl_list[cnt]
            # get site positions
            hand_sites = []
            hand_sites_id = []
            obj_sites = []
            obj_sites_id = []
            sides = {
                "right": ["right"],
                "left": ["left"],
                "bimanual": ["right", "left"],
            }.get(embodiment_type, [])
            tips = {
                "allegro": ["thumb", "index", "middle", "ring"],
                "metahand": ["thumb", "index", "middle", "ring"],
            }.get(
                robot_type,
                ["thumb", "index", "middle", "ring", "pinky"],
            )
            for side in sides:
                for tip_name in tips:
                    hand_site = f"{side}_{tip_name}_tip"
                    hand_sites.append(hand_site)
                    hand_sites_id.append(
                        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, hand_site)
                    )
                    obj_site = f"{side}_object_{tip_name}_tip_contact_site"
                    obj_sites.append(obj_site)
                    obj_sites_id.append(
                        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, obj_site)
                    )
            # update reference
            ref_data.qpos[:] = qpos_list[cnt]
            mujoco.mj_forward(model, ref_data)
            if enable_xfrc:
                # get object id
                right_obj_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, "right_object"
                )
                left_obj_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, "left_object"
                )
                # apply force to object
                kp_pos = 100.0
                kd_pos = 3.0
                kp_rot = 0.0
                kd_rot = 0.0
                if right_obj_id != -1:
                    right_obj_pos_ref = ref_data.xpos[right_obj_id]
                    right_obj_quat_ref = ref_data.xquat[right_obj_id]
                    right_obj_pos = data.xpos[right_obj_id]
                    right_obj_quat = data.xquat[right_obj_id]
                    right_obj_vel = data.cvel[right_obj_id]
                    pos_err = right_obj_pos_ref - right_obj_pos
                    rot_err = np.zeros(3)
                    mass = model.body_mass[right_obj_id]
                    inertia = model.body_inertia[right_obj_id]
                    mujoco.mju_subQuat(rot_err, right_obj_quat_ref, right_obj_quat)
                    data.xfrc_applied[right_obj_id, :3] = mass * (
                        kp_pos * pos_err - kd_pos * right_obj_vel[3:]
                    )
                    data.xfrc_applied[right_obj_id, 3:] = inertia * (
                        kp_rot * rot_err - kd_rot * right_obj_vel[:3]
                    )
                if left_obj_id != -1:
                    left_obj_pos_ref = ref_data.xpos[left_obj_id]
                    left_obj_quat_ref = ref_data.xquat[left_obj_id]
                    left_obj_pos = data.xpos[left_obj_id]
                    left_obj_quat = data.xquat[left_obj_id]
                    left_obj_vel = data.cvel[left_obj_id]
                    pos_err = left_obj_pos_ref - left_obj_pos
                    rot_err = np.zeros(3)
                    mass = model.body_mass[left_obj_id]
                    inertia = model.body_inertia[left_obj_id]
                    mujoco.mju_subQuat(rot_err, left_obj_quat_ref, left_obj_quat)
                    data.xfrc_applied[left_obj_id, :3] = mass * (
                        kp_pos * pos_err - kd_pos * left_obj_vel[3:]
                    )
                    data.xfrc_applied[left_obj_id, 3:] = inertia * (
                        kp_rot * rot_err - kd_rot * left_obj_vel[:3]
                    )
            mujoco.mj_step(model, data)
            info = {
                "qpos": data.qpos.copy(),
                "qvel": data.qvel.copy(),
                "ctrl": data.ctrl.copy(),
                "time": data.time,
            }
            if save_info:
                info_list.append(info)
            if show_viewer:
                mujoco.mjv_updateScene(
                    model,
                    ref_data,
                    vopt,
                    pert,
                    gui.cam,
                    catmask,
                    gui.user_scn,
                )
                gui.sync()
            cnt = (cnt + 1) % ctrl_list.shape[0]
            if cnt == ctrl_list.shape[0] - 1:
                if save_info or save_video:
                    break
                else:
                    ctrl_list = get_ctrl_list()[:motion_length]
                    data.qpos[:] = qpos_list[0]
                    data.qvel[:] = qvel_list[0]
                    data.ctrl[:] = ctrl_list[0]
                    mujoco.mj_forward(model, data)
                    cnt = 0
            if save_video:
                renderer.update_scene(data, "front")
                images.append(renderer.render())
            rate_limiter.sleep()
    info_aggregated = {}
    if save_info:
        for key in info_list[0].keys():
            info_aggregated[key] = np.stack([info[key] for info in info_list], axis=0)
        np.savez(
            f"../../outputs/mjcpu_{robot_type}_{embodiment_type}_{task}.npz",
            **info_aggregated,
        )
        print(f"Saved {robot_type}_{embodiment_type}_{task}.npz")
    if save_video:
        video_dir = f"../../recordings/{robot_type}/{embodiment_type}/mjcpu/{task}"
        os.makedirs(video_dir, exist_ok=True)
        imageio.mimsave(
            f"{video_dir}/mjcpu_{robot_type}_{embodiment_type}_{task}.mp4",
            images,
            fps=int(1 / sim_dt),
        )
        loguru.logger.info(
            f"Saved video to {video_dir}/mjcpu_{robot_type}_{embodiment_type}_{task}.mp4"
        )


if __name__ == "__main__":
    tyro.cli(main)
