"""Visualize MJWarp Simulation

Author: Chaoyi Pan
Date: 2025-06-30
"""

import mujoco
import mujoco.viewer
import tyro
import warp as wp
from loop_rate_limiters import RateLimiter

from spider.config import Config, process_config
from spider.io import load_data
from spider.simulators.mjwp import (
    get_qpos,
    load_state,
    save_state,
    setup_env,
    setup_mj_model,
    step_env,
)
from spider.viewers import setup_viewer

try:
    # Initialize Warp
    wp.init()
except RuntimeError:
    # Already initialized
    pass


def main(
    dataset_dir: str = "../../example_datasets",
    dataset_name: str = "oakink",
    robot_type: str = "allegro",
    embodiment_type: str = "bimanual",
    task: str = "pick_spoon_bowl",
    data_id: int = 0,
    ref_dt: float = 0.02,
):
    config = Config(
        robot_type=robot_type,
        task=task,
        dataset_name=dataset_name,
        embodiment_type=embodiment_type,
        data_id=data_id,
        ref_dt=ref_dt,
        num_samples=1,
        max_num_iterations=1,
        num_dyn=1,
        eq_solimp_min=[0.0, 1.0, 0.1, 0.5, 2.0],
        sim_dt=0.01,
        njmax_per_env=100,
        nconmax_per_env=512,
    )
    config = process_config(config)

    qpos_ref, qvel_ref, ctrl_ref, contact_ref, contact_pos_ref = load_data(
        config, config.data_path
    )
    contact_ref[:] = 1.0
    ref_data = (qpos_ref, qvel_ref, ctrl_ref, contact_ref, contact_pos_ref)
    env = setup_env(config, ref_data)

    # setup mujoco (for viewer only)
    mj_model = setup_mj_model(config)
    mj_data = mujoco.MjData(mj_model)
    mj_data_ref = mujoco.MjData(mj_model)
    mj_data.qpos[:] = qpos_ref[0].detach().cpu().numpy()
    mj_data.qvel[:] = 0.0
    mj_data.ctrl[:] = qpos_ref[0][: -config.nq_obj].detach().cpu().numpy()
    mujoco.mj_step(mj_model, mj_data)
    mj_data.time = 0.0
    images = []
    config.trace_site_ids = []
    config.track_site_ids = []
    config.ref_site_ids = []
    config.ref_mocap_ids = []
    object_trace_site_ids = []
    hand_trace_site_ids = []
    for sid in range(mj_model.nsite):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SITE, sid)
        if name is not None:
            if name.startswith("trace"):
                if "object" in name:
                    object_trace_site_ids.append(sid)
                else:
                    hand_trace_site_ids.append(sid)
            elif name.startswith("track"):
                config.track_site_ids.append(sid)
    config.trace_site_ids = object_trace_site_ids + hand_trace_site_ids
    for sid in config.track_site_ids:
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SITE, sid)
        # replace "track_hand_" or "track_object_" with "ref_"
        name = name.replace("track", "ref")
        ref_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, name)
        config.ref_site_ids.append(ref_id)
        # get mocap id
        mocap_name = (
            name.replace("hand", "object")
            if "hand" in name
            else name.replace("object", "hand")
        )
        mocap_body_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, mocap_name
        )
        mocap_id = mj_model.body_mocapid[mocap_body_id]
        config.ref_mocap_ids.append(mocap_id)

    run_viewer = setup_viewer(config, mj_model, mj_data)

    rate_limiter = RateLimiter(1.0 / config.sim_dt)
    init_state = save_state(env)
    env.active_annealing_group = 0
    env.active_dr_group = 0
    with run_viewer() as viewer:
        cnt = 0
        while viewer.is_running():
            step_env(config, env, ctrl_ref[cnt].unsqueeze(0))
            # mj_data.ctrl[:] = ctrl_ref[cnt].detach().cpu().numpy()
            # mujoco.mj_step(mj_model, mj_data)
            # get qpos for render
            qpos = get_qpos(config, env)[0].detach().cpu().numpy()
            mocap_pos = wp.to_torch(env.data_wp.mocap_pos)[0].detach().cpu().numpy()
            mj_data.qpos[:] = qpos
            mj_data.mocap_pos[:] = mocap_pos
            mujoco.mj_kinematics(mj_model, mj_data)
            viewer.sync()
            rate_limiter.sleep()
            cnt += 1
            if cnt >= ctrl_ref.shape[0] - 1:
                env = load_state(env, init_state)
                cnt = 0


if __name__ == "__main__":
    tyro.cli(main)
