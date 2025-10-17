"""A standalone script to run DIAL MPC with Genesis (DexMachina)

Author: Chaoyi Pan
Date: 2025-10-16
"""

from __future__ import annotations

import time

import hydra
import loguru
import numpy as np
import torch
from omegaconf import DictConfig

from spider.config import Config, process_config
from spider.interp import get_slice
from spider.optimizers.sampling import (
    make_optimize_fn,
    make_optimize_once_fn,
    make_rollout_fn,
)
from spider.simulators.dexmachina import (
    get_obj_arti_dist,
    get_obj_pos_dist,
    get_obj_quat_dist,
    get_qpos,
    get_reward,
    get_terminal_reward,
    get_trace,
    load_env_params,
    load_state,
    save_env_params,
    save_state,
    setup_env,
    step_env,
    sync_env,
)
from spider.viewers import setup_viewer, update_viewer


def main(config: Config):
    """Run the DIAL MPC using Genesis (DexMachina) backend"""
    # Setup env (ref_data set to None since environment has built-in reference)
    env = setup_env(config, None)
    config.nu = env.num_actions
    # process config, set defaults and derived fields
    config = process_config(config)
    if config.max_sim_steps == -1:
        config.max_sim_steps = env.max_episode_length
    else:
        config.max_sim_steps = min(env.max_episode_length, config.max_sim_steps)

    # Create placeholder reference data for compatibility
    ref_data = (
        torch.zeros(
            config.max_sim_steps + config.horizon_steps + config.ctrl_steps,
            config.nu,
            device=config.device,
        ),
    )

    # Setup env params (curriculum for object joint stiffness/damping)
    env_params_list = []
    kp_max = 0.0
    kv_max = 0.0
    eta = 0.001 ** (1.0 / (config.max_num_iterations - 1))
    kp_list = kp_max * (eta ** np.arange(config.max_num_iterations))
    kp_list[-1] = 0.0
    kv_list = kv_max * (eta ** np.arange(config.max_num_iterations))
    kv_list[-1] = 0.0
    kp_env = env.objects[env.object_names[0]].entity.get_dofs_kp()
    kv_env = env.objects[env.object_names[0]].entity.get_dofs_kv()
    for i in range(config.max_num_iterations):
        env_params = []
        for j in range(config.num_dr):
            kp = torch.ones_like(kp_env, device=config.device) * kp_list[i]
            kv = torch.ones_like(kv_env, device=config.device) * kv_list[i]
            env_params.append({"kp": kp, "kv": kv})
        env_params_list.append(env_params)
    config.env_params_list = env_params_list

    # setup viewer
    run_viewer = setup_viewer(config, None, None)

    # setup optimizer
    rollout = make_rollout_fn(
        step_env,
        save_state,
        load_state,
        get_reward,
        get_terminal_reward,
        get_trace,
        save_env_params,
        load_env_params,
    )
    optimize_once = make_optimize_once_fn(rollout)
    optimize = make_optimize_fn(optimize_once)

    # initial controls
    ctrls = torch.zeros(
        config.horizon_steps,
        config.nu,
        device=config.device,
    )

    # buffers for saving info and trajectory
    info_list = []

    # run viewer + control loop
    t_start = time.perf_counter()
    sim_step = 0
    with run_viewer() as viewer:
        while viewer.is_running():
            t0 = time.perf_counter()

            # optimize using future reference window at control-rate (+1 lookahead)
            ref_slice = get_slice(
                ref_data, sim_step + 1, sim_step + config.horizon_steps + 1
            )
            ctrls, infos = optimize(config, env, ctrls, ref_slice)
            infos["sim_step"] = sim_step

            # step environment for ctrl_steps
            step_info = {
                "obj_pos_dist": [],
                "obj_quat_dist": [],
                "obj_arti_dist": [],
                "qpos": [],
            }
            for i in range(config.ctrl_steps):
                ctrl = ctrls[i]
                ctrl_repeat = ctrl.unsqueeze(0).repeat(
                    int(config.num_samples), 1
                )  # (batch_size, num_actions)
                step_env(config, env, ctrl_repeat)
                step_info["obj_pos_dist"].append(get_obj_pos_dist(env)[0].item())
                step_info["obj_quat_dist"].append(get_obj_quat_dist(env)[0].item())
                step_info["obj_arti_dist"].append(get_obj_arti_dist(env)[0].item())
                step_info["qpos"].append(get_qpos(env)[0].detach().cpu().numpy())
                sim_step += 1
                if config.save_video:
                    env._recording = True
                    env._render_headless()
                    env._recording = False
            for k in step_info:
                step_info[k] = np.stack(step_info[k], axis=0)
            infos.update(step_info)

            # sync env state
            env = sync_env(config, env)

            # receding horizon update
            prev_ctrl = ctrls[config.ctrl_steps :]
            new_ctrl = torch.zeros(
                (config.ctrl_steps, config.nu),
                device=config.device,
            )
            ctrls = torch.cat([prev_ctrl, new_ctrl], dim=0)

            # sync viewer state and render
            # Create dummy mj_data with time for viewer
            dummy_mj_data = type(
                "DummyMjData", (), {"time": sim_step * config.sim_dt}
            )()
            update_viewer(
                config,
                viewer,
                mj_model=None,
                mj_data=dummy_mj_data,
                mj_data_ref=None,
                info=infos,
            )

            # progress
            t1 = time.perf_counter()
            rtr = config.ctrl_dt / (t1 - t0)
            print(
                f"Realtime rate: {rtr:.2f}, plan time: {t1 - t0:.4f}s, sim_steps: {sim_step}/{config.max_sim_steps}, opt_steps: {infos['opt_steps'][0]}",
                end="\r",
            )

            # record info/trajectory at control tick
            # rule out "trace"
            info_list.append({k: v for k, v in infos.items() if k != "trace_sample"})

            if sim_step >= config.max_sim_steps:
                break

        t_end = time.perf_counter()
        print(f"\nTotal time: {t_end - t_start:.4f}s")

    # print object dist in mean and std
    obj_pos_dists = np.concatenate([info["obj_pos_dist"] for info in info_list])
    obj_quat_dists = np.concatenate([info["obj_quat_dist"] for info in info_list])
    obj_arti_dists = np.concatenate([info["obj_arti_dist"] for info in info_list])
    print(f"obj_pos_dist: {obj_pos_dists.mean():.4f} ± {obj_pos_dists.std():.4f}")
    print(f"obj_quat_dist: {obj_quat_dists.mean():.4f} ± {obj_quat_dists.std():.4f}")
    print(f"obj_arti_dist: {obj_arti_dists.mean():.4f} ± {obj_arti_dists.std():.4f}")

    # save retargeted trajectory
    if config.save_info and len(info_list) > 0:
        info_aggregated = {}
        for k in info_list[0].keys():
            info_aggregated[k] = np.stack([info[k] for info in info_list], axis=0)
        np.savez(f"{config.output_dir}/trajectory_dexmachina.npz", **info_aggregated)
        loguru.logger.info(
            f"Saved info to {config.output_dir}/trajectory_dexmachina.npz"
        )

    # save video
    if config.save_video and len(env._recorded_frames) > 0:
        env.export_video(
            f"{config.output_dir}/trajectory_dexmachina.mp4", wait_for_max=False
        )
        loguru.logger.info(
            f"Saved video to {config.output_dir}/trajectory_dexmachina.mp4"
        )

    return


@hydra.main(version_base=None, config_path="config", config_name="dexmachina")
def run_main(cfg: DictConfig) -> None:
    # Convert DictConfig to Config dataclass, handling special fields
    config_dict = dict(cfg)

    # Handle special conversions
    if "noise_scale" in config_dict and config_dict["noise_scale"] is None:
        config_dict.pop("noise_scale")  # Let the default factory handle it

    # Convert lists to tuples where needed
    if "pair_margin_range" in config_dict:
        config_dict["pair_margin_range"] = tuple(config_dict["pair_margin_range"])
    if "xy_offset_range" in config_dict:
        config_dict["xy_offset_range"] = tuple(config_dict["xy_offset_range"])

    config = Config(**config_dict)
    main(config)


if __name__ == "__main__":
    run_main()
