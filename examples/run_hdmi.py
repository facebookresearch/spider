"""Retargeting for humand-object interaction with HDMI simulator.

Author: Chaoyi Pan
Date: 2025-10-18
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
from spider.simulators.hdmi import (
    get_initial_ctrls,
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
    """Run the SPIDER using HDMI backend"""
    # Setup env (ref_data set to None since environment has built-in reference)
    env = setup_env(config, None)
    config.nu = env.action_spec.shape[-1]

    # Process config, set defaults and derived fields
    config = process_config(config)

    # Create placeholder reference data for compatibility
    ref_data = (
        torch.zeros(
            config.max_sim_steps + config.horizon_steps + config.ctrl_steps,
            config.nu,
            device=config.device,
        ),
    )

    # Setup env params (empty for HDMI, no domain randomization)
    env_params_list = []
    for i in range(config.max_num_iterations):
        env_params = [{}] * config.num_dr
        env_params_list.append(env_params)
    config.env_params_list = env_params_list

    # Setup viewer
    run_viewer = setup_viewer(config, None, None)

    # Setup optimizer
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

    # Get full reference control sequence
    ctrl_ref = get_initial_ctrls(config, env)

    # Initial controls - first horizon_steps from reference
    ctrls = ctrl_ref[: config.horizon_steps]

    # Buffers for saving info and trajectory
    info_list = []

    # Run viewer + control loop
    t_start = time.perf_counter()
    sim_step = 0
    with run_viewer() as viewer:
        while viewer.is_running():
            t0 = time.perf_counter()

            # Optimize using future reference window at control-rate (+1 lookahead)
            ref_slice = get_slice(
                ref_data, sim_step + 1, sim_step + config.horizon_steps + 1
            )
            if config.max_num_iterations > 0:
                ctrls, infos = optimize(config, env, ctrls, ref_slice)
            else:
                infos = {"opt_steps": [0], "improvement": 0.0}
            infos["sim_step"] = sim_step

            # Step environment for ctrl_steps
            for i in range(config.ctrl_steps):
                ctrl = ctrls[i]
                ctrl_repeat = ctrl.unsqueeze(0).repeat(
                    int(config.num_samples), 1
                )  # (batch_size, num_actions)
                step_env(config, env, ctrl_repeat)
                sim_step += 1

            # Sync env state (broadcast from first env to all)
            env = sync_env(config, env)

            # Receding horizon update
            prev_ctrl = ctrls[config.ctrl_steps :]
            new_ctrl = ctrl_ref[
                sim_step + prev_ctrl.shape[0] : sim_step
                + prev_ctrl.shape[0]
                + config.ctrl_steps
            ]
            ctrls = torch.cat([prev_ctrl, new_ctrl], dim=0)

            # Sync viewer state and render
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

            # Progress
            t1 = time.perf_counter()
            rtr = config.ctrl_dt / (t1 - t0)
            print(
                f"Realtime rate: {rtr:.2f}, plan time: {t1 - t0:.4f}s, sim_steps: {sim_step}/{config.max_sim_steps}, opt_steps: {infos['opt_steps'][0]}",
                end="\r",
            )

            # Record info/trajectory at control tick
            info_list.append({k: v for k, v in infos.items() if k != "trace_sample"})

            if sim_step >= config.max_sim_steps:
                break

        t_end = time.perf_counter()
        print(f"\nTotal time: {t_end - t_start:.4f}s")

    # Save retargeted trajectory
    if config.save_info and len(info_list) > 0:
        info_aggregated = {}
        for k in info_list[0].keys():
            info_aggregated[k] = np.stack([info[k] for info in info_list], axis=0)
        np.savez(f"{config.output_dir}/trajectory_hdmi.npz", **info_aggregated)
        loguru.logger.info(f"Saved info to {config.output_dir}/trajectory_hdmi.npz")

    return


@hydra.main(version_base=None, config_path="config", config_name="hdmi")
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
