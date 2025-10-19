"""Retargeting for humand-object interaction with HDMI simulator.

Author: Chaoyi Pan
Date: 2025-10-18
"""

from __future__ import annotations

import time

import hydra
import imageio
import loguru
import mujoco
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
    get_reference,
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
from spider.viewers import (
    build_and_log_scene_from_spec,
    render_image,
    setup_renderer,
    setup_viewer,
    update_viewer,
)


def main(config: Config):
    """Run the SPIDER using HDMI backend."""
    # Setup env (ref_data set to None since environment has built-in reference)
    env = setup_env(config, None)
    if config.max_sim_steps == -1:
        config.max_sim_steps = env.max_episode_length
        loguru.logger.info(f"Max simulation steps set to {config.max_sim_steps}")

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
    for _ in range(config.max_num_iterations):
        env_params = [{}] * config.num_dr
        env_params_list.append(env_params)
    config.env_params_list = env_params_list

    # Get reference data (states and controls)
    qpos_ref, qvel_ref, ctrl_ref = get_reference(config, env)

    # Setup mujoco model and data from HDMI env (for rendering)
    mj_model = env.sim.mj_model
    mj_data = mujoco.MjData(mj_model)
    mj_data_ref = mujoco.MjData(mj_model)

    # Initialize mj_data with current env state
    sim_data = env.sim.data
    mj_data.qpos[:] = sim_data.qpos[0].detach().cpu().numpy()
    mj_data.qvel[:] = sim_data.qvel[0].detach().cpu().numpy()
    mujoco.mj_step(mj_model, mj_data)
    mj_data.time = 0.0

    # Initialize reference mj_data
    mj_data_ref.qpos[:] = qpos_ref[0].detach().cpu().numpy()
    mj_data_ref.qvel[:] = qvel_ref[0].detach().cpu().numpy()
    mujoco.mj_step(mj_model, mj_data_ref)

    # Setup for video rendering
    images = []

    # Setup viewer and renderer
    # Note: mjlab viewer is disabled via cfg.viewer.headless in setup_env
    # For HDMI with rerun, build scene directly from spec and model
    if "rerun" in config.viewer:
        # Build and log 3D scene from HDMI's spec and model
        loguru.logger.info("Building Rerun scene from HDMI spec and model...")
        config.viewer_body_entity_and_ids = build_and_log_scene_from_spec(
            spec=env.scene.spec,
            model=mj_model,
            xml_path=None,  # No XML file, HDMI creates scene programmatically
            entity_root="mujoco",
        )
        loguru.logger.info(
            f"Rerun scene built with {len(config.viewer_body_entity_and_ids)} body entities"
        )
        # Set model_path to dummy value to indicate scene is already built
        config.model_path = "hdmi_scene_from_spec"

    run_viewer = setup_viewer(config, mj_model, mj_data)
    renderer = setup_renderer(config, mj_model)

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
            step_info = {"qpos": [], "qvel": [], "time": [], "ctrl": [], "ctrl_ref": []}
            for i in range(config.ctrl_steps):
                ctrl = ctrls[i]
                ctrl_repeat = ctrl.unsqueeze(0).repeat(
                    int(config.num_samples), 1
                )  # (batch_size, num_actions)
                step_env(config, env, ctrl_repeat)

                # Update mj_data with current state
                sim_data = env.sim.data
                mj_data.qpos[:] = sim_data.qpos[0].detach().cpu().numpy()
                mj_data.qvel[:] = sim_data.qvel[0].detach().cpu().numpy()
                mj_data.time = (sim_step + 1) * config.sim_dt

                # Render video if enabled
                should_render = (
                    config.save_video
                    and renderer is not None
                    and i % int(np.round(config.render_dt / config.sim_dt)) == 0
                )
                if should_render:
                    # Get reference state from reference data
                    ref_idx = min(sim_step, qpos_ref.shape[0] - 1)
                    mj_data_ref.qpos[:] = qpos_ref[ref_idx].detach().cpu().numpy()
                    mj_data_ref.qvel[:] = qvel_ref[ref_idx].detach().cpu().numpy()
                    mujoco.mj_step(mj_model, mj_data_ref)
                    image = render_image(
                        config, renderer, mj_model, mj_data, mj_data_ref
                    )
                    images.append(image)

                # Record state info
                step_info["qpos"].append(mj_data.qpos.copy())
                step_info["qvel"].append(mj_data.qvel.copy())
                step_info["time"].append(mj_data.time)
                step_info["ctrl"].append(ctrl.detach().cpu().numpy())
                step_info["ctrl_ref"].append(ctrl_ref[sim_step].detach().cpu().numpy())

                sim_step += 1

            # Stack step info
            for k in step_info:
                step_info[k] = np.stack(step_info[k], axis=0)
            infos.update(step_info)

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
            sim_data = env.sim.data
            mj_data.qpos[:] = sim_data.qpos[0].detach().cpu().numpy()
            mj_data.qvel[:] = sim_data.qvel[0].detach().cpu().numpy()
            # Update reference state
            ref_idx = min(sim_step, qpos_ref.shape[0] - 1)
            mj_data_ref.qpos[:] = qpos_ref[ref_idx].detach().cpu().numpy()
            mj_data_ref.qvel[:] = qvel_ref[ref_idx].detach().cpu().numpy()
            mujoco.mj_step(mj_model, mj_data_ref)
            update_viewer(config, viewer, mj_model, mj_data, mj_data_ref, infos)

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
        for k in info_list[0]:
            info_aggregated[k] = np.stack([info[k] for info in info_list], axis=0)
        np.savez(f"{config.output_dir}/trajectory_hdmi.npz", **info_aggregated)
        loguru.logger.info(f"Saved info to {config.output_dir}/trajectory_hdmi.npz")

    # Save video
    if config.save_video and len(images) > 0:
        video_path = f"{config.output_dir}/visualization_hdmi.mp4"
        imageio.mimsave(
            video_path,
            images,
            fps=int(1 / config.render_dt),
        )
        loguru.logger.info(f"Saved video to {video_path}")

    return


@hydra.main(version_base=None, config_path="config", config_name="hdmi")
def run_main(cfg: DictConfig) -> None:
    """Main entry point for HDMI retargeting."""
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
