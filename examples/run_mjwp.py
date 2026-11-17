# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A standalone script to run DIAL MPC with Mujoco + Warp

Up to now, domain randomization is not supported. Will add it later.

Author: Chaoyi Pan
Date: 2025-08-11
"""

from __future__ import annotations

import sys
import time
from dataclasses import fields
from pathlib import Path

import hydra
import imageio
import loguru
import mujoco
import numpy as np
import torch
import warp as wp
from omegaconf import DictConfig, OmegaConf

from spider.config import (
    Config,
    filter_config_fields,
    load_config_yaml,
    process_config,
)
from spider.interp import get_slice
from spider.io import load_data
from spider.optimizers.sampling import (
    make_optimize_fn,
    make_optimize_once_fn,
    make_rollout_fn,
)
from spider.postprocess.get_success_rate import compute_object_tracking_error
from spider.simulators.mjwp import (
    _initial_state_sanity_check,
    compute_contact_point_delta,
    copy_sample_state,
    get_qpos,
    get_qvel,
    get_reward,
    get_terminal_reward,
    get_terminate,
    get_trace,
    load_env_params,
    load_state,
    save_env_params,
    save_state,
    setup_env,
    setup_mj_model,  # mjwp specific
    step_env,
    sync_env,
)
from spider.viewers import (
    log_frame,
    render_image,
    setup_renderer,
    setup_viewer,
    update_viewer,
)

_CONFIG_SKIP_FIELDS = {
    "noise_scale",
    "env_params_list",
    "viewer_body_entity_and_ids",
}


def _parse_override_tokens(tokens: list[str]) -> dict:
    allowed = {field.name for field in fields(Config)}
    override_dict: dict = {}
    for item in tokens:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.lstrip("+")
        if key not in allowed:
            continue
        parsed = OmegaConf.to_container(
            OmegaConf.from_dotlist([f"{key}={value}"]), resolve=True
        )
        if isinstance(parsed, dict) and key in parsed:
            override_dict[key] = parsed[key]
    return override_dict


def _extract_cli_overrides(cfg: DictConfig) -> dict:
    """Extract CLI overrides so they can be applied on top of a loaded config."""
    overrides = OmegaConf.select(cfg, "hydra.overrides.task") or []
    override_dict = _parse_override_tokens(overrides)
    if override_dict:
        return override_dict
    return _parse_override_tokens(sys.argv[1:])


def _assert_object_actuator_gains_zero(
    env, config: Config, stage: str, atol: float = 1e-4
) -> None:
    if not config.contact_guidance or not config.object_actuator_ids:
        return
    actuator_ids = np.asarray(config.object_actuator_ids, dtype=int)
    if not hasattr(env, "model_wp") or not hasattr(env.model_wp, "actuator_gainprm"):
        raise AssertionError("MJWarp model does not expose actuator_gainprm.")
    gainprm = wp.to_torch(env.model_wp.actuator_gainprm).detach().cpu().numpy()
    biasprm = wp.to_torch(env.model_wp.actuator_biasprm).detach().cpu().numpy()
    if gainprm.ndim == 3:
        gainprm = gainprm[0]
    if biasprm.ndim == 3:
        biasprm = biasprm[0]
    kp = gainprm[actuator_ids, 0]
    kd = -biasprm[actuator_ids, 1]
    assert np.allclose(kp, 0.0, atol=atol), (
        f"Object actuator Kp not near zero at {stage}: max={np.max(np.abs(kp))}"
    )
    assert np.allclose(kd, 0.0, atol=atol), (
        f"Object actuator Kd not near zero at {stage}: max={np.max(np.abs(kd))}"
    )


def _normalize_yaml_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    return value


def _save_config_yaml(config: Config) -> None:
    if not config.save_config:
        return
    config_dict = {}
    for field in fields(config):
        if field.name in _CONFIG_SKIP_FIELDS:
            continue
        config_dict[field.name] = _normalize_yaml_value(getattr(config, field.name))
    output_path = (
        Path(config.output_dir)
        / f"config{'_act' if config.contact_guidance else ''}.yaml"
    )
    OmegaConf.save(config=OmegaConf.create(config_dict), f=str(output_path))
    loguru.logger.info(f"Saved config to {output_path}")


def _get_bimanual_hand_indices(config: Config) -> tuple[list[int], list[int]]:
    robot_nu = int(config.nu)
    if config.contact_guidance:
        obj_dims = (
            int(config.object_action_dims) if config.object_action_dims > 0 else 12
        )
        robot_nu = max(robot_nu - obj_dims, 0)
    half = robot_nu // 2
    right_ids = list(range(0, half))
    left_ids = list(range(half, robot_nu))
    return right_ids, left_ids


def _apply_noise_mask(
    base_noise_scale: torch.Tensor, zero_indices: list[int]
) -> torch.Tensor:
    noise_scale = base_noise_scale.clone()
    if zero_indices:
        idx = torch.as_tensor(
            zero_indices, device=base_noise_scale.device, dtype=torch.long
        )
        noise_scale[:, :, idx] *= 0.0
    return noise_scale


def main(config: Config):
    """Run the SPIDER using MuJoCo Warp backend"""
    # process config, set defaults and derived fields
    config = process_config(config)
    if config.contact_guidance and config.improvement_threshold > 0.0:
        loguru.logger.warning(
            "contact_guidance requires improvement_threshold <= 0; overriding to 0.0."
        )
        config.improvement_threshold = 0.0

    # load reference data (already interpolated and extended)
    qpos_ref, qvel_ref, ctrl_ref, contact, contact_pos = load_data(
        config, config.data_path
    )
    if (
        config.contact_guidance
        and ctrl_ref.shape[1] != config.nu
        and qpos_ref.shape[1] >= config.nu
    ):
        loguru.logger.info(
            "Using qpos as ctrl reference for contact guidance (ctrl dims: {} -> {}).",
            ctrl_ref.shape[1],
            config.nu,
        )
        ctrl_ref = qpos_ref[:, : config.nu]
    if config.contact_guidance and torch.all(contact <= 0):
        raise ValueError("contact_guidance is enabled, but contact mask is all zeros.")
    # hack: start from step 500
    # qpos_ref = qpos_ref[500:]
    # qvel_ref = qvel_ref[500:]
    # ctrl_ref = ctrl_ref[500:]
    # contact = contact[500:]
    # contact_pos = contact_pos[500:]
    ref_data = (qpos_ref, qvel_ref, ctrl_ref, contact, contact_pos)
    config.max_sim_steps = (
        config.max_sim_steps
        if config.max_sim_steps > 0
        else qpos_ref.shape[0] - config.horizon_steps - config.ctrl_steps
    )

    # setup env with initial state from first sim qpos
    env = setup_env(config, ref_data)

    # setup mujoco (for viewer only)
    mj_model = setup_mj_model(config)
    mj_data = mujoco.MjData(mj_model)
    mj_data_ref = mujoco.MjData(mj_model)
    mj_data.qpos[:] = qpos_ref[0].detach().cpu().numpy()
    mj_data.qvel[:] = qvel_ref[0].detach().cpu().numpy()
    mj_data.ctrl[:] = ctrl_ref[0].detach().cpu().numpy()
    mujoco.mj_step(mj_model, mj_data)
    mj_data.time = 0.0
    _assert_object_actuator_gains_zero(env, config, "start")

    # Initial-state stability sanity check (CPU MuJoCo only). Disabled when
    # config.sanity_check_seconds <= 0.0.
    if config.sanity_check_seconds > 0.0:
        _initial_state_sanity_check(
            mj_model,
            mj_data,
            qpos_ref,
            qvel_ref,
            ctrl_ref,
            config,
            save_video_dir=config.output_dir,
        )
        # Restore initial scene state after the check.
        mj_data.qpos[:] = qpos_ref[0].detach().cpu().numpy()
        mj_data.qvel[:] = qvel_ref[0].detach().cpu().numpy()
        mj_data.ctrl[:] = ctrl_ref[0].detach().cpu().numpy()
        mujoco.mj_step(mj_model, mj_data)
        mj_data.time = 0.0

    images = []
    object_trace_site_ids = []
    robot_trace_site_ids = []
    for sid in range(mj_model.nsite):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SITE, sid)
        if name is not None:
            if name.startswith("trace"):
                if "object" in name:
                    object_trace_site_ids.append(sid)
                else:
                    robot_trace_site_ids.append(sid)
    config.trace_site_ids = object_trace_site_ids + robot_trace_site_ids
    contact_guidance_enabled = (
        config.contact_guidance and len(config.object_actuator_ids) > 0
    )
    if config.contact_guidance and not contact_guidance_enabled:
        loguru.logger.warning(
            "contact_guidance is enabled but no object actuators were resolved."
        )
    contact_offset = 0
    if contact_guidance_enabled:
        config.contact_len = int(
            min(contact.shape[1], contact_pos.shape[1], len(config.contact_order))
        )
        if (
            config.contact_len != len(config.contact_order)
            or config.contact_len != contact.shape[1]
        ):
            loguru.logger.warning(
                "Contact length mismatch (mask={}, pos={}, expected={}); truncating to {}.",
                contact.shape[1],
                contact_pos.shape[1],
                len(config.contact_order),
                config.contact_len,
            )
        config.contact_order = config.contact_order[: config.contact_len]
        config.hand_contact_site_ids = config.hand_contact_site_ids[
            : config.contact_len
        ]
        contact_offset = max(contact.shape[1] - config.contact_len, 0)

    # setup env params
    env_params_list = []
    if config.num_dr == 0:
        xy_offset_list = [0.0]
        pair_margin_list = [0.0]
    else:
        xy_offset_list = np.linspace(
            config.xy_offset_range[0], config.xy_offset_range[1], config.num_dr
        )
        pair_margin_list = np.linspace(
            config.pair_margin_range[0], config.pair_margin_range[1], config.num_dr
        )
    kp_schedule = []
    kd_schedule = []
    if contact_guidance_enabled and config.max_num_iterations > 0:
        actuator_names = config.object_actuator_names
        if not actuator_names:
            actuator_names = [
                mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(aid))
                for aid in config.object_actuator_ids
            ]
        base_kp = np.array(
            [
                (
                    config.init_rot_actuator_gain
                    if ("_rot_" in (name or ""))
                    else config.init_pos_actuator_gain
                )
                for name in actuator_names
            ],
            dtype=np.float32,
        )
        base_kd = np.array(
            [
                (
                    config.init_rot_actuator_bias
                    if ("_rot_" in (name or ""))
                    else config.init_pos_actuator_bias
                )
                for name in actuator_names
            ],
            dtype=np.float32,
        )
        for i in range(config.max_num_iterations):
            decay = float(config.guidance_decay_ratio) ** i
            kp_i = base_kp * decay
            kd_i = base_kd * decay
            if i == config.max_num_iterations - 1:
                kp_i = np.zeros_like(base_kp, dtype=np.float32)
                kd_i = np.zeros_like(base_kd, dtype=np.float32)
            kp_schedule.append(kp_i)
            kd_schedule.append(kd_i)

    for i in range(config.max_num_iterations):
        env_params = []
        for j in range(config.num_dr):
            params = {
                "xy_offset": xy_offset_list[j],
                "pair_margin": pair_margin_list[j],
            }
            if contact_guidance_enabled and kp_schedule:
                params["kp"] = kp_schedule[i]
                params["kd"] = kd_schedule[i]
            env_params.append(params)
        env_params_list.append(env_params)
    config.env_params_list = env_params_list
    _save_config_yaml(config)

    # setup viewer and renderer
    run_viewer = setup_viewer(config, mj_model, mj_data)
    renderer = setup_renderer(config, mj_model)

    # setup optimizer
    rollout = make_rollout_fn(
        step_env,
        save_state,
        load_state,
        get_reward,
        get_terminal_reward,
        get_terminate,
        get_trace,
        save_env_params,
        load_env_params,
        copy_sample_state,
    )
    optimize_once = make_optimize_once_fn(rollout)
    optimize = make_optimize_fn(optimize_once)
    base_noise_scale = config.noise_scale.clone()
    gibbs_enabled = config.gibbs_sampling and config.embodiment_type == "bimanual"
    if config.gibbs_sampling and not gibbs_enabled:
        loguru.logger.warning(
            "gibbs_sampling is enabled but embodiment_type is {}, disabling.",
            config.embodiment_type,
        )
    if gibbs_enabled:
        right_ids, left_ids = _get_bimanual_hand_indices(config)
        right_only_zero = left_ids
        left_only_zero = right_ids

    # initial controls
    ctrls = ctrl_ref[: config.horizon_steps]
    # buffers for saving info and trajectory
    info_list = []

    # run viewer + control loop
    t_start = time.perf_counter()
    with run_viewer() as viewer:
        while viewer.is_running():
            t0 = time.perf_counter()

            # optimize using future reference window at control-rate (+1 lookahead)
            sim_step = int(np.round(mj_data.time / config.sim_dt))
            ref_slice = get_slice(
                ref_data, sim_step + 1, sim_step + config.horizon_steps + 1
            )
            ctrls_for_opt = ctrls
            if contact_guidance_enabled and config.contact_len > 0:
                contact_mask_step = contact[sim_step][
                    contact_offset : contact_offset + config.contact_len
                ]
                contact_pos_ref_step = contact_pos[sim_step]
                site_xpos = wp.to_torch(env.data_wp.site_xpos)[0]

                right_delta = compute_contact_point_delta(
                    contact_mask_step,
                    contact_pos_ref_step,
                    site_xpos,
                    config.hand_contact_site_ids,
                    config.right_contact_indices,
                )
                left_delta = compute_contact_point_delta(
                    contact_mask_step,
                    contact_pos_ref_step,
                    site_xpos,
                    config.hand_contact_site_ids,
                    config.left_contact_indices,
                )
                if (
                    right_delta is not None
                    and config.right_pos_ctrl_ids
                    and sim_step + ctrls.shape[0] <= ctrl_ref.shape[0]
                ):
                    ctrls_for_opt = ctrls_for_opt.clone()
                    ref_ctrl_slice = ctrl_ref[sim_step : sim_step + ctrls.shape[0]]
                    ctrls_for_opt[:, config.right_pos_ctrl_ids] = ref_ctrl_slice[
                        :, config.right_pos_ctrl_ids
                    ] + torch.clip(right_delta, -0.01, 0.01)
                if (
                    left_delta is not None
                    and config.left_pos_ctrl_ids
                    and sim_step + ctrls.shape[0] <= ctrl_ref.shape[0]
                ):
                    if ctrls_for_opt is ctrls:
                        ctrls_for_opt = ctrls_for_opt.clone()
                        ref_ctrl_slice = ctrl_ref[sim_step : sim_step + ctrls.shape[0]]
                    ctrls_for_opt[:, config.left_pos_ctrl_ids] = ref_ctrl_slice[
                        :, config.left_pos_ctrl_ids
                    ] + torch.clip(left_delta, -0.01, 0.01)
            if gibbs_enabled:
                config.noise_scale = _apply_noise_mask(
                    base_noise_scale, right_only_zero
                )
                ctrls, infos = optimize(config, env, ctrls_for_opt, ref_slice)
                config.noise_scale = _apply_noise_mask(base_noise_scale, left_only_zero)
                ctrls, infos = optimize(config, env, ctrls, ref_slice)
                config.noise_scale = base_noise_scale
            else:
                config.noise_scale = base_noise_scale
                ctrls, infos = optimize(config, env, ctrls_for_opt, ref_slice)

            # Compute trace_ref from reference qpos over the horizon
            if len(config.trace_site_ids) > 0:
                trace_ref = []
                qpos_ref_horizon = ref_slice[0]
                for h in range(config.horizon_steps):
                    mj_data_ref.qpos[:] = qpos_ref_horizon[h].detach().cpu().numpy()
                    mujoco.mj_kinematics(mj_model, mj_data_ref)
                    site_xpos = np.array(
                        [mj_data_ref.site_xpos[sid] for sid in config.trace_site_ids]
                    )
                    trace_ref.append(site_xpos)
                # (H, K, 3) -> (1, 1, H, K, 3) to match trace_sample shape
                trace_ref_np = np.stack(trace_ref, axis=0)[None, None, :, :, :]
                infos["trace_ref"] = trace_ref_np

            # step environment for ctrl_steps
            step_info = {"qpos": [], "qvel": [], "time": [], "ctrl": []}
            for i in range(config.ctrl_steps):
                ctrl_step = ctrls[i]

                # option 1: use mujoco step
                # mj_data.ctrl[:] = ctrls[i].detach().cpu().numpy()
                # mujoco.mj_step(mj_model, mj_data)
                # option 2: use warp step
                step_env(config, env, ctrl_step)
                mj_data.qpos[:] = get_qpos(config, env)[0].detach().cpu().numpy()
                mj_data.qvel[:] = get_qvel(config, env)[0].detach().cpu().numpy()
                mj_data.ctrl[:] = ctrl_step.detach().cpu().numpy()
                mj_data.time += config.sim_dt
                if config.save_video and renderer is not None:
                    if i % int(np.round(config.render_dt / config.sim_dt)) == 0:
                        mj_data_ref.qpos[:] = (
                            qpos_ref[sim_step + i].detach().cpu().numpy()
                        )
                        image = render_image(
                            config, renderer, mj_model, mj_data, mj_data_ref
                        )
                        images.append(image)
                if "rerun" in config.viewer or "viser" in config.viewer:
                    mj_data_ref.qpos[:] = qpos_ref[sim_step + i].detach().cpu().numpy()
                    mujoco.mj_kinematics(mj_model, mj_data_ref)
                    log_frame(
                        mj_data,
                        sim_time=mj_data.time,
                        viewer_body_entity_and_ids=config.viewer_body_entity_and_ids,
                        data_ref=mj_data_ref,
                    )
                step_info["qpos"].append(mj_data.qpos.copy())
                step_info["qvel"].append(mj_data.qvel.copy())
                step_info["time"].append(mj_data.time)
                step_info["ctrl"].append(mj_data.ctrl.copy())
            for k in step_info:
                step_info[k] = np.stack(step_info[k], axis=0)
            infos.update(step_info)
            # sync env state
            sync_env(config, env, mj_data)

            # receding horizon update
            sim_step = int(np.round(mj_data.time / config.sim_dt))
            prev_ctrl = ctrls[config.ctrl_steps :]
            new_ctrl = ctrl_ref[
                sim_step + prev_ctrl.shape[0] : sim_step
                + prev_ctrl.shape[0]
                + config.ctrl_steps
            ]
            ctrls = torch.cat([prev_ctrl, new_ctrl], dim=0)

            # sync viewer state and render
            mj_data.qpos[:] = get_qpos(config, env)[0].detach().cpu().numpy()
            mj_data.qvel[:] = get_qvel(config, env)[0].detach().cpu().numpy()
            mj_data_ref.qpos[:] = qpos_ref[sim_step].detach().cpu().numpy()
            update_viewer(config, viewer, mj_model, mj_data, mj_data_ref, infos)

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
        print(f"Total time: {t_end - t_start:.4f}s")

    # save retargeted trajectory
    if config.save_info and len(info_list) > 0:
        info_aggregated = {}
        for k in info_list[0].keys():
            info_aggregated[k] = np.stack([info[k] for info in info_list], axis=0)
        np.savez(
            f"{config.output_dir}/trajectory_mjwp{'_act' if config.contact_guidance else ''}.npz",
            **info_aggregated,
        )
        loguru.logger.info(
            f"Saved info to {config.output_dir}/trajectory_mjwp{'_act' if config.contact_guidance else ''}.npz"
        )

    # save video
    if config.save_video and len(images) > 0:
        video_path = f"{config.output_dir}/visualization_mjwp{'_act' if config.contact_guidance else ''}.mp4"
        imageio.mimsave(
            video_path,
            images,
            fps=int(1 / config.render_dt),
        )
        loguru.logger.info(f"Saved video to {video_path}")

    errors = None
    if info_list:
        qpos_traj = np.concatenate([info["qpos"] for info in info_list], axis=0)
        qpos_ref_np = qpos_ref[: qpos_traj.shape[0]].detach().cpu().numpy()
        data_type = "mjwp_act" if config.contact_guidance else "mjwp"
        errors = compute_object_tracking_error(
            qpos_traj, qpos_ref_np, config.embodiment_type, data_type
        )
        loguru.logger.info(
            "Final object tracking error: pos={:.4f}, quat={:.4f}",
            errors["obj_pos_err"],
            errors["obj_quat_err"],
        )

    _assert_object_actuator_gains_zero(env, config, "end")

    if config.show_viewer and "viser" in config.viewer and config.wait_on_finish:
        loguru.logger.info(
            "Optimization complete! Keeping Viser server alive. Press Ctrl+C to exit."
        )
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    return errors


@hydra.main(version_base=None, config_path="config", config_name="default")
def run_main(cfg: DictConfig) -> None:
    """Entry point for Hydra configuration runner."""
    # Convert DictConfig to Config dataclass, handling special fields
    config_dict = dict(cfg)

    # Optionally load a saved config YAML and merge; CLI overrides take priority.
    load_config_path = config_dict.get("load_config_path", "")
    if load_config_path:
        loaded_config = load_config_yaml(load_config_path)
        cli_overrides = _extract_cli_overrides(cfg)
        config_dict = {**loaded_config, **cli_overrides}
    else:
        config_dict = filter_config_fields(config_dict)

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
