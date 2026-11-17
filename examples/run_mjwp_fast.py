# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Deterministic optimization variant of run_mjwp.py.

Instead of re-rolling out the optimized controls, this script replays the
pre-computed best trajectory states and restores the full simulation state
from the optimizer's snapshot.

Retry/revert is modeled as an explicit state machine — see `_StepState`
and the transition table at the top of `main()`.

Author: Chaoyi Pan
Date: 2025-04-16
"""

from __future__ import annotations

import collections
import math
import sys
import time
from dataclasses import dataclass, field, fields
from pathlib import Path

import hydra
import imageio
import loguru
import mujoco
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from spider.config import (
    Config,
    filter_config_fields,
    load_config_yaml,
    process_config,
)
from spider.interp import get_slice
from spider.io import load_data
from spider.optimizers.sampling_fast import (
    _restore_best_state_to_env,
    make_optimize_fn_fast,
    make_optimize_once_fn_fast,
    make_rollout_fn_fast,
)
from spider.postprocess.get_success_rate import compute_object_tracking_error
from spider.simulators.mjwp import (
    _initial_state_sanity_check,
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
    setup_mj_model,
    step_env,
    sync_env_mujoco,
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
    for fld in fields(config):
        if fld.name in _CONFIG_SKIP_FIELDS:
            continue
        config_dict[fld.name] = _normalize_yaml_value(getattr(config, fld.name))
    output_path = Path(config.output_dir) / "config.yaml"
    OmegaConf.save(config=OmegaConf.create(config_dict), f=str(output_path))
    loguru.logger.info(f"Saved config to {output_path}")


# --- Progressive retry boost ------------------------------------------------
#
# On each retry/revert pop, boost_level increments. Iterations double per
# level, noise scales by sqrt(2) per level — both fixed-base (relative to
# the captured originals), so level=2 is orig*4 iterations, orig*2 noise.

_BOOST_NOISE_FACTOR = math.sqrt(2.0)


def _apply_boost(
    config: Config,
    boost_level: int,
    orig_max_iters: int,
    orig_env_params_list: list,
    orig_noise_scale: torch.Tensor,
) -> None:
    iters = orig_max_iters * (2**boost_level)
    config.max_num_iterations = iters
    config.beta_traj = config.final_noise_scale ** (1.0 / iters)
    extended = list(orig_env_params_list)
    while len(extended) < iters:
        extended.append(extended[-1])
    config.env_params_list = extended
    config.noise_scale = orig_noise_scale * (_BOOST_NOISE_FACTOR**boost_level)


# --- Retry/revert state machine ---------------------------------------------
#
# States and transitions (event -> action):
#
#   OPTIMIZE    succeed                              -> OPTIMIZE
#               fail, local<cap                      -> LOCAL_RETRY
#               fail, local>=cap, history            -> REVERT (first)
#               fail, local>=cap, no history         -> DONE
#
#   LOCAL_RETRY (transient) reload curr_cp, boost    -> (unchanged state)
#
#   REVERT      (first, from OPTIMIZE) pop once, save anchor -> RECOVERING
#
#   RECOVERING  succeed, sim_step > anchor           -> OPTIMIZE (full recovery)
#               succeed, sim_step <= anchor          -> RECOVERING (no push)
#               fail, local<cap                      -> LOCAL_RETRY
#               fail, local>=cap, attempts<max       -> RE-REVERT (same anchor)
#               fail, local>=cap, attempts>=max      -> DONE
#
# "Recovery anchor" = sim_step right after the first REVERT pop. On subsequent
# failures during recovery, we reload the *same* anchor checkpoint instead of
# popping deeper into history. Recovery is declared when a step succeeds at
# sim_step > anchor.


class _StepState:
    OPTIMIZE = "OPTIMIZE"
    RECOVERING = "RECOVERING"
    DONE = "DONE"


@dataclass
class _LoopCtx:
    state: str = _StepState.OPTIMIZE
    history: collections.deque = field(default_factory=collections.deque)
    local_retries: int = 0
    boost_level: int = 0
    revert_anchor_step: int | None = None
    revert_anchor_checkpoint: dict | None = None
    revert_attempts_from_anchor: int = 0


def main(config: Config):
    """Run SPIDER using MuJoCo Warp backend with deterministic best-trajectory replay."""
    config = process_config(config)

    if config.contact_guidance:
        raise NotImplementedError(
            "contact_guidance is not supported in run_mjwp_fast.py. "
            "Use examples/run_mjwp.py for contact-guided runs."
        )
    if config.gibbs_sampling:
        raise NotImplementedError(
            "gibbs_sampling is not supported in run_mjwp_fast.py. "
            "Use examples/run_mjwp.py for Gibbs sampling."
        )

    qpos_ref, qvel_ref, ctrl_ref, contact, contact_pos = load_data(
        config, config.data_path
    )

    ref_data = (qpos_ref, qvel_ref, ctrl_ref, contact, contact_pos)
    config.max_sim_steps = (
        config.max_sim_steps
        if config.max_sim_steps > 0
        else qpos_ref.shape[0] - config.horizon_steps - config.ctrl_steps
    )

    env = setup_env(config, ref_data)

    mj_model = setup_mj_model(config)
    mj_data = mujoco.MjData(mj_model)
    mj_data_ref = mujoco.MjData(mj_model)
    mj_data.qpos[:] = qpos_ref[0].detach().cpu().numpy()
    mj_data.qvel[:] = qvel_ref[0].detach().cpu().numpy()
    mj_data.ctrl[:] = ctrl_ref[0].detach().cpu().numpy()
    mujoco.mj_step(mj_model, mj_data)
    mj_data.time = 0.0

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
        if name is not None and name.startswith("trace"):
            if "object" in name:
                object_trace_site_ids.append(sid)
            else:
                robot_trace_site_ids.append(sid)
    config.trace_site_ids = object_trace_site_ids + robot_trace_site_ids

    # setup env params (no DR-schedule dependency on contact guidance)
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
    env_params_list = []
    for _ in range(config.max_num_iterations):
        env_params = [
            {"xy_offset": xy_offset_list[j], "pair_margin": pair_margin_list[j]}
            for j in range(config.num_dr)
        ]
        env_params_list.append(env_params)
    config.env_params_list = env_params_list
    _save_config_yaml(config)

    # setup viewer and renderer
    run_viewer = setup_viewer(config, mj_model, mj_data)
    renderer = setup_renderer(config, mj_model)

    # If viser is enabled and the user wants the recording saved, start
    # recording NOW so every per-frame transform update from log_frame is
    # captured to a serializable timeline (a single .viser file).
    _viser_recording_active = False
    if "viser" in (config.viewer or "").lower() and config.save_viser:
        try:
            from spider.viewers.viser_viewer import start_recording

            start_recording(dt=float(config.render_dt))
            _viser_recording_active = True
        except Exception as e:
            loguru.logger.warning(f"Failed to start viser recording: {e}")

    # Register an atexit + SIGTERM handler so timeout-killed runs still
    # flush whatever optimization video / viser timeline was captured.
    # Without this, mp4 + .viser are only written if main() reaches its
    # end-of-loop save block, which fails-to-converge runs never do.
    import atexit
    import os as _os
    import signal as _signal

    _flushed = {"done": False}

    def _emergency_flush() -> None:
        if _flushed["done"]:
            return
        _flushed["done"] = True
        # Save partial mp4 if any frames are queued
        try:
            if config.save_video and len(images) > 0:
                video_path = (
                    f"{config.output_dir}/visualization_mjwp_fast.mp4"
                )
                imageio.mimsave(
                    video_path,
                    images,
                    fps=int(1 / config.render_dt),
                )
                loguru.logger.warning(
                    f"[emergency flush] Saved partial video ({len(images)} frames) to {video_path}"
                )
        except Exception as e:  # noqa: BLE001
            loguru.logger.warning(f"[emergency flush] mp4 save failed: {e}")
        # Save viser timeline if recording was started
        try:
            if _viser_recording_active:
                from spider.viewers.viser_viewer import end_recording_and_save

                viser_path = (
                    f"{config.output_dir}/visualization_mjwp_fast.viser"
                )
                if end_recording_and_save(viser_path):
                    loguru.logger.warning(
                        f"[emergency flush] Saved partial viser timeline to {viser_path}"
                    )
        except Exception as e:  # noqa: BLE001
            loguru.logger.warning(f"[emergency flush] viser save failed: {e}")

    atexit.register(_emergency_flush)

    def _on_sigterm(_sig, _frame) -> None:
        loguru.logger.warning("Received SIGTERM; flushing artifacts.")
        _emergency_flush()
        # Re-raise default handler so the process actually exits.
        _signal.signal(_signal.SIGTERM, _signal.SIG_DFL)
        _os.kill(_os.getpid(), _signal.SIGTERM)

    try:
        _signal.signal(_signal.SIGTERM, _on_sigterm)
    except Exception:  # noqa: BLE001
        pass

    # setup optimizer — use fast variants
    rollout = make_rollout_fn_fast(
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
        get_qpos,
        get_qvel,
    )
    optimize_once = make_optimize_once_fn_fast(rollout)
    optimize = make_optimize_fn_fast(optimize_once)

    ctrls = ctrl_ref[: config.horizon_steps]
    info_list = []

    orig_max_iters = config.max_num_iterations
    orig_env_params_list = list(config.env_params_list)
    orig_noise_scale = config.noise_scale.clone()

    last_snap = None
    last_snap_idx = None

    def _save_checkpoint():
        return {
            "qpos": mj_data.qpos.copy(),
            "qvel": mj_data.qvel.copy(),
            "ctrl": mj_data.ctrl.copy(),
            "time": mj_data.time,
            "ctrls": ctrls.clone(),
            "env_snap": last_snap,
            "env_idx": last_snap_idx,
            "images_len": len(images),
            "info_list_len": len(info_list),
        }

    def _load_checkpoint(cp):
        nonlocal ctrls, last_snap, last_snap_idx
        mj_data.qpos[:] = cp["qpos"]
        mj_data.qvel[:] = cp["qvel"]
        mj_data.ctrl[:] = cp["ctrl"]
        mj_data.time = cp["time"]
        mujoco.mj_forward(mj_model, mj_data)
        ctrls = cp["ctrls"].clone()
        last_snap = cp["env_snap"]
        last_snap_idx = cp["env_idx"]
        if last_snap is not None:
            _restore_best_state_to_env(env, last_snap, last_snap_idx)
        else:
            sync_env_mujoco(config, env, mj_data)
        del images[cp["images_len"] :]
        del info_list[cp["info_list_len"] :]

    def _run_optimize():
        nonlocal ctrls
        s = int(np.round(mj_data.time / config.sim_dt))
        rs = get_slice(ref_data, s + 1, s + config.horizon_steps + 1)
        ctrls, inf = optimize(config, env, ctrls, rs)
        if len(config.trace_site_ids) > 0:
            tr = []
            qrh = rs[0]
            for h in range(config.horizon_steps):
                mj_data_ref.qpos[:] = qrh[h].detach().cpu().numpy()
                mujoco.mj_kinematics(mj_model, mj_data_ref)
                tr.append(
                    np.array(
                        [mj_data_ref.site_xpos[sid] for sid in config.trace_site_ids]
                    )
                )
            inf["trace_ref"] = np.stack(tr, axis=0)[None, None, :, :, :]
        return s, rs, inf

    def _replay_and_advance(sim_step_val, inf):
        nonlocal ctrls
        bqp = torch.from_numpy(inf["best_qpos"])
        bqv = torch.from_numpy(inf["best_qvel"])
        si = {"qpos": [], "qvel": [], "time": [], "ctrl": []}
        for i in range(config.ctrl_steps):
            mj_data.qpos[:] = bqp[i].numpy()
            mj_data.qvel[:] = bqv[i].numpy()
            mj_data.ctrl[:] = ctrls[i].detach().cpu().numpy()
            mj_data.time += config.sim_dt
            mujoco.mj_forward(mj_model, mj_data)
            if (
                config.save_video
                and renderer is not None
                and i % int(np.round(config.render_dt / config.sim_dt)) == 0
            ):
                mj_data_ref.qpos[:] = qpos_ref[sim_step_val + i].detach().cpu().numpy()
                img = render_image(config, renderer, mj_model, mj_data, mj_data_ref)
                images.append(img)
            if "rerun" in config.viewer or "viser" in config.viewer:
                mj_data_ref.qpos[:] = qpos_ref[sim_step_val + i].detach().cpu().numpy()
                mujoco.mj_kinematics(mj_model, mj_data_ref)
                log_frame(
                    mj_data,
                    sim_time=mj_data.time,
                    viewer_body_entity_and_ids=config.viewer_body_entity_and_ids,
                    data_ref=mj_data_ref,
                )
                # Tick the optimization-process recording so this frame is
                # placed at its own timestamp on the timeline.
                if "viser" in config.viewer and config.save_viser:
                    try:
                        from spider.viewers.viser_viewer import tick_recording

                        tick_recording()
                    except Exception:
                        pass
            si["qpos"].append(mj_data.qpos.copy())
            si["qvel"].append(mj_data.qvel.copy())
            si["time"].append(mj_data.time)
            si["ctrl"].append(mj_data.ctrl.copy())
        for k in si:
            si[k] = np.stack(si[k], axis=0)
        inf.update(si)
        _restore_best_state_to_env(env, inf["state_snapshot"], inf["best_idx"])
        new_sim_step = int(np.round(mj_data.time / config.sim_dt))
        prev_c = ctrls[config.ctrl_steps :]
        new_c = ctrl_ref[
            new_sim_step + prev_c.shape[0] : new_sim_step
            + prev_c.shape[0]
            + config.ctrl_steps
        ]
        ctrls = torch.cat([prev_c, new_c], dim=0)
        mj_data_ref.qpos[:] = qpos_ref[new_sim_step].detach().cpu().numpy()
        update_viewer(config, viewer, mj_model, mj_data, mj_data_ref, inf)
        info_list.append(
            {
                k: v
                for k, v in inf.items()
                if k not in ["trace_sample", "state_snapshot"]
            }
        )
        return new_sim_step

    def _full_reset():
        ctx.boost_level = 0
        ctx.revert_anchor_step = None
        ctx.revert_anchor_checkpoint = None
        ctx.revert_attempts_from_anchor = 0
        ctx.local_retries = 0
        _apply_boost(config, 0, orig_max_iters, orig_env_params_list, orig_noise_scale)

    def _truncate_to_latest_commit():
        if ctx.history:
            anchor_cp = ctx.history[-1]
            del images[anchor_cp["images_len"] :]
            del info_list[anchor_cp["info_list_len"] :]
        else:
            del images[:]
            del info_list[:]

    ctx = _LoopCtx()
    # sim_step is defined in the loop; keep a module-visible default for the
    # `sim_step >= max_sim_steps` terminal check.
    sim_step = 0

    t_start = time.perf_counter()
    with run_viewer() as viewer:
        while ctx.state != _StepState.DONE and viewer.is_running():
            t0 = time.perf_counter()
            curr_checkpoint = _save_checkpoint()
            sim_step, ref_slice, infos = _run_optimize()

            # --- Per-step tracking error for diagnostics ---
            bqp_diag = infos.get("best_qpos")
            err_info = {"max_pos_error": float("nan"), "max_rot_error": float("nan")}
            if bqp_diag is not None:
                from spider.optimizers.sampling_fast import check_tracking_error

                bqp_t = (
                    torch.from_numpy(bqp_diag)
                    if isinstance(bqp_diag, np.ndarray)
                    else bqp_diag
                ).to(config.device)
                # Same alignment as in sampling_fast.optimize: recorded_qpos[i]
                # lives at ref index sim_step+1+i = ref_slice[0][i] (NOT [i+1]).
                avail = min(config.ctrl_steps, ref_slice[0].shape[0])
                ref_qpos_diag = ref_slice[0][:avail].to(config.device)
                _, err_info = check_tracking_error(config, bqp_t[:avail], ref_qpos_diag)
                infos["step_pos_error"] = np.array([err_info["max_pos_error"]])
                infos["step_rot_error"] = np.array([err_info["max_rot_error"]])

            succeeded = bool(infos["succeeded"][0])

            if succeeded:
                # --- SUCCESS ---
                full_recovery = (
                    ctx.state == _StepState.RECOVERING
                    and ctx.revert_anchor_step is not None
                    and sim_step > ctx.revert_anchor_step
                )
                if full_recovery:
                    loguru.logger.info(
                        "Recovered after {} revert attempts at sim_step={}.",
                        ctx.revert_attempts_from_anchor,
                        sim_step,
                    )
                    _full_reset()
                    ctx.state = _StepState.OPTIMIZE

                ctx.local_retries = 0
                if ctx.state == _StepState.OPTIMIZE:
                    ctx.history.append(curr_checkpoint)
                last_snap = infos["state_snapshot"]
                last_snap_idx = infos["best_idx"]
                sim_step = _replay_and_advance(sim_step, infos)

            elif ctx.local_retries < config.max_local_retries:
                # --- LOCAL_RETRY ---
                ctx.local_retries += 1
                loguru.logger.warning(
                    "Local retry {}/{} at sim_step={} (pos={:.4f}, rot={:.4f}).",
                    ctx.local_retries,
                    config.max_local_retries,
                    sim_step,
                    err_info["max_pos_error"],
                    err_info["max_rot_error"],
                )
                _load_checkpoint(curr_checkpoint)
                ctx.boost_level += 1
                _apply_boost(
                    config,
                    ctx.boost_level,
                    orig_max_iters,
                    orig_env_params_list,
                    orig_noise_scale,
                )
                continue

            elif (
                ctx.state == _StepState.RECOVERING
                and ctx.revert_anchor_checkpoint is not None
            ):
                # --- RE-REVERT to same anchor ---
                ctx.revert_attempts_from_anchor += 1
                if ctx.revert_attempts_from_anchor > config.max_revert_depth:
                    loguru.logger.error(
                        "Exhausted {} revert attempts from anchor sim_step={}.",
                        config.max_revert_depth,
                        ctx.revert_anchor_step,
                    )
                    _apply_boost(
                        config,
                        0,
                        orig_max_iters,
                        orig_env_params_list,
                        orig_noise_scale,
                    )
                    _truncate_to_latest_commit()
                    sim_step = _replay_and_advance(sim_step, infos)
                    ctx.state = _StepState.DONE
                    break
                ctx.local_retries = 0
                _load_checkpoint(ctx.revert_anchor_checkpoint)
                # Reset boost so iters/noise don't compound across anchors —
                # each new anchor starts a fresh local-retry sweep at level 0.
                ctx.boost_level = 0
                _apply_boost(
                    config,
                    ctx.boost_level,
                    orig_max_iters,
                    orig_env_params_list,
                    orig_noise_scale,
                )
                loguru.logger.warning(
                    "Re-revert attempt {}/{} back to anchor sim_step={}.",
                    ctx.revert_attempts_from_anchor,
                    config.max_revert_depth,
                    ctx.revert_anchor_step,
                )
                continue

            elif ctx.history:
                # --- FIRST REVERT (from OPTIMIZE state) ---
                ctx.local_retries = 0
                prev_cp = ctx.history.pop()
                ctx.revert_anchor_checkpoint = prev_cp
                _load_checkpoint(prev_cp)
                ctx.revert_anchor_step = int(np.round(mj_data.time / config.sim_dt))
                ctx.revert_attempts_from_anchor = 1
                # Reset boost so iters/noise don't compound across anchors —
                # each new anchor starts a fresh local-retry sweep at level 0.
                ctx.boost_level = 0
                _apply_boost(
                    config,
                    ctx.boost_level,
                    orig_max_iters,
                    orig_env_params_list,
                    orig_noise_scale,
                )
                ctx.state = _StepState.RECOVERING
                loguru.logger.warning(
                    "Revert attempt 1/{} from sim_step={} to anchor sim_step={}.",
                    config.max_revert_depth,
                    sim_step,
                    ctx.revert_anchor_step,
                )
                continue

            else:
                # --- DONE (no history to revert to) ---
                loguru.logger.error(
                    "Unrecoverable failure at sim_step={} (no history).",
                    sim_step,
                )
                _apply_boost(
                    config, 0, orig_max_iters, orig_env_params_list, orig_noise_scale
                )
                _truncate_to_latest_commit()
                sim_step = _replay_and_advance(sim_step, infos)
                ctx.state = _StepState.DONE
                break

            t1 = time.perf_counter()
            rtr = config.ctrl_dt / (t1 - t0)
            n_att = infos.get("num_attempts", [1])
            n_att_v = n_att[0] if hasattr(n_att, "__len__") else n_att
            print(
                f"Realtime rate: {rtr:.2f}, plan time: {t1 - t0:.4f}s, "
                f"sim_steps: {sim_step}/{config.max_sim_steps}, "
                f"opt_steps: {infos['opt_steps'][0]}, attempts: {n_att_v}, "
                f"state: {ctx.state}, boost: {ctx.boost_level}",
                end="\r",
            )

            if sim_step >= config.max_sim_steps:
                ctx.state = _StepState.DONE
                break

        t_end = time.perf_counter()
        print(f"Total time: {t_end - t_start:.4f}s")

    if config.save_info and len(info_list) > 0:
        info_aggregated = {}
        for k in info_list[0]:
            if not isinstance(info_list[0][k], np.ndarray):
                continue
            vals = [info[k] for info in info_list if k in info]
            if len(vals) == 0:
                continue
            if all(v.shape == vals[0].shape for v in vals):
                info_aggregated[k] = np.stack(vals, axis=0)
            elif k in ("qpos", "qvel", "time", "ctrl"):
                info_aggregated[k] = np.concatenate(vals, axis=0)
        np.savez(
            f"{config.output_dir}/trajectory_mjwp_fast.npz",
            **info_aggregated,
        )
        loguru.logger.info(
            f"Saved info to {config.output_dir}/trajectory_mjwp_fast.npz"
        )

    if config.save_video and len(images) > 0:
        video_path = f"{config.output_dir}/visualization_mjwp_fast.mp4"
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
        errors = compute_object_tracking_error(
            qpos_traj, qpos_ref_np, config.embodiment_type, "mjwp"
        )
        loguru.logger.info(
            "Final object tracking error: pos={:.4f}, quat={:.4f}",
            errors["obj_pos_err"],
            errors["obj_quat_err"],
        )
        # Smoothness check: detect jumps between control ticks
        if qpos_traj.shape[0] > 1:
            diffs = np.linalg.norm(np.diff(qpos_traj, axis=0), axis=1)
            max_jump = diffs.max()
            max_jump_frame = int(diffs.argmax())
            loguru.logger.info(
                "Smoothness: max frame-to-frame qpos diff={:.6f} at frame {}/{}",
                max_jump,
                max_jump_frame,
                qpos_traj.shape[0],
            )
            for tick in range(config.ctrl_steps, qpos_traj.shape[0], config.ctrl_steps):
                boundary_jump = np.linalg.norm(qpos_traj[tick] - qpos_traj[tick - 1])
                if boundary_jump > 0.5:
                    loguru.logger.warning(
                        "Large jump at tick boundary frame {}: diff={:.4f}",
                        tick,
                        boundary_jump,
                    )

    if "viser" in config.viewer and config.save_viser:
        viser_path = f"{config.output_dir}/visualization_mjwp_fast.viser"
        try:
            from spider.viewers.viser_viewer import (
                _STATE as _viser_state,
                end_recording_and_save,
            )

            # If a recording was started, prefer the timeline recording.
            # Falls back to a single-frame snapshot when no recording exists.
            ok = end_recording_and_save(viser_path)
            if ok:
                loguru.logger.info(
                    f"Saved viser optimization recording to {viser_path}"
                )
            else:
                server = _viser_state.server
                if server is not None:
                    Path(viser_path).write_bytes(
                        server.get_scene_serializer().serialize()
                    )
                    loguru.logger.info(f"Saved viser scene to {viser_path}")
        except Exception as e:
            loguru.logger.warning(f"Failed to save viser scene: {e}")

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
    config_dict = dict(cfg)

    load_config_path = config_dict.get("load_config_path", "")
    if load_config_path:
        loaded_config = load_config_yaml(load_config_path)
        cli_overrides = _extract_cli_overrides(cfg)
        config_dict = {**loaded_config, **cli_overrides}
    else:
        config_dict = filter_config_fields(config_dict)

    if "noise_scale" in config_dict and config_dict["noise_scale"] is None:
        config_dict.pop("noise_scale")

    if "pair_margin_range" in config_dict:
        config_dict["pair_margin_range"] = tuple(config_dict["pair_margin_range"])
    if "xy_offset_range" in config_dict:
        config_dict["xy_offset_range"] = tuple(config_dict["xy_offset_range"])

    config = Config(**config_dict)
    main(config)


if __name__ == "__main__":
    run_main()
