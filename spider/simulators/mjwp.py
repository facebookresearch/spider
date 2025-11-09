"""Simulator for sampling with MuJoCo Warp (mjwarp).

This module provides a minimal MJWP backend that matches the sampling API used by
the generic optimizer pipeline. It intentionally keeps the implementation simple
and robust (no DR groups here; see legacy script for advanced features).
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp

# NOTE: this is a hacky solution to make sure domain randomization works for contact margin. Otherwise, it will create a surrogate memory for all worlds and we cannot override each individual world's contact parameters.
# mjwarp._src.io.MAX_WORLDS = 1024
from spider.config import Config
from spider.math import quat_sub

# Initialize Warp once per process
try:
    wp.init()
except RuntimeError:
    # Already initialized
    pass


@dataclass
class MJWPEnv:
    model_cpu: mujoco.MjModel
    data_cpu: mujoco.MjData
    # Unified data sink always reflecting last step's state
    model_wp: mjwarp.Model
    data_wp: mjwarp.Data
    data_wp_prev: mjwarp.Data
    graph: wp.ScopedCapture.Graph
    # Device alias used for Warp allocations/launches (e.g., "cuda:1" or "cpu")
    device: str
    num_worlds: int


def _compile_step(
    model_wp: mjwarp.Model, data_wp: mjwarp.Data
) -> wp.ScopedCapture.Graph:
    """Warm up and capture a CUDA graph that runs a single mjwarp.step."""

    def _step_once():
        mjwarp.step(model_wp, data_wp)

    # Warmup/compile
    _step_once()
    _step_once()
    wp.synchronize()
    # Capture
    with wp.ScopedCapture() as capture:
        _step_once()
    wp.synchronize()
    return capture.graph


# TODO: define update environment parameter kernel functions, combine them compile step, also add parameter to be modified into MJWPEnv

# --
# Key functions
# --


def setup_mj_model(config: Config) -> mujoco.MjModel:
    model_cpu = mujoco.MjModel.from_xml_path(config.model_path)
    model_cpu.opt.timestep = float(config.sim_dt)
    if config.embodiment_type in ["left", "right", "bimanual"]:
        # setup for hand
        model_cpu.opt.iterations = 20
        model_cpu.opt.ls_iterations = 50
        model_cpu.opt.o_solref = [0.02, 1.0]
        model_cpu.opt.o_solimp = [
            0.0,
            0.95,
            0.03,
            0.5,
            2,
        ]  # softer contact for sim2real
        model_cpu.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    elif config.embodiment_type in ["humanoid", "humanoid_object"]:
        # setup for humanoid
        model_cpu.opt.iterations = 2
        model_cpu.opt.ls_iterations = 10
        model_cpu.opt.o_solref = [0.02, 1.0]
        model_cpu.opt.o_solimp = [
            0.9,
            0.95,
            0.001,
            0.5,
            2,
        ]  # softer contact for sim2real
        model_cpu.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    return model_cpu


def setup_env(config: Config, ref_data: tuple[torch.Tensor, ...]) -> MJWPEnv:
    """Setup and reset the environment backed by MJWP.
    Returns an MJWPEnv with captured graph.
    """
    qpos_ref, qvel_ref, ctrl_ref, contact_ref, contact_pos_ref = ref_data
    qpos_init = qpos_ref[0]

    # CPU model/data
    model_cpu = setup_mj_model(config)
    data_cpu = mujoco.MjData(model_cpu)
    # Seed initial state
    arrs = (qpos_init, qvel_ref[0], ctrl_ref[0])
    data_cpu.qpos[:] = arrs[0].detach().cpu().numpy()
    data_cpu.qvel[:] = arrs[1].detach().cpu().numpy()
    data_cpu.ctrl[:] = arrs[2].detach().cpu().numpy()
    mujoco.mj_step(model_cpu, data_cpu)

    # Move to Warp (batched worlds)
    # Set Warp default device to match config to ensure kernels/modules load on it
    wp.set_device(str(config.device))
    # Build default model/data/graph on the configured device
    dev = str(config.device)
    with wp.ScopedDevice(dev):
        default_model_wp = mjwarp.put_model(model_cpu)
        # pair_margin_override_np = (
        #     np.zeros((int(config.num_samples), model_cpu.npair)).astype(np.float32)
        #     + 0.01
        # )
        # pair_margin_override_wp = wp.from_numpy(
        #     pair_margin_override_np, dtype=wp.float32, device=dev
        # )
        # default_model_wp.pair_margin = pair_margin_override_wp

        default_data_wp = mjwarp.put_data(
            model_cpu,
            data_cpu,
            nworld=int(config.num_samples),
            nconmax=int(config.nconmax_per_env),
            njmax=int(config.njmax_per_env),
        )
        data_wp_prev = mjwarp.put_data(
            model_cpu,
            data_cpu,
            nworld=int(config.num_samples),
            nconmax=int(config.nconmax_per_env),
            njmax=int(config.njmax_per_env),
        )
        default_graph = _compile_step(default_model_wp, default_data_wp)

    # Initialize env; default active is main
    return MJWPEnv(
        model_cpu=model_cpu,
        data_cpu=data_cpu,
        model_wp=default_model_wp,
        data_wp=default_data_wp,
        data_wp_prev=data_wp_prev,
        graph=default_graph,
        device=dev,
        num_worlds=int(config.num_samples),
    )


def _weight_diff_qpos(config: Config) -> torch.Tensor:
    w = torch.ones(config.nv, device=config.device)
    if config.embodiment_type == "bimanual":
        half_dof = int(config.nu // 2)
        w[:3] = config.base_pos_rew_scale
        w[3:6] = config.base_rot_rew_scale
        w[6:half_dof] = config.joint_rew_scale
        w[half_dof : half_dof + 3] = config.base_pos_rew_scale
        w[half_dof + 3 : half_dof + 6] = config.base_rot_rew_scale
        w[half_dof + 6 : config.nu] = config.joint_rew_scale
        # object
        w[-12:-9] = config.pos_rew_scale
        w[-9:-6] = config.rot_rew_scale
        w[-6:-3] = config.pos_rew_scale
        w[-3:] = config.rot_rew_scale
    elif config.embodiment_type in ["right", "left"]:
        w[:3] = config.base_pos_rew_scale
        w[3:6] = config.base_rot_rew_scale
        w[6 : config.nu] = config.joint_rew_scale
        w[-6:-3] = config.pos_rew_scale
        w[-3:] = config.rot_rew_scale
    elif config.embodiment_type in ["humanoid"]:  # humanoid robot
        # robot pos and rot
        w[:3] = config.pos_rew_scale
        w[3:6] = config.rot_rew_scale
        # robot joint
        w[6:] = config.joint_rew_scale
    elif config.embodiment_type in ["humanoid_object"]:
        # robot pos and rot
        w[:3] = config.base_pos_rew_scale
        w[3:6] = config.base_rot_rew_scale
        # robot joint
        w[6:-6] = config.joint_rew_scale
        # object pos and rot
        w[-6:-3] = config.pos_rew_scale
        w[-3:] = config.rot_rew_scale
    else:
        raise ValueError(f"Invalid embodiment_type: {config.embodiment_type}")
    return w


def _diff_qpos(
    config: Config, qpos_sim: torch.Tensor, qpos_ref: torch.Tensor
) -> torch.Tensor:
    """Compute the difference between qpos_sim and qpos_ref
    TODO: replace with mujoco built-in function, not sure how to call warp internal function yet.
    """
    batch_size = qpos_sim.shape[0]
    qpos_diff = torch.zeros((batch_size, config.nv), device=config.device)
    if config.embodiment_type == "bimanual":
        # joint
        qpos_diff[:, :-12] = qpos_sim[:, :-14] - qpos_ref[:, :-14]
        # position
        qpos_diff[:, -12:-9] = qpos_sim[:, -14:-11] - qpos_ref[:, -14:-11]
        qpos_diff[:, -6:-3] = qpos_sim[:, -7:-4] - qpos_ref[:, -7:-4]
        # rotation
        qpos_diff[:, -9:-6] = quat_sub(qpos_sim[:, -11:-7], qpos_ref[:, -11:-7])
        qpos_diff[:, -3:] = quat_sub(qpos_sim[:, -4:], qpos_ref[:, -4:])
    elif config.embodiment_type in ["right", "left"]:
        # joint
        qpos_diff[:, :-6] = qpos_sim[:, :-7] - qpos_ref[:, :-7]
        # position
        qpos_diff[:, -6:-3] = qpos_sim[:, -7:-4] - qpos_ref[:, -7:-4]
        # rotation
        qpos_diff[:, -3:] = quat_sub(qpos_sim[:, -4:], qpos_ref[:, -4:])
    elif config.embodiment_type in ["humanoid"]:
        # joint
        qpos_diff[:, 6:] = qpos_sim[:, 7:] - qpos_ref[:, 7:]
        # position
        qpos_diff[:, :3] = qpos_sim[:, :3] - qpos_ref[:, :3]
        # rotation
        qpos_diff[:, 3:6] = quat_sub(qpos_sim[:, 3:7], qpos_ref[:, 3:7])
    elif config.embodiment_type in ["humanoid_object"]:
        qpos_humanoid = qpos_sim[:, :-7]
        qpos_object = qpos_sim[:, -7:]
        qpos_ref_humanoid = qpos_ref[:, :-7]
        qpos_ref_object = qpos_ref[:, -7:]
        # position
        qpos_diff[:, :3] = qpos_humanoid[:, :3] - qpos_ref_humanoid[:, :3]
        # rotation
        qpos_diff[:, 3:6] = quat_sub(qpos_humanoid[:, 3:7], qpos_ref_humanoid[:, 3:7])
        # joint
        qpos_diff[:, 6:-6] = qpos_humanoid[:, 7:] - qpos_ref_humanoid[:, 7:]
        # object
        qpos_diff[:, -6:-3] = qpos_object[:, :3] - qpos_ref_object[:, :3]
        qpos_diff[:, -3:] = quat_sub(qpos_object[:, 3:7], qpos_ref_object[:, 3:7])
    else:
        raise ValueError(f"Invalid embodiment_type: {config.embodiment_type}")
    return qpos_diff


def get_reward(
    config: Config,
    env: MJWPEnv,
    ref: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Non-terminal step reward for MJWP batched worlds.
    ref is a tuple: (qpos_ref, qvel_ref, ctrl_ref, contact_ref, contact_pos_ref)
    Returns (N,)

    TODO: move reward computation to task-specific module
    """
    qpos_ref, qvel_ref, ctrl_ref, contact_ref, _contact_pos_ref = ref
    qpos_sim = wp.to_torch(env.data_wp.qpos)
    qvel_sim = wp.to_torch(env.data_wp.qvel)
    # weighted qpos tracking
    qpos_diff = _diff_qpos(
        config, qpos_sim, qpos_ref.unsqueeze(0).repeat(qpos_sim.shape[0], 1)
    )
    qpos_weight = _weight_diff_qpos(config)
    delta_qpos = qpos_diff * qpos_weight
    qpos_dist = torch.norm(delta_qpos, p=2, dim=1)
    qvel_dist = torch.norm(qvel_sim - qvel_ref, p=2, dim=1)

    qpos_rew = -qpos_dist * 1.0
    qvel_rew = -config.vel_rew_scale * qvel_dist * 1.0
    reward = qpos_rew + qvel_rew

    info = {
        "qpos_dist": qpos_dist,
        "qvel_dist": qvel_dist,
        "qpos_rew": qpos_rew,
        "qvel_rew": qvel_rew,
    }
    return reward, info


def get_terminal_reward(
    config: Config,
    env: MJWPEnv,
    ref_slice: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Terminal reward focusing on object tracking."""
    # return config.terminal_rew_scale * get_reward(config, env, ref_slice)
    # qpos_ref, qvel_ref, ctrl_ref, contact_ref, _ = ref_slice
    # qpos_sim = wp.to_torch(env.data_wp.qpos)
    # qpos_weight = torch.zeros(qpos_sim.shape[1], device=config.device)
    # if config.embodiment_type == "bimanual":
    #     qpos_weight[-14:-11] = config.pos_rew_scale
    #     qpos_weight[-11:-7] = config.rot_rew_scale
    #     qpos_weight[-7:-4] = config.pos_rew_scale
    #     qpos_weight[-4:] = config.rot_rew_scale
    # elif config.embodiment_type in ["right", "left"]:
    #     qpos_weight[-7:-4] = config.pos_rew_scale
    #     qpos_weight[-4:] = config.rot_rew_scale
    # elif config.embodiment_type in ["CMU", "DanceDB"]:
    #     qpos_weight[:3] = config.pos_rew_scale
    #     qpos_weight[3:7] = config.rot_rew_scale
    # else:
    #     raise ValueError(f"Invalid embodiment_type: {config.embodiment_type}")
    # delta_qpos = (qpos_sim - qpos_ref) * qpos_weight
    # cost_object = config.terminal_rew_scale * torch.sum(delta_qpos**2, dim=1)

    rew, info = get_reward(config, env, ref_slice)
    terminal_rew = config.terminal_rew_scale * rew
    return terminal_rew, info


def get_qpos(config: Config, env: MJWPEnv) -> torch.Tensor:
    return wp.to_torch(env.data_wp.qpos)


def set_qpos(config: Config, env: MJWPEnv, qpos: torch.Tensor):
    qpos = qpos.to(config.device)
    if qpos.dim() == 1:
        qpos = qpos.unsqueeze(0).repeat(env.num_worlds, 1)
    wp.copy(env.data_wp.qpos, wp.from_torch(qpos))
    # reset velocities/time as well for consistency
    zero_qvel = torch.zeros((env.num_worlds, env.model_cpu.nv), device=config.device)
    wp.copy(env.data_wp.qvel, wp.from_torch(zero_qvel))
    wp.copy(
        env.data_wp.time,
        wp.from_torch(
            torch.zeros(env.num_worlds, dtype=torch.float32, device=config.device)
        ),
    )


def get_qvel(config: Config, env: MJWPEnv) -> torch.Tensor:
    return wp.to_torch(env.data_wp.qvel)


def get_trace(config: Config, env: MJWPEnv) -> torch.Tensor:
    """Return per-world trace points used for visualization. Minimal default returns
    an empty trace set of shape (N, 0, 3) when not configured.
    """
    site_xpos = wp.to_torch(env.data_wp.site_xpos)  # (N, nsite, 3)
    return site_xpos[:, config.trace_site_ids, :]


def save_state(env: MJWPEnv):
    """Clone the essential set of Warp arrays to restore later.
    Includes core state variables and key derived quantities.
    """
    _copy_state(env.data_wp, env.data_wp_prev)
    return env
    # qpos = wp.clone(env.data_wp.qpos)
    # qvel = wp.clone(env.data_wp.qvel)
    # qacc = wp.clone(env.data_wp.qacc)
    # time_arr = wp.clone(env.data_wp.time)
    # ctrl = wp.clone(env.data_wp.ctrl) if hasattr(env.data_wp, "ctrl") else None
    # act = wp.clone(env.data_wp.act) if hasattr(env.data_wp, "act") else None
    # act_dot = wp.clone(env.data_wp.act_dot) if hasattr(env.data_wp, "act_dot") else None
    # site_xpos = wp.clone(env.data_wp.site_xpos)
    # site_xmat = wp.clone(env.data_wp.site_xmat)
    # mocap_pos = (
    #     wp.clone(env.data_wp.mocap_pos) if hasattr(env.data_wp, "mocap_pos") else None
    # )
    # mocap_quat = (
    #     wp.clone(env.data_wp.mocap_quat) if hasattr(env.data_wp, "mocap_quat") else None
    # )
    # energy = wp.clone(env.data_wp.energy) if hasattr(env.data_wp, "energy") else None
    # return (
    #     qpos,
    #     qvel,
    #     qacc,
    #     time_arr,
    #     ctrl,
    #     act,
    #     act_dot,
    #     site_xpos,
    #     site_xmat,
    #     mocap_pos,
    #     mocap_quat,
    #     energy,
    # )


def load_state(env: MJWPEnv, state):
    _copy_state(env.data_wp_prev, env.data_wp)
    return env
    # (
    #     qpos,
    #     qvel,
    #     qacc,
    #     time_arr,
    #     ctrl,
    #     act,
    #     act_dot,
    #     site_xpos,
    #     site_xmat,
    #     mocap_pos,
    #     mocap_quat,
    #     energy,
    # ) = state
    # wp.copy(env.data_wp.qpos, qpos)
    # wp.copy(env.data_wp.qvel, qvel)
    # wp.copy(env.data_wp.qacc, qacc)
    # wp.copy(env.data_wp.time, time_arr)
    # if ctrl is not None and hasattr(env.data_wp, "ctrl"):
    #     wp.copy(env.data_wp.ctrl, ctrl)
    # if act is not None and hasattr(env.data_wp, "act"):
    #     wp.copy(env.data_wp.act, act)
    # if act_dot is not None and hasattr(env.data_wp, "act_dot"):
    #     wp.copy(env.data_wp.act_dot, act_dot)
    # if mocap_pos is not None and hasattr(env.data_wp, "mocap_pos"):
    #     wp.copy(env.data_wp.mocap_pos, mocap_pos)
    # if mocap_quat is not None and hasattr(env.data_wp, "mocap_quat"):
    #     wp.copy(env.data_wp.mocap_quat, mocap_quat)
    # if energy is not None and hasattr(env.data_wp, "energy"):
    #     wp.copy(env.data_wp.energy, energy)
    # if site_xpos is not None and hasattr(env.data_wp, "site_xpos"):
    #     wp.copy(env.data_wp.site_xpos, site_xpos)
    # if site_xmat is not None and hasattr(env.data_wp, "site_xmat"):
    #     wp.copy(env.data_wp.site_xmat, site_xmat)
    # return env


def apply_perturbation(config: Config, env: MJWPEnv):
    # get object id
    right_obj_id = mujoco.mj_name2id(
        env.model_cpu, mujoco.mjtObj.mjOBJ_BODY, "right_object"
    )
    left_obj_id = mujoco.mj_name2id(
        env.model_cpu, mujoco.mjtObj.mjOBJ_BODY, "left_object"
    )
    xfrc_applied = wp.to_torch(env.data_wp.xfrc_applied)
    if right_obj_id != -1:
        xfrc_applied[:, right_obj_id, :3] = config.perturb_force
        xfrc_applied[:, right_obj_id, 3:] = config.perturb_torque
    if left_obj_id != -1:
        xfrc_applied[:, left_obj_id, :3] = config.perturb_force
        xfrc_applied[:, left_obj_id, 3:] = config.perturb_torque
    wp.copy(env.data_wp.xfrc_applied, wp.from_torch(xfrc_applied))
    return env


def step_env(config: Config, env: MJWPEnv, ctrl_mujoco: torch.Tensor):
    """Step all worlds with provided MuJoCo-format controls of shape (N, nu)."""
    if ctrl_mujoco.dim() == 1:
        ctrl_mujoco = ctrl_mujoco.unsqueeze(0).repeat(env.num_worlds, 1)
    # Ensure we operate on the correct CUDA context/device
    with wp.ScopedDevice(env.device):
        # apply perturbation
        env = apply_perturbation(config, env)
        # step control
        wp.copy(env.data_wp.ctrl, wp.from_torch(ctrl_mujoco.to(torch.float32)))
        wp.capture_launch(env.graph)


def save_env_params(config: Config, env: MJWPEnv):
    """Save the current simulation parameters."""
    # Only record which group is active; parameters are embedded in separate models
    # TODO: explicitly read pair_margin and xy_offset from env.data_wp
    # currently we choose this solution since pair_margin has a huge virtual dimension,
    # convert it to torch would lead to OOM
    pair_margin = 0.0
    xy_offset = 0.0
    return {"pair_margin": pair_margin, "xy_offset": xy_offset}


def load_env_params(config: Config, env: MJWPEnv, env_param: dict):
    """Load the simulation parameters.

    Parameters to be updated:
    - pair_margin
    - xy_offset of the object
    """
    # update model parameters (pair_margin)
    if "pair_margin" in env_param:
        pair_margin_single_np = np.full(
            shape=(config.npair,), fill_value=env_param["pair_margin"], dtype=np.float32
        )

        # 2. Copy this small array to the GPU
        pair_margin_override_wp = wp.from_numpy(
            pair_margin_single_np, dtype=wp.float32, device=config.device
        )

        # 3. Apply the stride trick to broadcast it
        # This makes Warp treat the single instance as if it were num_samples copies
        # without allocating any new memory.
        pair_margin_override_wp.strides = (0,) + pair_margin_override_wp.strides
        pair_margin_override_wp.shape = (
            config.num_samples,
        ) + pair_margin_override_wp.shape
        pair_margin_override_wp.ndim += 1
        wp.copy(env.model_wp.pair_margin, pair_margin_override_wp)

    # update object position (NOTE: currently, xy_offset is only one scalar, which means we only update in the diagonal direction)
    if "xy_offset" in env_param:
        qpos_override_th = wp.to_torch(env.data_wp.qpos)
        # TODO: make object pos detection automatic
        if config.embodiment_type == "bimanual":
            qpos_override_th[:, -14:-12] = (
                qpos_override_th[:, -14:-12] + env_param["xy_offset"]
            )
            qpos_override_th[:, -12:-10] = (
                qpos_override_th[:, -12:-10] + env_param["xy_offset"]
            )
        elif config.embodiment_type in ["right", "left"]:
            qpos_override_th[:, -7:-5] = (
                qpos_override_th[:, -7:-5] + env_param["xy_offset"]
            )

        wp.copy(env.data_wp.qpos, wp.from_torch(qpos_override_th))

    return env


def _broadcast_state(data_wp, num_worlds: int):
    """Broadcast state from first world/env to all worlds/envs.

    This is a generic function that can be used by both MJWP and HDMI simulators.

    Args:
        data_wp: MuJoCo Warp data object (mjwarp.Data or wrapped version)
        num_worlds: Number of parallel worlds/environments
    """
    # Core state variables - always try these first
    qpos0 = wp.to_torch(data_wp.qpos)[:1]
    qvel0 = wp.to_torch(data_wp.qvel)[:1]
    time0 = wp.to_torch(data_wp.time)[:1]
    ctrl0 = wp.to_torch(data_wp.ctrl)[:1]

    # Handle time specially as it might be 1D
    if time0.dim() == 1:
        time_repeated = time0.repeat(num_worlds)
    else:
        time_repeated = time0.repeat(num_worlds, 1)

    wp.copy(data_wp.qpos, wp.from_torch(qpos0.repeat(num_worlds, 1)))
    wp.copy(data_wp.qvel, wp.from_torch(qvel0.repeat(num_worlds, 1)))
    wp.copy(data_wp.time, wp.from_torch(time_repeated))
    wp.copy(data_wp.ctrl, wp.from_torch(ctrl0.repeat(num_worlds, 1)))

    # Additional core state variables
    qacc0 = wp.to_torch(data_wp.qacc)[:1]
    wp.copy(data_wp.qacc, wp.from_torch(qacc0.repeat(num_worlds, 1)))

    act0 = wp.to_torch(data_wp.act)[:1]
    wp.copy(data_wp.act, wp.from_torch(act0.repeat(num_worlds, 1)))

    act_dot0 = wp.to_torch(data_wp.act_dot)[:1]
    wp.copy(data_wp.act_dot, wp.from_torch(act_dot0.repeat(num_worlds, 1)))

    # Forces and applied forces
    qfrc_applied0 = wp.to_torch(data_wp.qfrc_applied)[:1]
    wp.copy(data_wp.qfrc_applied, wp.from_torch(qfrc_applied0.repeat(num_worlds, 1)))

    xfrc_applied0 = wp.to_torch(data_wp.xfrc_applied)[:1]
    wp.copy(data_wp.xfrc_applied, wp.from_torch(xfrc_applied0.repeat(num_worlds, 1, 1)))

    # Mocap data
    mocap_pos0 = wp.to_torch(data_wp.mocap_pos)[:1]
    wp.copy(data_wp.mocap_pos, wp.from_torch(mocap_pos0.repeat(num_worlds, 1, 1)))

    mocap_quat0 = wp.to_torch(data_wp.mocap_quat)[:1]
    wp.copy(data_wp.mocap_quat, wp.from_torch(mocap_quat0.repeat(num_worlds, 1, 1)))

    # Spatial transformations
    xpos0 = wp.to_torch(data_wp.xpos)[:1]
    wp.copy(data_wp.xpos, wp.from_torch(xpos0.repeat(num_worlds, 1, 1)))

    xquat0 = wp.to_torch(data_wp.xquat)[:1]
    wp.copy(data_wp.xquat, wp.from_torch(xquat0.repeat(num_worlds, 1, 1)))

    xmat0 = wp.to_torch(data_wp.xmat)[:1]
    wp.copy(data_wp.xmat, wp.from_torch(xmat0.repeat(num_worlds, 1, 1, 1)))

    # Geometry positions
    geom_xpos0 = wp.to_torch(data_wp.geom_xpos)[:1]
    wp.copy(data_wp.geom_xpos, wp.from_torch(geom_xpos0.repeat(num_worlds, 1, 1)))

    geom_xmat0 = wp.to_torch(data_wp.geom_xmat)[:1]
    wp.copy(data_wp.geom_xmat, wp.from_torch(geom_xmat0.repeat(num_worlds, 1, 1, 1)))

    # Site positions
    site_xpos0 = wp.to_torch(data_wp.site_xpos)[:1]
    wp.copy(data_wp.site_xpos, wp.from_torch(site_xpos0.repeat(num_worlds, 1, 1)))


def sync_env(config: Config, env: MJWPEnv, mj_data: mujoco.MjData):
    """Broadcast the state from first env to all envs

    This function synchronizes states from the first environment to all environments.
    Uses safe copying with buffer size validation to avoid mismatches.
    """
    _broadcast_state(env.data_wp, env.num_worlds)


def sync_env_mujoco(config: Config, env: MJWPEnv, mj_data: mujoco.MjData):
    """Sync state from mj_data to env.data_wp"""
    # Define field mappings with their data and target shapes
    fields = [
        # Core state variables
        ("qpos", mj_data.qpos, (env.data_wp.nworld, -1)),
        ("qvel", mj_data.qvel, (env.data_wp.nworld, -1)),
        ("qacc", mj_data.qacc, (env.data_wp.nworld, -1)),
        ("time", np.array([mj_data.time], dtype=np.float32), (env.data_wp.nworld, 1)),
        ("ctrl", mj_data.ctrl, (env.data_wp.nworld, -1)),
        ("act", mj_data.act, (env.data_wp.nworld, -1)),
        ("act_dot", mj_data.act_dot, (env.data_wp.nworld, -1)),
        ("qacc_warmstart", mj_data.qacc_warmstart, (env.data_wp.nworld, -1)),
        # Forces
        ("qfrc_applied", mj_data.qfrc_applied, (env.data_wp.nworld, -1)),
        ("xfrc_applied", mj_data.xfrc_applied, (env.data_wp.nworld, -1, -1)),
        # Energy (2D: kinetic + potential)
        ("energy", mj_data.energy, (env.data_wp.nworld, 2)),
        # Mocap data
        ("mocap_pos", mj_data.mocap_pos, (env.data_wp.nworld, -1, 3)),
        ("mocap_quat", mj_data.mocap_quat, (env.data_wp.nworld, -1, 4)),
        # Spatial transformations
        ("xpos", mj_data.xpos, (env.data_wp.nworld, -1, 3)),
        ("xquat", mj_data.xquat, (env.data_wp.nworld, -1, 4)),
        ("xmat", mj_data.xmat, (env.data_wp.nworld, -1, 9)),
        ("xipos", mj_data.xipos, (env.data_wp.nworld, -1, 3)),
        ("ximat", mj_data.ximat, (env.data_wp.nworld, -1, 9)),
        # Geometry positions
        ("geom_xpos", mj_data.geom_xpos, (env.data_wp.nworld, -1, 3)),
        ("geom_xmat", mj_data.geom_xmat, (env.data_wp.nworld, -1, 9)),
        ("site_xpos", mj_data.site_xpos, (env.data_wp.nworld, -1, 3)),
        ("site_xmat", mj_data.site_xmat, (env.data_wp.nworld, -1, 9)),
        # Body dynamics (spatial vectors)
        ("cacc", mj_data.cacc, (env.data_wp.nworld, -1, 6)),
        ("cfrc_int", mj_data.cfrc_int, (env.data_wp.nworld, -1, 6)),
        ("cfrc_ext", mj_data.cfrc_ext, (env.data_wp.nworld, -1, 6)),
        # Sensor data
        ("sensordata", mj_data.sensordata, (env.data_wp.nworld, -1)),
        # Actuator data
        ("actuator_length", mj_data.actuator_length, (env.data_wp.nworld, -1)),
        ("actuator_velocity", mj_data.actuator_velocity, (env.data_wp.nworld, -1)),
        ("actuator_force", mj_data.actuator_force, (env.data_wp.nworld, -1)),
        # Tendon data
        ("ten_length", mj_data.ten_length, (env.data_wp.nworld, -1)),
        ("ten_velocity", mj_data.ten_velocity, (env.data_wp.nworld, -1)),
    ]

    # Contact struct fields - these need special handling
    contact_fields = [
        ("dist", "contact.dist"),
        ("pos", "contact.pos"),
        ("frame", "contact.frame"),
        ("includemargin", "contact.includemargin"),
        ("friction", "contact.friction"),
        ("solref", "contact.solref"),
        ("solreffriction", "contact.solreffriction"),
        ("solimp", "contact.solimp"),
        ("dim", "contact.dim"),
        ("geom", "contact.geom"),
        ("efc_address", "contact.efc_address"),
        ("worldid", "contact.worldid"),
    ]

    # Constraint (efc) fields - these are direct fields on mj_data, not nested in a struct
    efc_fields = [
        ("efc_type", "efc.type"),
        ("efc_id", "efc.id"),
        ("efc_J", "efc.J"),
        ("efc_pos", "efc.pos"),
        ("efc_margin", "efc.margin"),
        ("efc_D", "efc.D"),
        ("efc_vel", "efc.vel"),
        ("efc_aref", "efc.aref"),
        ("efc_frictionloss", "efc.frictionloss"),
        ("efc_force", "efc.force"),
    ]

    # Copy data to all environments
    for field_name, source_data, target_shape in fields:
        # Skip if field doesn't exist in either source or destination
        if not hasattr(mj_data, field_name) or not hasattr(env.data_wp, field_name):
            continue

        source_data_np = np.array(source_data, dtype=np.float32)
        tensor = torch.from_numpy(source_data_np).to(config.device)

        # Handle scalar time field
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)

        # Reshape tensor to match target shape
        if len(target_shape) == 2:
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0).repeat(target_shape[0], 1)
            elif tensor.dim() == 2:
                tensor = tensor.unsqueeze(0).repeat(target_shape[0], 1, 1).squeeze(-1)
        elif len(target_shape) == 3:
            if tensor.dim() == 1:
                # For 1D data that needs to be 3D
                tensor = (
                    tensor.unsqueeze(0)
                    .unsqueeze(-1)
                    .repeat(target_shape[0], 1, target_shape[2])
                )
            elif tensor.dim() == 2:
                tensor = tensor.unsqueeze(0).repeat(target_shape[0], 1, 1)
            elif tensor.dim() == 3:
                tensor = tensor.unsqueeze(0).repeat(target_shape[0], 1, 1, 1).squeeze(1)
        elif len(target_shape) == 4:
            tensor = tensor.unsqueeze(0).repeat(target_shape[0], 1, 1, 1)

        wp.copy(getattr(env.data_wp, field_name), wp.from_torch(tensor))

    # Handle contact struct fields
    for mj_field, wp_field in contact_fields:
        if hasattr(mj_data.contact, mj_field):
            source_data = getattr(mj_data.contact, mj_field)
            source_data_np = np.array(source_data, dtype=np.float32)
            tensor = torch.from_numpy(source_data_np).to(config.device)

            # Handle scalar fields
            if tensor.dim() == 0:
                tensor = tensor.unsqueeze(0)

            # Reshape for batched environments
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0).repeat(env.data_wp.nworld, 1)
            elif tensor.dim() == 2:
                tensor = tensor.unsqueeze(0).repeat(env.data_wp.nworld, 1, 1)
            elif tensor.dim() == 3:
                tensor = tensor.unsqueeze(0).repeat(env.data_wp.nworld, 1, 1, 1)

            # Get the destination field using nested attribute access
            dst_obj = env.data_wp
            for attr in wp_field.split("."):
                dst_obj = getattr(dst_obj, attr)
            wp.copy(dst_obj, wp.from_torch(tensor))

    # Handle efc fields - these are direct fields on mj_data
    for mj_field, wp_field in efc_fields:
        if hasattr(mj_data, mj_field):
            source_data = getattr(mj_data, mj_field)
            source_data_np = np.array(source_data, dtype=np.float32)
            tensor = torch.from_numpy(source_data_np).to(config.device)

            # Handle scalar fields
            if tensor.dim() == 0:
                tensor = tensor.unsqueeze(0)

            # Reshape for batched environments
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0).repeat(env.data_wp.nworld, 1)
            elif tensor.dim() == 2:
                tensor = tensor.unsqueeze(0).repeat(env.data_wp.nworld, 1, 1)
            elif tensor.dim() == 3:
                tensor = tensor.unsqueeze(0).repeat(env.data_wp.nworld, 1, 1, 1)

            # Get the destination field using nested attribute access
            dst_obj = env.data_wp
            for attr in wp_field.split("."):
                dst_obj = getattr(dst_obj, attr)
            wp.copy(dst_obj, wp.from_torch(tensor))

    return env


def copy_sample_state(
    config: Config, env: MJWPEnv, src_indices: torch.Tensor, dst_indices: torch.Tensor
):
    """Copy simulation state from source samples to destination samples.

    Args:
        config: Config
        env: MJWPEnv environment
        src_indices: Tensor of shape (n,) containing source sample indices
        dst_indices: Tensor of shape (n,) containing destination sample indices
    """
    # Convert to numpy for indexing
    src_idx = src_indices.cpu().numpy()
    dst_idx = dst_indices.cpu().numpy()

    # Get all state data as torch tensors
    qpos = wp.to_torch(env.data_wp.qpos)
    qvel = wp.to_torch(env.data_wp.qvel)
    qacc = wp.to_torch(env.data_wp.qacc)
    time_arr = wp.to_torch(env.data_wp.time)
    ctrl = wp.to_torch(env.data_wp.ctrl)
    act = wp.to_torch(env.data_wp.act)
    act_dot = wp.to_torch(env.data_wp.act_dot)
    qacc_warmstart = wp.to_torch(env.data_wp.qacc_warmstart)
    qfrc_applied = wp.to_torch(env.data_wp.qfrc_applied)
    xfrc_applied = wp.to_torch(env.data_wp.xfrc_applied)
    energy = wp.to_torch(env.data_wp.energy)
    mocap_pos = wp.to_torch(env.data_wp.mocap_pos)
    mocap_quat = wp.to_torch(env.data_wp.mocap_quat)
    xpos = wp.to_torch(env.data_wp.xpos)
    xquat = wp.to_torch(env.data_wp.xquat)
    xmat = wp.to_torch(env.data_wp.xmat)
    xipos = wp.to_torch(env.data_wp.xipos)
    ximat = wp.to_torch(env.data_wp.ximat)
    geom_xpos = wp.to_torch(env.data_wp.geom_xpos)
    geom_xmat = wp.to_torch(env.data_wp.geom_xmat)
    site_xpos = wp.to_torch(env.data_wp.site_xpos)
    site_xmat = wp.to_torch(env.data_wp.site_xmat)
    cacc = wp.to_torch(env.data_wp.cacc)
    cfrc_int = wp.to_torch(env.data_wp.cfrc_int)
    cfrc_ext = wp.to_torch(env.data_wp.cfrc_ext)
    sensordata = wp.to_torch(env.data_wp.sensordata)
    actuator_length = wp.to_torch(env.data_wp.actuator_length)
    actuator_velocity = wp.to_torch(env.data_wp.actuator_velocity)
    actuator_force = wp.to_torch(env.data_wp.actuator_force)
    ten_length = wp.to_torch(env.data_wp.ten_length)
    ten_velocity = wp.to_torch(env.data_wp.ten_velocity)

    # Copy from src to dst
    qpos[dst_idx] = qpos[src_idx]
    qvel[dst_idx] = qvel[src_idx]
    qacc[dst_idx] = qacc[src_idx]
    time_arr[dst_idx] = time_arr[src_idx]
    ctrl[dst_idx] = ctrl[src_idx]
    act[dst_idx] = act[src_idx]
    act_dot[dst_idx] = act_dot[src_idx]
    qacc_warmstart[dst_idx] = qacc_warmstart[src_idx]
    qfrc_applied[dst_idx] = qfrc_applied[src_idx]
    xfrc_applied[dst_idx] = xfrc_applied[src_idx]
    energy[dst_idx] = energy[src_idx]
    mocap_pos[dst_idx] = mocap_pos[src_idx]
    mocap_quat[dst_idx] = mocap_quat[src_idx]
    xpos[dst_idx] = xpos[src_idx]
    xquat[dst_idx] = xquat[src_idx]
    xmat[dst_idx] = xmat[src_idx]
    xipos[dst_idx] = xipos[src_idx]
    ximat[dst_idx] = ximat[src_idx]
    geom_xpos[dst_idx] = geom_xpos[src_idx]
    geom_xmat[dst_idx] = geom_xmat[src_idx]
    site_xpos[dst_idx] = site_xpos[src_idx]
    site_xmat[dst_idx] = site_xmat[src_idx]
    cacc[dst_idx] = cacc[src_idx]
    cfrc_int[dst_idx] = cfrc_int[src_idx]
    cfrc_ext[dst_idx] = cfrc_ext[src_idx]
    sensordata[dst_idx] = sensordata[src_idx]
    actuator_length[dst_idx] = actuator_length[src_idx]
    actuator_velocity[dst_idx] = actuator_velocity[src_idx]
    actuator_force[dst_idx] = actuator_force[src_idx]
    ten_length[dst_idx] = ten_length[src_idx]
    ten_velocity[dst_idx] = ten_velocity[src_idx]

    # Copy back to warp arrays
    wp.copy(env.data_wp.qpos, wp.from_torch(qpos))
    wp.copy(env.data_wp.qvel, wp.from_torch(qvel))
    wp.copy(env.data_wp.qacc, wp.from_torch(qacc))
    wp.copy(env.data_wp.time, wp.from_torch(time_arr))
    wp.copy(env.data_wp.ctrl, wp.from_torch(ctrl))
    wp.copy(env.data_wp.act, wp.from_torch(act))
    wp.copy(env.data_wp.act_dot, wp.from_torch(act_dot))
    wp.copy(env.data_wp.qacc_warmstart, wp.from_torch(qacc_warmstart))
    wp.copy(env.data_wp.qfrc_applied, wp.from_torch(qfrc_applied))
    wp.copy(env.data_wp.xfrc_applied, wp.from_torch(xfrc_applied))
    wp.copy(env.data_wp.energy, wp.from_torch(energy))
    wp.copy(env.data_wp.mocap_pos, wp.from_torch(mocap_pos))
    wp.copy(env.data_wp.mocap_quat, wp.from_torch(mocap_quat))
    wp.copy(env.data_wp.xpos, wp.from_torch(xpos))
    wp.copy(env.data_wp.xquat, wp.from_torch(xquat))
    wp.copy(env.data_wp.xmat, wp.from_torch(xmat))
    wp.copy(env.data_wp.xipos, wp.from_torch(xipos))
    wp.copy(env.data_wp.ximat, wp.from_torch(ximat))
    wp.copy(env.data_wp.geom_xpos, wp.from_torch(geom_xpos))
    wp.copy(env.data_wp.geom_xmat, wp.from_torch(geom_xmat))
    wp.copy(env.data_wp.site_xpos, wp.from_torch(site_xpos))
    wp.copy(env.data_wp.site_xmat, wp.from_torch(site_xmat))
    wp.copy(env.data_wp.cacc, wp.from_torch(cacc))
    wp.copy(env.data_wp.cfrc_int, wp.from_torch(cfrc_int))
    wp.copy(env.data_wp.cfrc_ext, wp.from_torch(cfrc_ext))
    wp.copy(env.data_wp.sensordata, wp.from_torch(sensordata))
    wp.copy(env.data_wp.actuator_length, wp.from_torch(actuator_length))
    wp.copy(env.data_wp.actuator_velocity, wp.from_torch(actuator_velocity))
    wp.copy(env.data_wp.actuator_force, wp.from_torch(actuator_force))
    wp.copy(env.data_wp.ten_length, wp.from_torch(ten_length))
    wp.copy(env.data_wp.ten_velocity, wp.from_torch(ten_velocity))


def _copy_state(src: mjwarp.Data, dst: mjwarp.Data):
    """Copy the state from src to dst

    TODO: this function is a temporary solution for domain randomization. A better way should be defining a new warp kernel to update simulation parameter accordingly.

    Args:
        src: mjwarp.Data
            the source data to be copied from
        dst: mjwarp.Data
            the destination data to be copied to
    """
    # Core state variables
    wp.copy(dst.qpos, src.qpos)
    wp.copy(dst.qvel, src.qvel)
    wp.copy(dst.qacc, src.qacc)
    wp.copy(dst.time, src.time)
    wp.copy(dst.ctrl, src.ctrl)
    wp.copy(dst.act, src.act)
    wp.copy(dst.act_dot, src.act_dot)
    wp.copy(dst.qacc_warmstart, src.qacc_warmstart)

    # Forces and applied forces
    wp.copy(dst.qfrc_applied, src.qfrc_applied)
    wp.copy(dst.xfrc_applied, src.xfrc_applied)

    # Energy tracking
    wp.copy(dst.energy, src.energy)

    # Mocap data
    wp.copy(dst.mocap_pos, src.mocap_pos)
    wp.copy(dst.mocap_quat, src.mocap_quat)

    # Spatial transformations
    wp.copy(dst.xpos, src.xpos)
    wp.copy(dst.xquat, src.xquat)
    wp.copy(dst.xmat, src.xmat)
    wp.copy(dst.xipos, src.xipos)
    wp.copy(dst.ximat, src.ximat)

    # Geometry positions
    wp.copy(dst.geom_xpos, src.geom_xpos)
    wp.copy(dst.geom_xmat, src.geom_xmat)
    wp.copy(dst.site_xpos, src.site_xpos)
    wp.copy(dst.site_xmat, src.site_xmat)

    # Camera and lighting (if present)
    if hasattr(src, "cam_xpos") and hasattr(dst, "cam_xpos"):
        wp.copy(dst.cam_xpos, src.cam_xpos)
        wp.copy(dst.cam_xmat, src.cam_xmat)
    if hasattr(src, "light_xpos") and hasattr(dst, "light_xpos"):
        wp.copy(dst.light_xpos, src.light_xpos)
        wp.copy(dst.light_xdir, src.light_xdir)

    # Body dynamics
    wp.copy(dst.cacc, src.cacc)
    wp.copy(dst.cfrc_int, src.cfrc_int)
    wp.copy(dst.cfrc_ext, src.cfrc_ext)

    # Sensor data
    wp.copy(dst.sensordata, src.sensordata)

    # Actuator data
    wp.copy(dst.actuator_length, src.actuator_length)
    wp.copy(dst.actuator_velocity, src.actuator_velocity)
    wp.copy(dst.actuator_force, src.actuator_force)

    # Tendon data
    wp.copy(dst.ten_length, src.ten_length)
    wp.copy(dst.ten_velocity, src.ten_velocity)

    # Contact struct - copy all fields
    wp.copy(dst.contact.dist, src.contact.dist)
    wp.copy(dst.contact.pos, src.contact.pos)
    wp.copy(dst.contact.frame, src.contact.frame)
    wp.copy(dst.contact.includemargin, src.contact.includemargin)
    wp.copy(dst.contact.friction, src.contact.friction)
    wp.copy(dst.contact.solref, src.contact.solref)
    wp.copy(dst.contact.solreffriction, src.contact.solreffriction)
    wp.copy(dst.contact.solimp, src.contact.solimp)
    wp.copy(dst.contact.dim, src.contact.dim)
    wp.copy(dst.contact.geom, src.contact.geom)
    wp.copy(dst.contact.efc_address, src.contact.efc_address)
    wp.copy(dst.contact.worldid, src.contact.worldid)

    # Constraint (efc) struct - copy all fields
    wp.copy(dst.efc.type, src.efc.type)
    wp.copy(dst.efc.id, src.efc.id)
    wp.copy(dst.efc.J, src.efc.J)
    wp.copy(dst.efc.pos, src.efc.pos)
    wp.copy(dst.efc.margin, src.efc.margin)
    wp.copy(dst.efc.D, src.efc.D)
    wp.copy(dst.efc.vel, src.efc.vel)
    wp.copy(dst.efc.aref, src.efc.aref)
    wp.copy(dst.efc.frictionloss, src.efc.frictionloss)
    wp.copy(dst.efc.force, src.efc.force)
    # Note: The workspace fields (Jaref, Ma, grad, etc.) are typically not needed for state transfer
    # as they are recomputed during solving, but include if needed:
    # wp.copy(dst.efc.Jaref, src.efc.Jaref)
    # wp.copy(dst.efc.Ma, src.efc.Ma)
    # wp.copy(dst.efc.grad, src.efc.grad)
    # ... (other workspace fields)
    #
    return dst
