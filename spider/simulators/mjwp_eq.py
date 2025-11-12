"""
Simulator for sampling with MuJoCo Warp (mjwarp) and support equality constraint (advanced features to enable contact guidancce).

This module provides a minimal MJWP backend that matches the sampling API used by
the generic optimizer pipeline. It intentionally keeps the implementation simple
and robust (no DR groups here; see legacy script for advanced features).
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
import mujoco
import warp as wp
import mujoco_warp as mjwarp
import loguru

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
    data_wp: mjwarp.Data
    # Device alias used for Warp allocations/launches (e.g., "cuda:1" or "cpu")
    device: str
    # Default handles
    default_model_wp: mjwarp.Model
    default_data_wp: mjwarp.Data
    default_graph: wp.ScopedCapture.Graph
    # Optional DR groups (lists aligned by index)
    dr_model_wps: list[mjwarp.Model]
    dr_data_wps: list[mjwarp.Data]
    dr_graphs: list[wp.ScopedCapture.Graph]
    # Current active group index: -1 for main, or [0..G-1] for DR groups
    active_dr_group: int
    active_annealing_group: int
    num_worlds: int
    # static data
    mocap_pos: torch.Tensor
    contact: torch.Tensor


def _compile_step(
    model_wp: mjwarp.Model, data_wp: mjwarp.Data
) -> wp.ScopedCapture.Graph:
    """
    Warm up and capture a CUDA graph that runs a single mjwarp.step.
    """

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
    model_cpu.opt.iterations = 20
    model_cpu.opt.ls_iterations = 50
    model_cpu.opt.o_solref = [0.02, 1.0]
    model_cpu.opt.o_solimp = [0.9, 0.95, 0.001, 0.5, 2]
    return model_cpu


def setup_env(config: Config, ref_data: tuple[torch.Tensor, ...]) -> MJWPEnv:
    """
    Setup and reset the environment backed by MJWP.
    Returns an MJWPEnv with captured graph.
    """
    qpos_ref, qvel_ref, ctrl_ref, contact_ref, contact_pos_ref = ref_data
    qpos_init = qpos_ref[0]
    mocap_pos = contact_pos_ref

    # CPU model/data
    model_cpu = setup_mj_model(config)
    data_cpu = mujoco.MjData(model_cpu)
    # Seed initial state
    qpos_np = qpos_init.detach().cpu().numpy()
    data_cpu.qpos[:] = qpos_np
    data_cpu.qvel[:] = 0.0
    data_cpu.ctrl[:] = qpos_np[: -config.nq_obj]
    mujoco.mj_step(model_cpu, data_cpu)

    # Move to Warp (batched worlds)
    # Set Warp default device to match config to ensure kernels/modules load on it
    wp.set_device(str(config.device))
    # Build default model/data/graph on the configured device
    dev = str(config.device)
    with wp.ScopedDevice(dev):
        default_model_wp = mjwarp.put_model(model_cpu)
        default_data_wp = mjwarp.put_data(
            model_cpu,
            data_cpu,
            nworld=int(config.num_samples),
            nconmax=int(config.nconmax_per_env * config.num_samples),
            njmax=int(config.njmax_per_env),
        )
        default_graph = _compile_step(default_model_wp, default_data_wp)

        # Build optional DR groups: separate cpu models to avoid parameter coupling
        dr_model_wps: list[mjwarp.Model] = []
        dr_data_wps: list[mjwarp.Data] = []
        dr_graphs: list[
            list[wp.ScopedCapture.Graph]
        ] = []  # its size should be (config.max_num_iterations, config.num_dr)

        # Precompute per-group values for pair params
        def lin(min_v, max_v, dim, num_groups):
            a_min = (
                np.array(min_v, dtype=np.float32)
                if dim > 1
                else np.array([min_v], dtype=np.float32)
            )
            a_max = (
                np.array(max_v, dtype=np.float32)
                if dim > 1
                else np.array([max_v], dtype=np.float32)
            )
            vals = []
            for gi in range(num_groups):
                alpha = 0.0 if num_groups == 1 else gi / (num_groups - 1)
                vals.append(((1 - alpha) * a_min + alpha * a_max).astype(np.float32))
            return np.stack(vals, axis=0)

        # create model function
        def create_model(
            config: Config,
            qpos_init: torch.Tensor,
            margin: float = None,
            friction: list[float] = None,
            eq_solref: list[float] = None,
            eq_solimp: list[float] = None,
        ):
            m_cpu = setup_mj_model(config)
            if margin is not None:
                m_cpu.pair_margin = margin
            if friction is not None:
                m_cpu.pair_friction = friction
            if eq_solref is not None:
                m_cpu.eq_solref = eq_solref
            if eq_solimp is not None:
                m_cpu.eq_solimp = eq_solimp
            d_cpu = mujoco.MjData(m_cpu)
            d_cpu.qpos[:] = qpos_init
            d_cpu.qvel[:] = 0.0
            d_cpu.ctrl[:] = qpos_init[: -config.nq_obj]
            mujoco.mj_step(m_cpu, d_cpu)
            # allocate and capture on the configured device
            m_wp = mjwarp.put_model(m_cpu)
            d_wp = mjwarp.put_data(
                m_cpu,
                d_cpu,
                nworld=int(config.num_samples),
                nconmax=int(config.nconmax_per_env * config.num_samples),
                njmax=int(config.njmax_per_env),
            )
            return m_wp, d_wp, _compile_step(m_wp, d_wp)

        # DR parameter schedule
        npair = int(model_cpu.npair)
        margin_vals = lin(
            config.dr_pair_margin_min, config.dr_pair_margin_max, 1, config.num_dr
        )
        friction_vals = lin(
            config.dr_pair_friction_min, config.dr_pair_friction_max, 5, config.num_dr
        )
        # Dynamics annealing schedule
        config.num_dyn = min(config.num_dyn, config.max_num_iterations - 1)
        if config.num_dyn > 1:
            eq_solref_vals = lin(
                config.eq_solref_min, config.eq_solref_max, 2, config.num_dyn
            )
            eq_solimp_vals = lin(
                config.eq_solimp_min, config.eq_solimp_max, 5, config.num_dyn
            )

        for iteration in range(config.max_num_iterations):
            iteration_graphs = []
            iteration_models = []
            iteration_datas = []

            # Determine if we're in the annealing phase
            use_annealing = iteration < config.num_dyn and config.num_dyn > 1

            for dr_group in range(config.num_dr):
                if config.num_dr == 1 and not use_annealing:
                    # No DR, no annealing - use default
                    model_wp = default_model_wp
                    data_wp = default_data_wp
                    graph = default_graph
                else:
                    # Create custom model with appropriate parameters
                    kwargs = {}
                    if config.num_dr > 1:
                        kwargs["margin"] = margin_vals[dr_group]
                        kwargs["friction"] = friction_vals[dr_group]
                    if use_annealing:
                        kwargs["eq_solref"] = eq_solref_vals[iteration]
                        kwargs["eq_solimp"] = eq_solimp_vals[iteration]

                    if kwargs:  # Only create if we have parameters to modify
                        model_wp, data_wp, graph = create_model(
                            config, data_cpu.qpos, **kwargs
                        )
                    else:
                        model_wp = default_model_wp
                        data_wp = default_data_wp
                        graph = default_graph

                iteration_graphs.append(graph)
                iteration_models.append(model_wp)
                iteration_datas.append(data_wp)

            dr_graphs.append(iteration_graphs)
            dr_model_wps.append(iteration_models)
            dr_data_wps.append(iteration_datas)

        assert len(dr_graphs) == config.max_num_iterations, (
            "DR groups should be the same size as max_num_iterations"
        )
        assert len(dr_graphs[0]) == config.num_dr, (
            "DR groups should be the same size as num_dr"
        )
        loguru.logger.info(
            f"Created {config.max_num_iterations} annealing groups and {config.num_dr} DR groups"
        )

    # Initialize env; default active is main
    return MJWPEnv(
        model_cpu=model_cpu,
        data_cpu=data_cpu,
        data_wp=default_data_wp,
        device=dev,
        default_model_wp=default_model_wp,
        default_data_wp=default_data_wp,
        default_graph=default_graph,
        dr_model_wps=dr_model_wps,
        dr_data_wps=dr_data_wps,
        dr_graphs=dr_graphs,
        active_dr_group=-1,
        active_annealing_group=-1,
        num_worlds=int(config.num_samples),
        mocap_pos=mocap_pos,
        contact=contact_ref,
    )


def _weight_diff_qpos(config: Config) -> torch.Tensor:
    w = torch.ones(config.nv, device=config.device)
    if config.hand_type == "bimanual":
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
    elif config.hand_type in ["right", "left"]:
        w[:3] = config.base_pos_rew_scale
        w[3:6] = config.base_rot_rew_scale
        w[6 : config.nu] = config.joint_rew_scale
        w[-6:-3] = config.pos_rew_scale
        w[-3:] = config.rot_rew_scale
    elif config.hand_type in ["CMU", "DanceDB"]:
        w[:3] = config.pos_rew_scale
        w[3:6] = config.rot_rew_scale
    else:
        raise ValueError(f"Invalid hand_type: {config.hand_type}")
    return w


def _diff_qpos(
    config: Config, qpos_sim: torch.Tensor, qpos_ref: torch.Tensor
) -> torch.Tensor:
    """
    Compute the difference between qpos_sim and qpos_ref
    """
    batch_size = qpos_sim.shape[0]
    qpos_diff = torch.zeros((batch_size, config.nv), device=config.device)
    if config.hand_type == "bimanual":
        # joint
        qpos_diff[:, :-12] = qpos_sim[:, :-14] - qpos_ref[:, :-14]
        # position
        qpos_diff[:, -12:-9] = qpos_sim[:, -14:-11] - qpos_ref[:, -14:-11]
        qpos_diff[:, -6:-3] = qpos_sim[:, -7:-4] - qpos_ref[:, -7:-4]
        # rotation
        qpos_diff[:, -9:-6] = quat_sub(qpos_sim[:, -11:-7], qpos_ref[:, -11:-7])
        qpos_diff[:, -3:] = quat_sub(qpos_sim[:, -4:], qpos_ref[:, -4:])
    elif config.hand_type in ["right", "left"]:
        # joint
        qpos_diff[:, :-6] = qpos_sim[:, :-7] - qpos_ref[:, :-7]
        # position
        qpos_diff[:, -6:-3] = qpos_sim[:, -7:-4] - qpos_ref[:, -7:-4]
        # rotation
        qpos_diff[:, -3:] = quat_sub(qpos_sim[:, -4:], qpos_ref[:, -4:])
    elif config.hand_type in ["CMU", "DanceDB"]:
        # joint
        qpos_diff[:, 7:] = qpos_sim[:, 7:] - qpos_ref[:, 7:]
        # position
        qpos_diff[:, :3] = qpos_sim[:, :3] - qpos_ref[:, :3]
        # rotation
        qpos_diff[:, 3:6] = quat_sub(qpos_sim[:, 3:7], qpos_ref[:, 3:7])
    return qpos_diff


def get_reward(
    config: Config,
    env: MJWPEnv,
    ref: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """
    Non-terminal step reward for MJWP batched worlds.
    ref is a tuple: (qpos_ref, qvel_ref, ctrl_ref, contact_ref, contact_pos_ref)
    Returns (N,)
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
    cost_qpos = torch.norm(delta_qpos, p=2, dim=1)
    cost_qvel = config.vel_rew_scale * torch.norm(qvel_sim - qvel_ref, p=2, dim=1)
    # Optional contact reward (requires site_xpos availability and configured ids)
    cost_contact = torch.zeros_like(cost_qpos, device=config.device)
    if config.contact_rew_scale > 0.0:
        site_xpos_torch = wp.to_torch(env.data_wp.site_xpos)
        track_sites_pos = site_xpos_torch[:, config.track_site_ids]
        ref_sites_pos = site_xpos_torch[:, config.ref_site_ids]
        delta_pos = ref_sites_pos - track_sites_pos
        dist = torch.norm(delta_pos, p=2, dim=-1)
        dist_masked = dist * contact_ref
        cost_contact = config.contact_rew_scale * dist_masked.sum(dim=-1)
    return -(cost_qpos + cost_qvel + cost_contact)


def get_terminal_reward(
    config: Config,
    env: MJWPEnv,
    ref_slice: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """
    Terminal reward focusing on object tracking.
    """
    # return config.terminal_rew_scale * get_reward(config, env, ref_slice)
    qpos_ref, qvel_ref, ctrl_ref, contact_ref, _ = ref_slice
    qpos_sim = wp.to_torch(env.data_wp.qpos)
    qpos_weight = torch.zeros(qpos_sim.shape[1], device=config.device)
    if config.hand_type == "bimanual":
        qpos_weight[-14:-11] = config.pos_rew_scale
        qpos_weight[-11:-7] = config.rot_rew_scale
        qpos_weight[-7:-4] = config.pos_rew_scale
        qpos_weight[-4:] = config.rot_rew_scale
    elif config.hand_type in ["right", "left"]:
        qpos_weight[-7:-4] = config.pos_rew_scale
        qpos_weight[-4:] = config.rot_rew_scale
    elif config.hand_type in ["CMU", "DanceDB"]:
        qpos_weight[:3] = config.pos_rew_scale
        qpos_weight[3:7] = config.rot_rew_scale
    else:
        raise ValueError(f"Invalid hand_type: {config.hand_type}")
    delta_qpos = (qpos_sim - qpos_ref) * qpos_weight
    cost_object = config.terminal_rew_scale * torch.sum(delta_qpos**2, dim=1)
    return -cost_object


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
    """
    Return per-world trace points used for visualization. Minimal default returns
    an empty trace set of shape (N, 0, 3) when not configured.
    """
    site_xpos = wp.to_torch(env.data_wp.site_xpos)  # (N, nsite, 3)
    return site_xpos[:, config.trace_site_ids, :]


def save_state(env: MJWPEnv):
    """
    Clone the essential set of Warp arrays to restore later.
    Includes core state variables and key derived quantities.
    """
    qpos = wp.clone(env.data_wp.qpos)
    qvel = wp.clone(env.data_wp.qvel)
    qacc = wp.clone(env.data_wp.qacc)
    time_arr = wp.clone(env.data_wp.time)
    ctrl = wp.clone(env.data_wp.ctrl) if hasattr(env.data_wp, "ctrl") else None
    act = wp.clone(env.data_wp.act) if hasattr(env.data_wp, "act") else None
    act_dot = wp.clone(env.data_wp.act_dot) if hasattr(env.data_wp, "act_dot") else None
    mocap_pos = (
        wp.clone(env.data_wp.mocap_pos) if hasattr(env.data_wp, "mocap_pos") else None
    )
    mocap_quat = (
        wp.clone(env.data_wp.mocap_quat) if hasattr(env.data_wp, "mocap_quat") else None
    )
    energy = wp.clone(env.data_wp.energy) if hasattr(env.data_wp, "energy") else None
    return (
        qpos,
        qvel,
        qacc,
        time_arr,
        ctrl,
        act,
        act_dot,
        mocap_pos,
        mocap_quat,
        energy,
    )


def load_state(env: MJWPEnv, state):
    qpos, qvel, qacc, time_arr, ctrl, act, act_dot, mocap_pos, mocap_quat, energy = (
        state
    )
    wp.copy(env.data_wp.qpos, qpos)
    wp.copy(env.data_wp.qvel, qvel)
    wp.copy(env.data_wp.qacc, qacc)
    wp.copy(env.data_wp.time, time_arr)
    if ctrl is not None and hasattr(env.data_wp, "ctrl"):
        wp.copy(env.data_wp.ctrl, ctrl)
    if act is not None and hasattr(env.data_wp, "act"):
        wp.copy(env.data_wp.act, act)
    if act_dot is not None and hasattr(env.data_wp, "act_dot"):
        wp.copy(env.data_wp.act_dot, act_dot)
    if mocap_pos is not None and hasattr(env.data_wp, "mocap_pos"):
        wp.copy(env.data_wp.mocap_pos, mocap_pos)
    if mocap_quat is not None and hasattr(env.data_wp, "mocap_quat"):
        wp.copy(env.data_wp.mocap_quat, mocap_quat)
    if energy is not None and hasattr(env.data_wp, "energy"):
        wp.copy(env.data_wp.energy, energy)
    return env


def _copy_state(src: mjwarp.Data, dst: mjwarp.Data):
    """
    Copy the state from src to dst

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


def update_mocap_pos(
    config: Config,
    env: MJWPEnv,
):
    """
    Update mocap pos for contact reference

    The mocap named `ref_hand_right_index_tip` should be set to the site `track_object_right_index_tip` position
    The mocap named `ref_object_right_index_tip` should be set to the site `track_hand_right_middle_tip` position
    i.e., the rule is, replace the `ref_` with `track_`, replace `hand` with `object` or vice versa

    to achieve that, we first create a index map for mocap and site, such that we can do `mj_data.mocap_pos[config.ref_mocap_ids] = site_xpos[config.track_site_ids]`


    Args:
        config: Config
            ref_mocap_ids: list[int], the ids of the sites to be tracked
            track_site_ids: list[int], the ids of the sites to be tracked
        env: MJWPEnv
            contact_ref: torch.Tensor, shape (N, N_mocap), the contact status of all tracking sites, 1 if in contact, 0 otherwise. If 1, the mocap pos is set to the site pos. otherwise, the mocap pos is set to the reference trajectory, i.e. mocap_pos_ref
            mocap_pos_ref: torch.Tensor, shape (N, N_mocap, 3), the reference trajectory of all tracking sites
            dwp: mjwarp.Data: the data_wp to be updated
    """
    # get data from ene
    dwp = env.data_wp

    # get discrete time to get reference contact and mocap pos
    time = wp.to_torch(dwp.time)
    discrete_time = (time / env.model_cpu.opt.timestep).int()  # (N)
    ref_discrete_time = torch.clamp(discrete_time, max=env.mocap_pos.shape[0] - 2) + 1
    mocap_pos_ref = env.mocap_pos[
        ref_discrete_time
    ]  # mocap reference without contact, shape (N, ntrack, 3)
    contact_ref = env.contact[ref_discrete_time]  # (N, ntrack)

    # mocap reference with contact, which is the contact site
    site_xpos = wp.to_torch(dwp.site_xpos)  # (N, nsite, 3)
    contact_site_pos = site_xpos[:, config.track_site_ids]  # (N, ntrack, 3)
    mocap_pos_contact = torch.zeros_like(mocap_pos_ref, device=config.device)
    mocap_pos_contact[:, config.ref_mocap_ids] = contact_site_pos  # (N, ntrack, 3)

    # get mocap pos according to contact
    mocap_pos = torch.where(
        contact_ref.unsqueeze(-1) > 0.5, mocap_pos_contact, mocap_pos_ref
    )

    # set mocap pos
    wp.copy(dwp.mocap_pos, wp.from_torch(mocap_pos))

    return dwp


def step_env(config: Config, env: MJWPEnv, ctrl_mujoco: torch.Tensor):
    """
    Step all worlds with provided MuJoCo-format controls of shape (N, nu).
    """
    if ctrl_mujoco.dim() == 1:
        ctrl_mujoco = ctrl_mujoco.unsqueeze(0).repeat(env.num_worlds, 1)
    # Step the selected group (default or DR) and then copy results into unified data_wp
    dwp = env.dr_data_wps[env.active_annealing_group][env.active_dr_group]
    graph = env.dr_graphs[env.active_annealing_group][env.active_dr_group]
    # Ensure we operate on the correct CUDA context/device
    with wp.ScopedDevice(env.device):
        # set mocap pos
        env.data_wp = update_mocap_pos(config, env)
        # data_wp -> dwp
        dwp = _copy_state(env.data_wp, dwp)
        # step control
        wp.copy(dwp.ctrl, wp.from_torch(ctrl_mujoco.to(torch.float32)))
        wp.capture_launch(graph)
        # dwp -> data_wp
        env.data_wp = _copy_state(dwp, env.data_wp)


def save_env_params(env: MJWPEnv):
    """
    Save the current simulation parameters.
    """
    # Only record which group is active; parameters are embedded in separate models
    return {
        "active_dr_group": env.active_dr_group,
        "active_annealing_group": env.active_annealing_group,
    }


def load_env_params(env: MJWPEnv, env_param: dict):
    """
    Load the simulation parameters.

    Parameters to be updated:
    - pair_margin
    - pair_friction
    - eq_solref
    - eq_solimp
    """
    env.active_dr_group = env_param["active_dr_group"]
    env.active_annealing_group = env_param["active_annealing_group"]
    return env


def sync_env(config: Config, env: MJWPEnv, mj_data: mujoco.MjData):
    """
    sync state from mj_data to env.data_wp
    """
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
