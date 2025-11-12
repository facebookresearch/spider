"""
Simulator for sampling.

NOTE: this requires external isaacgym installation.

Required functions:
setup_env: setup the environment (takes in mujoco ctrl, remember to remap the ctrl to environment ctrl)
step_env: step the environment
get_reward: get the reward
get_terminal_reward: get the terminal reward
get_qpos: get the qpos (i.e. mujoco qpos)
set_qpos: set the qpos
get_trace: get the trace (e.g. fingertip positions)
save_state: save the state of the environment
load_state: load the state of the environment
sync_env: sync the state of the environment
"""

from __future__ import annotations
import os
import sys
import numpy as np
import torch
import loguru
from isaacgym import gymapi, gymtorch
import mujoco

# our package
from spider.config import Config
from spider.math import quat_xyzw2wxyz, quat_wxyz2xyzw

# third party, where you should put the isaacgym
sys.path.append("/home/pcy/Research/code/human2robot/trajdex")
from configs.base import TrajDexConfig, EnvConfig, ControllerConfig
from trajdex.tasks.polytraj import PolyTraj

# ------------------------------------------------------------
# Conversion functions
# ------------------------------------------------------------


def ctrl_mujoco2isaac(ctrl_mujoco: torch.Tensor) -> torch.Tensor:
    """
    Map a standard 44-d control/qpos (right-first, original finger order) to
    env action/qpos order (left-first, Isaac finger order: index, thumb, middle, ring).

    Supports input shape (..., 44) and returns same leading dims with 44 at the end.
    """
    last_dim = ctrl_mujoco.shape[-1]
    assert last_dim == 44, f"Expected last dim 44, got {last_dim}"
    # Split
    right_base = ctrl_mujoco[..., 0:6]
    right_finger = ctrl_mujoco[..., 6:22]
    left_base = ctrl_mujoco[..., 22:28]
    left_finger = ctrl_mujoco[..., 28:44]
    # Right finger: original chunks [index(0:4), middle(4:8), ring(8:12), thumb(12:16)]
    r_index = right_finger[..., 0:4]
    r_middle = right_finger[..., 4:8]
    r_ring = right_finger[..., 8:12]
    r_thumb = right_finger[..., 12:16]
    right_isaac = torch.cat([r_index, r_thumb, r_middle, r_ring], dim=-1)
    # Left finger: original chunks [ring(0:4), middle(4:8), index(8:12), thumb(12:16)]
    l_ring = left_finger[..., 0:4]
    l_middle = left_finger[..., 4:8]
    l_index = left_finger[..., 8:12]
    l_thumb = left_finger[..., 12:16]
    left_isaac = torch.cat([l_index, l_thumb, l_middle, l_ring], dim=-1)
    # Env order is left first, then right
    ctrl_isaac = torch.cat([left_base, left_isaac, right_base, right_isaac], dim=-1)
    return ctrl_isaac


def ctrl_isaac2mujoco(ctrl_isaac: torch.Tensor) -> torch.Tensor:
    """
    Map isaac order into mujoco order, inverse version of mujoco2isaac
    """
    last_dim = ctrl_isaac.shape[-1]
    assert last_dim == 44, f"Expected last dim 44, got {last_dim}"
    # Split env order: left first, then right
    left_base = ctrl_isaac[..., 0:6]
    left_finger_isaac = ctrl_isaac[..., 6:22]
    right_base = ctrl_isaac[..., 22:28]
    right_finger_isaac = ctrl_isaac[..., 28:44]

    # Fingers in Isaac order: [index(0:4), thumb(4:8), middle(8:12), ring(12:16)]
    li = left_finger_isaac[..., 0:4]
    lt = left_finger_isaac[..., 4:8]
    lm = left_finger_isaac[..., 8:12]
    lr = left_finger_isaac[..., 12:16]

    ri = right_finger_isaac[..., 0:4]
    rt = right_finger_isaac[..., 4:8]
    rm = right_finger_isaac[..., 8:12]
    rr = right_finger_isaac[..., 12:16]

    # Standard (MuJoCo) finger order:
    #   - Right: [index, middle, ring, thumb]
    #   - Left:  [ring, middle, index, thumb]
    right_finger_mujoco = torch.cat([ri, rm, rr, rt], dim=-1)
    left_finger_mujoco = torch.cat([lr, lm, li, lt], dim=-1)

    # Standard overall order is right-first then left
    ctrl_mujoco = torch.cat(
        [right_base, right_finger_mujoco, left_base, left_finger_mujoco], dim=-1
    )
    return ctrl_mujoco


# ------------------------------------------------------------
# Key functions
# ------------------------------------------------------------


def setup_env(config: Config, qpos_init: torch.Tensor):
    """
    Setup and reset the environment

    Args:
        config: Config

    Returns:
        env: Environment
    """
    assert config.robot_type == "allegro", (
        f"Only Allegro is supported for now for isaac, robot type is {config.robot_type}"
    )
    assert config.hand_type == "bimanual", (
        f"Only bimanual is supported for now for isaac, hand type is {config.hand_type}"
    )

    # setup the environment
    env = PolyTraj(
        TrajDexConfig(
            env=EnvConfig(
                env_name="PolyTraj",
                arm_setting="float",
                num_envs=config.num_samples,
                num_act=44,
                clip_act=100,
            ),
            controller=ControllerConfig(
                torque_control=False,
                controlFrequencyInv=6,
                p_gain=3.0,
                d_gain=0.1,
            ),
            headless=not "isaac" in config.viewer,
        )
    )
    env.use_absolute_target = True
    env.disable_reset = True

    # get finger tip handles for trace
    env.left_hand_tip_link_handles = [
        env.gym.find_asset_rigid_body_index(env.left_hand_asset, name)
        for name in ["link_3.0_tip", "link_15.0_tip", "link_7.0_tip", "link_11.0_tip"]
    ]
    left_hand_link_num = env.gym.get_asset_rigid_body_count(env.left_hand_asset)
    env.right_hand_tip_link_handles = [
        env.gym.find_asset_rigid_body_index(env.right_hand_asset, name)
        + left_hand_link_num
        for name in ["link_3.0_tip", "link_15.0_tip", "link_7.0_tip", "link_11.0_tip"]
    ]

    # set the initial qpos
    set_qpos(config, env, qpos_init)

    return env


def get_reward(
    config: Config,
    env,
    ref: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """
    Get the reward
    Args:
        data_wp: Warp data
        qpos_ref: Reference qpos, shape (nq)
        qvel_ref: Reference qvel, shape (nv)
    Returns:
        Reward, shape (num_samples,)
    """
    qpos_ref, qvel_ref, ctrl_ref, contact_ref, contact_pos_ref = ref
    qpos_sim = get_qpos(config, env)
    qvel_sim = get_qvel(config, env)
    delta_qpos = qpos_sim - qpos_ref
    # more weights to object
    qpos_weight = torch.ones(qpos_sim.shape[1], device=qpos_sim.device)
    if config.hand_type == "bimanual":
        half_dof = config.nu // 2
        # hand reward
        qpos_weight[:3] = config.base_pos_rew_scale
        qpos_weight[3:6] = config.base_rot_rew_scale
        qpos_weight[6:half_dof] = config.joint_rew_scale
        qpos_weight[half_dof : half_dof + 3] = config.base_pos_rew_scale
        qpos_weight[half_dof + 3 : half_dof + 6] = config.base_rot_rew_scale
        qpos_weight[half_dof + 6 : config.nu] = config.joint_rew_scale
        # object reward
        qpos_weight[-14:-11] = config.pos_rew_scale
        qpos_weight[-11:-7] = config.rot_rew_scale
        qpos_weight[-7:-4] = config.pos_rew_scale
        qpos_weight[-4:] = config.rot_rew_scale
    elif config.hand_type in ["right", "left"]:
        # hand reward
        qpos_weight[:3] = config.base_pos_rew_scale
        qpos_weight[3:6] = config.base_rot_rew_scale
        qpos_weight[6 : config.nu] = config.joint_rew_scale
        # object reward
        qpos_weight[-7:-4] = config.pos_rew_scale
        qpos_weight[-4:] = config.rot_rew_scale
    elif config.hand_type in ["CMU", "DanceDB"]:
        # give base more weight
        qpos_weight[:3] = config.pos_rew_scale
        qpos_weight[3:7] = config.rot_rew_scale
    else:
        raise ValueError(f"Invalid hand_type: {config.hand_type}")
    delta_qpos *= qpos_weight
    # TODO: special case for quaternion
    cost_qpos = torch.norm(delta_qpos, p=2, dim=1)
    cost_qvel = config.vel_rew_scale * torch.norm(qvel_sim - qvel_ref, p=2, dim=1)
    cost_contact = torch.zeros_like(cost_qpos, device=cost_qpos.device)
    # TODO: add contact reward in isaac
    return -(cost_qpos + cost_qvel + cost_contact)


def get_terminal_reward(
    config: Config,
    env,
    ref_slice: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """
    Get the reward
    Args:
        data_wp: Warp data
        qpos_ref: Reference qpos, shape (nq)
        qvel_ref: Reference qvel, shape (nv)
    Returns:
        Reward, shape (num_samples,)
    """
    qpos_sim = get_qpos(config, env)
    qvel_sim = get_qvel(config, env)
    qpos_ref, qvel_ref, ctrl_ref, contact_ref, contact_pos_ref = ref_slice
    delta_qpos = qpos_sim - qpos_ref
    qpos_weight = torch.zeros(qpos_sim.shape[1], device=qpos_sim.device)
    if config.hand_type == "bimanual":
        # object reward
        qpos_weight[-14:-11] = config.pos_rew_scale
        qpos_weight[-11:-7] = config.rot_rew_scale
        qpos_weight[-7:-4] = config.pos_rew_scale
        qpos_weight[-4:] = config.rot_rew_scale
    elif config.hand_type in ["right", "left"]:
        # object reward
        qpos_weight[-7:-4] = config.pos_rew_scale
        qpos_weight[-4:] = config.rot_rew_scale
    elif config.hand_type in ["CMU", "DanceDB"]:
        # give base more weight
        qpos_weight[:3] = config.pos_rew_scale
        qpos_weight[3:7] = config.rot_rew_scale
    else:
        raise ValueError(f"Invalid hand_type: {config.hand_type}")
    delta_qpos *= qpos_weight
    cost_object = 100.0 * torch.norm(delta_qpos, p=2, dim=1)
    return -cost_object


def get_qpos(config: Config, env) -> torch.Tensor:
    """
    get mujoco qpos from isaac gym

    Args:
        config: Config
        env: Environment

    Returns:
        qpos: Mujoco qpos, shape (N, nq)
    """
    # Get all environments' dof positions (N, num_dofs)
    hand_qpos_mj = ctrl_isaac2mujoco(env.dof_pos)

    # Objects: convert xyzw -> wxyz, order right then left
    # env.right_object_pos shape: (N, 3)
    # env.right_object_rot shape: (N, 4) in xyzw format
    right_pos = env.right_object_pos  # (N, 3)
    right_rot_xyzw = env.right_object_rot  # (N, 4)
    right_rot_wxyz = quat_xyzw2wxyz(right_rot_xyzw)  # (N, 4)

    left_pos = env.left_object_pos  # (N, 3)
    left_rot_xyzw = env.left_object_rot  # (N, 4)
    left_rot_wxyz = quat_xyzw2wxyz(left_rot_xyzw)  # (N, 4)

    obj_tail = torch.cat(
        [right_pos, right_rot_wxyz, left_pos, left_rot_wxyz], dim=1
    )  # (N, 14)

    qpos = torch.cat([hand_qpos_mj, obj_tail], dim=1)  # (N, nq)
    return qpos


def set_qpos(config: Config, env, qpos: torch.Tensor):
    """
    Set the qpos of the environment
    """
    # Hands: map to env action/qpos order (left-first, Isaac finger order)
    hand_std_44 = qpos[..., :44]
    hand_env_44 = ctrl_mujoco2isaac(hand_std_44)
    left_hand_init_22 = hand_env_44[..., 0:22]
    right_hand_init_22 = hand_env_44[..., 22:44]
    env.left_hand_init_pose[:] = left_hand_init_22.to(env.device)
    env.right_hand_init_pose[:] = right_hand_init_22.to(env.device)

    # Objects: convert quat from wxyz (data) -> xyzw (Isaac)
    # Right object
    right_pos_xyz = qpos[..., 44:47]
    right_quat_wxyz = qpos[..., 47:51]
    right_quat_xyzw = torch.cat([right_quat_wxyz[1:4], right_quat_wxyz[0:1]], dim=-1)
    env.right_obj_init_pose[:] = torch.cat([right_pos_xyz, right_quat_xyzw], dim=-1).to(
        env.device
    )
    # Left object
    left_pos_xyz = qpos[..., 51:54]
    left_quat_wxyz = qpos[..., 54:58]
    left_quat_xyzw = torch.cat([left_quat_wxyz[1:4], left_quat_wxyz[0:1]], dim=-1)
    env.left_obj_init_pose[:] = torch.cat([left_pos_xyz, left_quat_xyzw], dim=-1).to(
        env.device
    )

    env.reset()


def get_qvel(config: Config, env) -> torch.Tensor:
    """
    get mujoco qvel from isaac gym
    Args:
        config: Config
        env: Environment
    Returns:
        qvel: Mujoco qvel, shape (N, nv)
    """
    return torch.zeros((config.num_samples, config.nv), device=env.device)


def get_trace(config: Config, env) -> dict:
    """
    Get the trace
    Args:
        config: Config
        env: Environment
    Returns:
        trace: Trace, shape (N, n_trace, 3)
    """
    # get object positions
    right_pos = env.right_object_pos.unsqueeze(1)  # (N, 1, 3)
    left_pos = env.left_object_pos.unsqueeze(1)  # (N, 1, 3)
    # get finger tip positions
    right_hand_pos = env.link_state[:, env.right_hand_tip_link_handles, 0:3]
    left_hand_pos = env.link_state[:, env.left_hand_tip_link_handles, 0:3]
    return torch.concat(
        [right_pos, left_pos, right_hand_pos, left_hand_pos], dim=1
    )  # (num_samples, num_traces_sites, 3)


def step_env(config, env, ctrl_mujoco: torch.Tensor):
    """
    Step the environment
    """
    ctrl_isaac = ctrl_mujoco2isaac(ctrl_mujoco)
    env.step(ctrl_isaac)


def save_state(env):
    """
    Save the state of the environment
    """
    saved_actor_root_state = env.actor_root_state.clone()
    saved_dof_state = env.dof_state.clone()
    return (saved_actor_root_state, saved_dof_state)


def load_state(env, state):
    """
    Load the state of the environment
    """
    saved_actor_root_state, saved_dof_state = state
    env.actor_root_state[:] = saved_actor_root_state
    env.dof_state[:] = saved_dof_state
    # push restored tensors back to simulator and refresh
    env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))
    env.gym.set_actor_root_state_tensor(
        env.sim, gymtorch.unwrap_tensor(env.actor_root_state)
    )
    env._refresh_gym()
    return env


def save_env_params(env):
    """
    Save the environment parameters
    """
    return {}


def load_env_params(env, env_param: dict):
    """
    Load the environment parameters
    """
    return env


def sync_env(config: Config, env, mj_data: mujoco.MjData):
    """
    boardcast the state of the environment to all environments
    """
    del mj_data  # for isaac, use board cast mode, i.e. only use env state

    # get first env state
    first_actor_root_state = env.actor_root_state.view(env.num_envs, -1, 13)[
        0:1
    ].clone()  # shape (1, num_actors, 13)
    first_dof_state = env.dof_state[0:1].clone()  # shape (1, num_dofs, 2)

    # repeat to all envs
    broadcasted_actor_root_state = first_actor_root_state.repeat(env.num_envs, 1, 1)
    broadcasted_dof_state = first_dof_state.repeat(env.num_envs, 1, 1)

    # set to all envs
    env.actor_root_state[:] = broadcasted_actor_root_state.view(-1, 13)
    env.dof_state[:] = broadcasted_dof_state

    # push tensors back to simulator and refresh
    env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))
    env.gym.set_actor_root_state_tensor(
        env.sim, gymtorch.unwrap_tensor(env.actor_root_state)
    )
    env._refresh_gym()

    return env
