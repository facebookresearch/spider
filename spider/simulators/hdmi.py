"""Simulator for sampling with HDMI based on mjlab.

Reference: https://github.com/LeCAR-Lab/HDMI

This module provides humanoid whole-body retargeting support with SPIDER.
"""

from __future__ import annotations

import os

import mujoco
import mujoco_warp as mjwarp
import torch
import warp as wp
from omegaconf import OmegaConf
from tensordict import TensorDict

from spider.config import Config
from spider.simulators.mjwp import _broadcast_state, _copy_state

# --
# Key functions
# --


def setup_env(config: Config, ref_data: tuple[torch.Tensor, ...]):
    """Setup and reset the environment backed by mjlab.

    Returns a SimpleEnv instance.
    """
    # reference data is stored in environment
    del ref_data

    # Import HDMI dependencies
    import active_adaptation
    from active_adaptation.envs import SimpleEnv

    # Use active_adaptation module path to find HDMI directory
    hdmi_dir = os.path.dirname(os.path.dirname(active_adaptation.__file__))

    # Load base config
    base_config_path = os.path.join(hdmi_dir, "cfg/task/base/hdmi-base.yaml")
    base_cfg = OmegaConf.load(base_config_path)

    # Load task-specific config (e.g., move_suitcase)
    task_config_path = os.path.join(hdmi_dir, f"cfg/task/G1/hdmi/{config.task}.yaml")
    print(f"task_config_path: {task_config_path}")
    task_cfg = OmegaConf.load(task_config_path)

    # Merge configurations (task overrides base)
    cfg = OmegaConf.merge(base_cfg, task_cfg)

    # Override with SPIDER config parameters
    cfg.num_envs = int(config.num_samples)
    # Disable mjlab viewer if "mjlab" is not in viewer string (e.g., "mujoco-rerun")
    cfg.viewer.headless = "mjlab" not in config.viewer.lower()

    # Set env_spacing to 0 so all environments share the same spatial location
    # This avoids needing to handle position offsets during state synchronization
    cfg.viewer.env_spacing = 0.0

    # Disable struct mode to allow modifications
    OmegaConf.set_struct(cfg, False)

    # Remove observation groups that are not needed for sampling
    # Keep only minimal observations for reward computation
    for obs_group_key in list(cfg.observation.keys()):
        if obs_group_key not in [
            "command",
            "policy",
            "priv",
        ] and not obs_group_key.endswith("_"):
            cfg.observation.pop(obs_group_key)

    # Remove randomizations in command manager
    if "pose_range" in cfg.command:
        for key in cfg.command.pose_range:
            cfg.command.pose_range[key] = [0.0, 0.0]
    if "velocity_range" in cfg.command:
        for key in cfg.command.velocity_range:
            cfg.command.velocity_range[key] = [0.0, 0.0]
    if "object_pose_range" in cfg.command:
        for key in cfg.command.object_pose_range:
            cfg.command.object_pose_range[key] = [0.0, 0.0]
    if "init_joint_pos_noise" in cfg.command:
        cfg.command.init_joint_pos_noise = 0.0
    if "init_joint_vel_noise" in cfg.command:
        cfg.command.init_joint_vel_noise = 0.0

    # Disable motion sampling for deterministic resets
    cfg.command.sample_motion = False
    # Set reset_range to None so it uses start_t.fill_(0) in eval mode
    cfg.command.reset_range = None

    # Disable action delay and alpha randomization
    if "action" in cfg:
        cfg.action.min_delay = 0
        cfg.action.max_delay = 0
        cfg.action.alpha = [1.0, 1.0]  # Set to fixed value

    # Remove randomizations - set all ranges to their mean
    for rand_key, rand_params in cfg.randomization.items():
        if rand_params is not None and isinstance(rand_params, dict):
            for param_key, param_value in rand_params.items():
                if isinstance(param_value, (list, tuple)) and len(param_value) == 2:
                    mean_val = (param_value[0] + param_value[1]) / 2.0
                    cfg.randomization[rand_key][param_key] = [mean_val, mean_val]
                elif isinstance(param_value, dict):
                    # Handle nested dicts
                    for sub_key, sub_value in param_value.items():
                        if isinstance(sub_value, (list, tuple)) and len(sub_value) == 2:
                            mean_val = (sub_value[0] + sub_value[1]) / 2.0
                            cfg.randomization[rand_key][param_key][sub_key] = [
                                mean_val,
                                mean_val,
                            ]

    # Filter out tracking reward groups and debug rewards
    filtered_rewards = {}
    for group_name, group_params in cfg.reward.items():
        if "tracking" in group_name.lower():
            # if group_name.lower() in ["tracking"]:
            # filter out tracking reward, only keep keypoint tracking
            filtered_rewards[group_name] = group_params
            # filter out reward term which has contact in the name
            for key, value in group_params.items():
                if "vel" in key.lower():
                    del filtered_rewards[group_name][key]

    cfg.reward = filtered_rewards
    # redefine tracking reward
    # cfg.reward = {
    #     "tracking": {
    #         "tracking_body_pos(keypoint_pos_tracking_local_product)": {
    #             "body_names": [
    #                 ".*_shoulder_pitch_link",
    #                 ".*_elbow_link",
    #                 ".*_wrist_yaw_link",
    #                 ".*_hip_pitch_link",
    #                 ".*_knee_link",
    #                 ".*_ankle_roll_link",
    #             ],
    #             "weight": 0.3,
    #             "sigma": 1.0,  # make it linear
    #         },
    #         "tracking_body_ori(keypoint_ori_tracking_local_product)": {
    #             "body_names": [
    #                 ".*_shoulder_pitch_link",
    #                 ".*_elbow_link",
    #                 ".*_wrist_yaw_link",
    #                 ".*_hip_pitch_link",
    #                 ".*_knee_link",
    #                 ".*_ankle_roll_link",
    #             ],
    #             "weight": 0.3,
    #             "sigma": 1.0,  # make it linear
    #         },
    #         "tracking_root_pos(keypoint_pos_tracking_product)": {
    #             "body_names": ["pelvis"],
    #             "weight": 1.0,
    #             "sigma": 1.0,  # make it linear
    #         },
    #         "tracking_root_ori(keypoint_ori_tracking_product)": {
    #             "body_names": ["pelvis"],
    #             "weight": 1.0,
    #             "sigma": 1.0,  # make it linear
    #         },
    #     },
    # }

    # Create environment
    env = SimpleEnv(cfg)

    # Set environment to eval mode for deterministic resets
    env.eval()

    env.reset()

    # Create a backup data structure for save/load state
    # This is similar to data_wp_prev in mjwp.py
    with wp.ScopedDevice(env.sim.device):
        env.data_wp_prev = mjwarp.put_data(
            env.sim.mj_model,
            env.sim.mj_data,
            nworld=env.num_envs,
            nconmax=env.sim.cfg.nconmax,
            njmax=env.sim.cfg.njmax,
        )

    return env


def save_state(env):
    """Save the state of the environment."""
    # reset method 1: use mjlab api
    # robot = env.scene.entities["robot"]

    # state = {
    #     # Robot state
    #     "robot_root_pos": robot.data.root_link_pos_w.clone(),
    #     "robot_root_quat": robot.data.root_link_quat_w.clone(),
    #     "robot_root_lin_vel": robot.data.root_com_lin_vel_w.clone(),
    #     "robot_root_ang_vel": robot.data.root_com_ang_vel_w.clone(),
    #     "robot_joint_pos": robot.data.joint_pos.clone(),
    #     "robot_joint_vel": robot.data.joint_vel.clone(),
    # }

    # # Save object state if available
    # for entity_name, entity in env.scene.entities.items():
    #     if entity_name != "robot":
    #         state[f"{entity_name}_root_pos"] = entity.data.root_link_pos_w.clone()
    #         state[f"{entity_name}_root_quat"] = entity.data.root_link_quat_w.clone()
    #         state[f"{entity_name}_root_lin_vel"] = (
    #             entity.data.root_com_lin_vel_w.clone()
    #         )
    #         state[f"{entity_name}_root_ang_vel"] = (
    #             entity.data.root_com_ang_vel_w.clone()
    #         )
    #         if hasattr(entity.data, "joint_pos") and entity.data.joint_pos is not None:
    #             state[f"{entity_name}_joint_pos"] = entity.data.joint_pos.clone()
    #             state[f"{entity_name}_joint_vel"] = entity.data.joint_vel.clone()

    # reset method 2: directly sync mujoco warp data
    # Copy current simulation state to backup using _copy_state
    _copy_state(env.sim.wp_data, env.data_wp_prev)

    # Also save environment-specific state like command manager
    state = {
        "data_wp_prev": env.data_wp_prev,  # Reference to backup data
    }

    # Save command manager state
    if hasattr(env.command_manager, "t"):
        state["command_t"] = env.command_manager.t.clone()
    if hasattr(env, "episode_length_buf"):
        state["episode_length_buf"] = env.episode_length_buf.clone()
    if hasattr(env, "timestamp"):
        state["timestamp"] = env.timestamp

    return state


def load_state(env, state):
    """Load the state of the environment by restoring from backup.

    Args:
        env: The environment instance
        state: Dict containing 'data_wp_prev' backup reference and other state
    """
    # reset method 1: use mjlab api (commented out, kept for reference)
    # robot = env.scene.entities["robot"]
    # env_ids = None
    # # Load robot state
    # robot_root_pose = torch.cat(
    #     [state["robot_root_pos"], state["robot_root_quat"]], dim=-1
    # )
    # robot.write_root_link_pose_to_sim(robot_root_pose, env_ids=env_ids)
    # robot_root_velocity = torch.cat(
    #     [state["robot_root_lin_vel"], state["robot_root_ang_vel"]], dim=-1
    # )
    # robot.write_root_com_velocity_to_sim(robot_root_velocity, env_ids=env_ids)
    # robot.write_joint_state_to_sim(
    #     state["robot_joint_pos"], state["robot_joint_vel"], env_ids=env_ids
    # )
    # # Load object states
    # for entity_name, entity in env.scene.entities.items():
    #     if entity_name != "robot" and f"{entity_name}_root_pos" in state:
    #         obj_root_pose = torch.cat(
    #             [state[f"{entity_name}_root_pos"], state[f"{entity_name}_root_quat"]],
    #             dim=-1,
    #         )
    #         entity.write_root_link_pose_to_sim(obj_root_pose, env_ids=env_ids)
    #         obj_root_velocity = torch.cat(
    #             [
    #                 state[f"{entity_name}_root_lin_vel"],
    #                 state[f"{entity_name}_root_ang_vel"],
    #             ],
    #             dim=-1,
    #         )
    #         entity.write_root_com_velocity_to_sim(obj_root_velocity, env_ids=env_ids)
    #         # Only write joint state if entity has joints
    #         if (
    #             f"{entity_name}_joint_pos" in state
    #             and hasattr(entity.data, "joint_pos")
    #             and entity.data.joint_pos is not None
    #             and entity.data.joint_pos.numel() > 0
    #         ):
    #             entity.write_joint_state_to_sim(
    #                 state[f"{entity_name}_joint_pos"],
    #                 state[f"{entity_name}_joint_vel"],
    #                 env_ids=env_ids,
    #             )

    # reset method 2: directly sync mujoco warp data (active implementation)
    _copy_state(state["data_wp_prev"], env.sim.wp_data)

    # Restore environment-specific state like command manager
    env.command_manager.t[:] = state["command_t"]
    env.episode_length_buf[:] = state["episode_length_buf"]
    env.timestamp = state["timestamp"]

    return env


def step_env(config: Config, env, ctrl: torch.Tensor):
    """Step all worlds with provided controls of shape (N, nu)."""
    if ctrl.dim() == 1:
        ctrl = ctrl.unsqueeze(0).repeat(int(config.num_samples), 1)

    tensordict = TensorDict(
        {"action": ctrl},
        batch_size=[env.num_envs],
        device=ctrl.device,
    )

    # Step physics with decimation
    for substep in range(env.decimation):
        # Apply action (this handles action delays and scaling)
        env.apply_action(tensordict, substep)

        # Write data to simulation
        env.scene.write_data_to_sim()

        # Step physics
        env.sim.step()

        # Update scene (read from simulation)
        env.scene.update(env.physics_dt)

    # Update viewer if enabled
    _update_viewer(env)

    # Update command manager
    env.command_manager.update()

    # Update episode length
    env.episode_length_buf.add_(1)
    env.timestamp += 1

    return env


def _update_viewer(env):
    """Update mjlab viewer.

    Args:
        env (_type_): _description_
    """
    if not env._viewer_enabled or env.viewer is None:
        return

    idx = min(env.viewer_env_index, max(env.num_envs - 1, 0))
    sim_data = env.sim.data
    base_qpos = sim_data.qpos[idx].detach().cpu().numpy()
    base_qvel = sim_data.qvel[idx].detach().cpu().numpy()

    # Render the focused environment.
    env.sim.mj_data.qpos[:] = base_qpos
    env.sim.mj_data.qvel[:] = base_qvel
    mujoco.mj_forward(env.sim.mj_model, env.sim.mj_data)

    # Clear scene geoms so we can append the rest.
    env.viewer.user_scn.ngeom = 0

    # Overlay remaining environments as ghost geoms.
    num_render = min(env.num_envs, 8)
    if (
        env.num_envs > 1
        and env._viewer_vd is not None
        and env._viewer_vopt is not None
        and env._viewer_pert is not None
        and env._viewer_catmask is not None
    ):
        sim_data = env.sim.data
        for i in range(num_render):
            if i == idx:
                continue
            try:
                qpos_i = sim_data.qpos[i].detach().cpu().numpy()
                qvel_i = sim_data.qvel[i].detach().cpu().numpy()
            except Exception:
                continue
            env._viewer_vd.qpos[:] = qpos_i
            env._viewer_vd.qvel[:] = qvel_i
            mujoco.mj_forward(env.sim.mj_model, env._viewer_vd)
            mujoco.mjv_addGeoms(
                env.sim.mj_model,
                env._viewer_vd,
                env._viewer_vopt,
                env._viewer_pert,
                env._viewer_catmask,
                env.viewer.user_scn,
            )

    if getattr(env.viewer, "is_running", lambda: True)():
        env.viewer.sync(state_only=True)


def get_reward(
    config: Config,
    env,
    ref: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, dict]:
    """Get reward for the current state.

    Returns (N,) shaped tensor and info dict with (N,) shaped values.
    """
    del config, ref  # Not used

    # Compute rewards from all reward groups
    total_reward = torch.zeros(env.num_envs, device=env.device)
    info = {}

    for group_name, reward_group in env.reward_groups.items():
        for key, func in reward_group.funcs.items():
            func.update()
        #     # rew = torch.log(func.compute()) * func.sigma
        #     # rew = func.weight * rew
        #     rew = func().squeeze(-1)
        #     total_reward += rew
        #     info[key] = rew
        group_reward = reward_group.compute()
        # Ensure reward is 1D with shape (num_envs,)
        if group_reward.dim() > 1:
            group_reward = group_reward.squeeze(-1)
        total_reward += group_reward
        # Store per-env rewards with shape (num_envs,)
        info[group_name] = group_reward

    return total_reward, info


def get_terminate(
    config: Config, env, ref_slice: tuple[torch.Tensor, ...]
) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device)


def get_terminal_reward(
    config: Config,
    env,
    ref_slice: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, dict]:
    """Terminal reward focusing on object tracking."""
    rew, info = get_reward(config, env, ref_slice)
    return config.terminal_rew_scale * rew, info


def get_trace(config: Config, env) -> torch.Tensor:
    """Get trace information for visualization.

    Returns trace points including hand positions, foot positions, and object position.
    Returns (N, num_trace_points, 3) shaped tensor.
    """
    robot = env.scene.entities["robot"]

    # Get hand positions (wrist yaw links)
    hand_ids = robot.find_bodies(".*_wrist_yaw_link")[0]
    hand_pos = robot.data.body_com_pos_w[:, hand_ids, :]  # (N, 2, 3) for left and right

    # Get foot positions (ankle roll links)
    foot_ids = robot.find_bodies(".*_ankle_roll_link")[0]
    foot_pos = robot.data.body_com_pos_w[:, foot_ids, :]  # (N, 2, 3) for left and right

    # Get object position if available
    trace_points = [hand_pos, foot_pos]

    for entity_name, entity in env.scene.entities.items():
        if entity_name != "robot":
            # Get object root position
            obj_pos = entity.data.root_link_pos_w.unsqueeze(1)  # (N, 1, 3)
            trace_points.append(obj_pos)
            break  # Only get first object

    # Concatenate all trace points: hands (2) + feet (2) + object (1) = 5 points
    trace = torch.cat(trace_points, dim=1)

    return trace


def save_env_params(config: Config, env):
    """Save environment parameters (for domain randomization).

    For HDMI, we don't use domain randomization, so return empty dict.
    """
    return {}


def load_env_params(config: Config, env, env_param: dict):
    """Load environment parameters (for domain randomization).

    For HDMI, we don't use domain randomization, so this is a no-op.
    """
    return env


def copy_sample_state(
    config: Config, env, src_indices: torch.Tensor, dst_indices: torch.Tensor
):
    """Copy simulation state from source samples to destination samples.

    Args:
        config: Config
        env: HDMI environment
        src_indices: Tensor of shape (n,) containing source sample indices
        dst_indices: Tensor of shape (n,) containing destination sample indices
    """
    # Convert to numpy for indexing
    src_idx = src_indices.cpu().numpy()
    dst_idx = dst_indices.cpu().numpy()

    # Get all state data as torch tensors from sim.wp_data
    qpos = wp.to_torch(env.sim.wp_data.qpos)
    qvel = wp.to_torch(env.sim.wp_data.qvel)
    qacc = wp.to_torch(env.sim.wp_data.qacc)
    time_arr = wp.to_torch(env.sim.wp_data.time)
    ctrl = wp.to_torch(env.sim.wp_data.ctrl)
    act = wp.to_torch(env.sim.wp_data.act)
    act_dot = wp.to_torch(env.sim.wp_data.act_dot)
    qacc_warmstart = wp.to_torch(env.sim.wp_data.qacc_warmstart)
    qfrc_applied = wp.to_torch(env.sim.wp_data.qfrc_applied)
    xfrc_applied = wp.to_torch(env.sim.wp_data.xfrc_applied)

    # Copy from src to dst (core state only for efficiency)
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

    # Copy back to warp arrays
    wp.copy(env.sim.wp_data.qpos, wp.from_torch(qpos))
    wp.copy(env.sim.wp_data.qvel, wp.from_torch(qvel))
    wp.copy(env.sim.wp_data.qacc, wp.from_torch(qacc))
    wp.copy(env.sim.wp_data.time, wp.from_torch(time_arr))
    wp.copy(env.sim.wp_data.ctrl, wp.from_torch(ctrl))
    wp.copy(env.sim.wp_data.act, wp.from_torch(act))
    wp.copy(env.sim.wp_data.act_dot, wp.from_torch(act_dot))
    wp.copy(env.sim.wp_data.qacc_warmstart, wp.from_torch(qacc_warmstart))
    wp.copy(env.sim.wp_data.qfrc_applied, wp.from_torch(qfrc_applied))
    wp.copy(env.sim.wp_data.xfrc_applied, wp.from_torch(xfrc_applied))


def sync_env(config: Config, env):
    """Broadcast the state from first env to all envs.

    This function synchronizes states from the first environment to all environments.
    Since env_spacing is set to 0, all environments share the same spatial location,
    so no position offset handling is needed.
    """
    # Use shared broadcast function from mjwp
    _broadcast_state(env.sim.wp_data, env.num_envs)
    return env


def get_reference(
    config: Config, env
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get full reference motion data (states and controls) from command manager.

    Returns tuple of:
        - qpos_ref: (max_sim_steps, nq) shaped tensor with full state including robot and objects
        - qvel_ref: (max_sim_steps, nv) shaped tensor with full velocities
        - ctrl_ref: (max_sim_steps, nu) shaped tensor with control actions
    """
    action_manager = env.action_manager
    command_manager = env.command_manager

    # Get full motion data slice
    motion_data = command_manager.dataset.get_slice(
        command_manager.motion_ids, 0, steps=config.max_sim_steps
    )

    # Get action joint indices in motion data
    action_indices_motion = [
        command_manager.dataset.joint_names.index(joint_name)
        for joint_name in action_manager.joint_names
    ]

    # Get reference joint positions for control
    ref_joint_pos = motion_data.joint_pos[0, :, action_indices_motion]

    # Convert to actions using action manager's normalization
    # action = (joint_pos - default_joint_pos) / action_scaling
    default_joint_pos = action_manager.default_joint_pos[0, action_manager.joint_ids]
    action_scaling = action_manager.action_scaling
    ctrl_ref = (ref_joint_pos - default_joint_pos) / action_scaling

    # Extract reference qpos and qvel for all bodies
    # Robot joint positions and velocities
    robot_joint_pos = motion_data.joint_pos[0]  # (steps, num_joints)
    robot_joint_vel = motion_data.joint_vel[0]  # (steps, num_joints)

    # Root body (pelvis) states
    root_body_idx = command_manager.root_body_idx_motion
    root_pos = motion_data.body_pos_w[0, :, root_body_idx, :]  # (steps, 3)
    root_quat = motion_data.body_quat_w[0, :, root_body_idx, :]  # (steps, 4)
    root_lin_vel = motion_data.body_lin_vel_w[0, :, root_body_idx, :]  # (steps, 3)
    root_ang_vel = motion_data.body_ang_vel_w[0, :, root_body_idx, :]  # (steps, 3)

    # Build qpos: [root_pos (3), root_quat (4), joint_pos (n_joints)]
    qpos_ref = torch.cat([root_pos, root_quat, robot_joint_pos], dim=-1)

    # Build qvel: [root_lin_vel (3), root_ang_vel (3), joint_vel (n_joints)]
    qvel_ref = torch.cat([root_lin_vel, root_ang_vel, robot_joint_vel], dim=-1)

    # Add object states if available
    if hasattr(command_manager, "object_body_id_motion"):
        object_body_idx = command_manager.object_body_id_motion
        object_pos = motion_data.body_pos_w[0, :, object_body_idx, :]  # (steps, 3)
        object_quat = motion_data.body_quat_w[0, :, object_body_idx, :]  # (steps, 4)
        object_lin_vel = motion_data.body_lin_vel_w[
            0, :, object_body_idx, :
        ]  # (steps, 3)
        object_ang_vel = motion_data.body_ang_vel_w[
            0, :, object_body_idx, :
        ]  # (steps, 3)

        # Append object states to qpos and qvel
        qpos_ref = torch.cat([qpos_ref, object_pos, object_quat], dim=-1)
        qvel_ref = torch.cat([qvel_ref, object_lin_vel, object_ang_vel], dim=-1)

        # Add object joint if available
        if (
            hasattr(command_manager, "object_joint_idx_motion")
            and command_manager.object_joint_idx_motion is not None
        ):
            object_joint_pos = motion_data.joint_pos[
                0, :, command_manager.object_joint_idx_motion
            ].unsqueeze(-1)  # (steps, 1)
            object_joint_vel = motion_data.joint_vel[
                0, :, command_manager.object_joint_idx_motion
            ].unsqueeze(-1)  # (steps, 1)
            qpos_ref = torch.cat([qpos_ref, object_joint_pos], dim=-1)
            qvel_ref = torch.cat([qvel_ref, object_joint_vel], dim=-1)

    # repeat last frame by config.horizon_steps + config.ctrl_steps times to avoid overflow
    last_frame = qpos_ref[-1:].repeat(config.horizon_steps + config.ctrl_steps, 1)
    qpos_ref = torch.cat([qpos_ref, last_frame], dim=0)
    last_frame = qvel_ref[-1:].repeat(config.horizon_steps + config.ctrl_steps, 1) * 0.0
    qvel_ref = torch.cat([qvel_ref, last_frame], dim=0)
    last_frame = ctrl_ref[-1:].repeat(config.horizon_steps + config.ctrl_steps, 1)
    ctrl_ref = torch.cat([ctrl_ref, last_frame], dim=0)

    # verify shape
    nq_ref = qpos_ref.shape[-1]
    nq_env = env.sim.mj_model.nq
    assert nq_ref == nq_env, f"nq_ref: {nq_ref}, nq_env: {nq_env}"
    nv_ref = qvel_ref.shape[-1]
    nv_env = env.sim.mj_model.nv
    assert nv_ref == nv_env, f"nv_ref: {nv_ref}, nv_env: {nv_env}"
    nu_ref = ctrl_ref.shape[-1]
    nu_env = config.nu  # NOTE: the action dimension in action manager is different from the one in simulator
    assert nu_ref == nu_env, f"nu_ref: {nu_ref}, nu_env: {nu_env}"

    return qpos_ref, qvel_ref, ctrl_ref
