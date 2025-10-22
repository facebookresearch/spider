"""Simulator for sampling with DexMachina based on Genesis.

This module provides a minimal DexMachina based on Genesis backend that matches the sampling API used by
the generic optimizer pipeline. It intentionally keeps the implementation simple
and robust.
"""

from __future__ import annotations

# Import genesis and dexmachina related modules
import genesis as gs
import loguru
import torch
from dexmachina.envs.base_env import BaseEnv
from dexmachina.envs.constructors import (
    get_all_env_cfg,
    get_common_argparser,
    parse_clip_string,
)
from dexmachina.envs.reward_utils import (
    chamfer_distance,
    position_distance,
    rotation_distance,
    transform_contact,
)

from spider.config import Config

# Initialize Genesis once per process
try:
    gs.init(backend=gs.gpu, logging_level="warning")
except Exception as e:
    loguru.logger.error(f"Failed to initialize Genesis: {e}")


# --
# Key functions
# --


def setup_env(config: Config, ref_data: tuple[torch.Tensor, ...]) -> BaseEnv:
    """Setup and reset the environment backed by Genesis.
    Returns a BaseEnv instance.
    """
    parser = get_common_argparser()
    args = parser.parse_args([])

    # Override the default arguments
    args.hand = config.robot_type  # "inspire_hand", "xhand", "dex3_hand"
    args.render_camera = "front"
    args.num_envs = int(config.num_samples)
    args.action_mode = "hybrid"
    args.is_eval = True
    args.early_reset_threshold = 0.0
    args.early_reset_aux_thres = {"con": 0, "imi": 0, "bc": 0}
    args.actuate_object = True
    if "gs" in config.viewer:
        args.vis = True
        args.n_render = 4
    args.retarget_name = "para"
    args.render_dir = f"{config.output_dir}/trajectory_gs.mp4"
    args.record_video = config.save_video
    args.clip = config.task  # "box-50-230"
    args.kp_init = 0.0
    args.kv_init = 0.0

    # --- Environment Configuration ---
    # Simplified environment configuration, focusing on core components
    obj_name, start, end, subject_name, use_clip = parse_clip_string(args.clip)
    args.arctic_object = obj_name
    args.frame_start = start
    args.frame_end = end
    env_kwargs = get_all_env_cfg(args, device=str(config.device))

    # Reward setup
    env_kwargs["reward_cfg"]["bc_rew_weight"] = 0.0
    # Set imitation and contact reward weights to enable loading demo data
    # These will be overridden by config parameters in get_reward()
    imi_rew_weight = getattr(config, "imi_rew_weight", 0.1)
    contact_rew_weight = getattr(config, "contact_rew_weight", 1.0)
    env_kwargs["reward_cfg"]["imi_rew_weight"] = imi_rew_weight
    env_kwargs["reward_cfg"]["contact_rew_weight"] = contact_rew_weight
    # Enable contact tracking if contact rewards are used
    env_kwargs["env_cfg"]["use_contact_reward"] = contact_rew_weight > 0.0
    if "gs" in config.viewer:
        env_kwargs["env_cfg"]["scene_kwargs"]["n_rendered_envs"] = 4
        env_kwargs["env_cfg"]["env_spacing"] = (0.0, 0.0)

    env = BaseEnv(**env_kwargs)
    env._recording = False  # disable recording since we will do shooting
    env.reset()

    return env


def get_obj_pos_dist(env: BaseEnv) -> torch.Tensor:
    """Get the position distance between the object and the demo object."""
    obj_pos = env.objects[env.object_names[0]].entity.get_pos()
    demo_pos = env.reward_module.match_demo_state("obj_pos", env.episode_length_buf)
    obj_pos_dist = position_distance(demo_pos, obj_pos)
    return obj_pos_dist


def get_obj_quat_dist(env: BaseEnv) -> torch.Tensor:
    """Get the rotation distance between the object and the demo object."""
    obj_quat = env.objects[env.object_names[0]].entity.get_quat()
    demo_quat = env.reward_module.match_demo_state("obj_quat", env.episode_length_buf)
    obj_quat_dist = rotation_distance(demo_quat, obj_quat)
    return obj_quat_dist


def get_obj_arti_dist(env: BaseEnv) -> torch.Tensor:
    """Get the articulation distance between the object and the demo object."""
    obj_arti = env.objects[env.object_names[0]].entity.get_dofs_position(
        env.objects[env.object_names[0]].dof_idxs
    )
    demo_arti = env.reward_module.match_demo_state("obj_arti", env.episode_length_buf)
    obj_arti_dist = position_distance(demo_arti, obj_arti)
    return obj_arti_dist


def reshape_contact_with_label(
    contact_link_pos: torch.Tensor, contact_link_valid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reshape the environment contact to better match with demos.
    In:
        contact_link_pos: (N, 2, num_hand_links, 3), 2 is each object part
        contact_link_valid: (N, 2, num_hand_links)
    Out:
        reshaped_contact_link_pos: (N, 2*num_hand_links, 4): 4 is (x, y, z, part_id), part_id is 1 or 2 or 0 if no valid
        reshaped contact_link_valid: (N, 2*num_hand_links)
    """
    # Add a part_id dim so (N, 2, num_links, 4) and last dim is part_id label
    labeled = torch.cat(
        [contact_link_pos, torch.zeros_like(contact_link_pos)[:, :, :, 0:1]], dim=-1
    )  # (N, 2, num_links, 4)
    # Fill in the labels 1 or 2 if valid contact
    valid_reshaped = contact_link_valid[..., None]  # (N, 2, num_links, 1)
    valid = torch.cat(
        [valid_reshaped, torch.zeros_like(valid_reshaped)], dim=-1
    )  # -> (N, 2, num_links, 2)
    for part_idx in [0, 1]:  # bottom, top (since obj urdf is flipped for parsing error)
        label_id = part_idx + 1
        # Need flipping, because in demos, id=2 is bottom
        label_id = 1 if label_id == 2 else 2
        labeled[:, part_idx, :, -1] = torch.where(
            contact_link_valid[:, part_idx, :], label_id, 0
        )
        valid[:, part_idx, :, 1] = torch.where(
            contact_link_valid[:, part_idx, :], label_id, 0
        )

    # Reshape to (N, num_links*2, 4)
    contact_link_reshaped = labeled.view(labeled.shape[0], -1, labeled.shape[-1])
    contact_valid_reshaped = valid.view(valid.shape[0], -1, valid.shape[-1])

    return contact_link_reshaped, contact_valid_reshaped


def contact_dist_to_rew(
    dist: torch.Tensor, beta: float = 30.0, function: str = "exp"
) -> torch.Tensor:
    """Convert contact distance to reward using exponential or sigmoid function."""
    if function == "exp":
        return torch.exp(-beta * dist)
    elif function == "sigmoid":
        a = 100.0
        b = 5.0
        sigmoid_offset = (
            torch.tensor(-1 / (1.0 + torch.exp(torch.tensor(-b))), device=dist.device)
            + 1.0
        )
        axb = a * dist + b
        rew = 1.0 / (1.0 + torch.exp(-axb)) + sigmoid_offset
        return rew
    else:
        raise ValueError(f"Unknown contact reward function: {function}")


def get_hand_contact_reward(
    env: BaseEnv,
    contact_link_pos: torch.Tensor,  # shape (N, num_obj_links * num_hand_links, 4)
    _contact_link_valid: torch.Tensor,  # shape (N, num_obj_links * num_hand_links, 1)
    wrist_pose: torch.Tensor,  # shape (N, 7)
    obj_pose: torch.Tensor,  # shape (N, 7)
    demo_obj_pose: torch.Tensor,  # shape (N, 7)
    side: str = "left",
    beta: float = 30.0,
    wrist_frame_contact: bool = True,
    mask_zero_contact: bool = True,
    multiply_frame_contact: bool = True,
    contact_rew_function: str = "exp",
) -> tuple[torch.Tensor, dict]:
    """Compute contact reward for one hand."""
    demo_wrist_pose = env.reward_module.match_demo_state(
        f"wrist_pose_{side}", env.episode_length_buf
    )
    demo_contacts = env.reward_module.match_demo_state(
        f"contact_links_{side}", env.episode_length_buf
    )
    # N, num_links * 2, 4 (last dim is contact pair ID)
    # NOTE in ARCTIC, part_id=2 is 'bottom' link, part_id=1 is 'top'
    demo_positions = demo_contacts[:, :, :3]
    demo_valid_contact = (
        demo_contacts[:, :, -1] > 0.0
    )  # (part id is <= 0 if no contact)
    chamfer_dists = {}
    contact_rewards = {}

    positions = contact_link_pos[:, :, :3]
    valid_mask = contact_link_pos[:, :, -1] > 0.0
    for part_id in [1, 2]:
        # Consider contact invalid if part_id is different
        demo_part_valid = (demo_contacts[:, :, -1] == part_id) & demo_valid_contact
        part_valid = (contact_link_pos[:, :, -1] == part_id) & valid_mask

        for frame in ["obj", "wrist"]:
            if not wrist_frame_contact and frame == "wrist":
                continue
            demo_pose = demo_obj_pose if frame == "obj" else demo_wrist_pose
            demo_in_frame = transform_contact(demo_positions, demo_pose)
            pose = obj_pose if frame == "obj" else wrist_pose
            in_frame = transform_contact(positions, pose)
            dist = chamfer_distance(
                in_frame, demo_in_frame, part_valid, demo_part_valid
            )
            chamfer_dists[f"CD_{side}_p{part_id}_frame_{frame}"] = dist
            rew = contact_dist_to_rew(dist, beta, contact_rew_function)
            both_zero = (part_valid.sum(dim=-1) == 0) & (
                demo_part_valid.sum(dim=-1) == 0
            )
            if mask_zero_contact:
                rew = torch.where(both_zero, torch.zeros_like(rew), rew)
            else:
                rew = torch.where(both_zero, torch.ones_like(rew), rew)
            contact_rewards[f"conrew_{side}_p{part_id}_frame_{frame}"] = rew

    # Per-part contact should be added, because sometimes full contact coverage is not feasible
    if multiply_frame_contact:
        # Multiply the rew in obj & wrist frame such that one does not dominate the other
        contact_rews = []
        for part_id in [1, 2]:
            obj_rew = contact_rewards[f"conrew_{side}_p{part_id}_frame_obj"]
            if not wrist_frame_contact:
                wrist_rew = 1.0
            else:
                wrist_rew = contact_rewards[f"conrew_{side}_p{part_id}_frame_wrist"]
            multi_rew = obj_rew * wrist_rew
            contact_rewards[f"mul_conrew_{side}_p{part_id}"] = multi_rew
            contact_rews.append(multi_rew)
        contact_rew = torch.stack(contact_rews, dim=-1).mean(dim=-1)
    else:  # sum & average all contact rewards
        contact_rew = sum(contact_rewards.values()) / len(contact_rewards)

    contact_dict = {
        **chamfer_dists,
        **contact_rewards,
        f"contact_rew_{side}": contact_rew,
    }
    return contact_rew, contact_dict


def get_keypoint_dist(
    env: BaseEnv, keypoint_pos: torch.Tensor, left_hand: bool = True
) -> torch.Tensor:
    """Get keypoint distance between policy and demo."""
    demo_key = "kpts_left" if left_hand else "kpts_right"
    demo_kpts = env.reward_module.match_demo_state(demo_key, env.episode_length_buf)
    return position_distance(keypoint_pos, demo_kpts)


def get_wrist_reward(
    env: BaseEnv,
    wrist_pose: torch.Tensor,
    side: str = "left",
    wrist_rot_beta: float = 3.0,
    wrist_pos_beta: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute wrist reward and distances."""
    demo_wrist = env.reward_module.match_demo_state(
        f"wrist_pose_{side}", env.episode_length_buf
    )
    wrist_rot_dist = rotation_distance(wrist_pose[:, 3:], demo_wrist[:, 3:])
    wrist_pos_dist = position_distance(wrist_pose[:, :3], demo_wrist[:, :3])
    wrist_rew = (
        torch.exp(-wrist_rot_beta * wrist_rot_dist)
        + torch.exp(-wrist_pos_beta * wrist_pos_dist)
    ) / 2.0
    return wrist_rew, wrist_pos_dist, wrist_rot_dist


def get_imitation_reward(
    env: BaseEnv,
    imi_rew_weight: float = 1.0,
    imi_fingertip_beta: float = 20.0,
    imi_wrist_weight: float = 0.0,
    imi_wrist_rot_beta: float = 3.0,
    imi_wrist_pos_beta: float = 10.0,
    exp_kpt_first: bool = True,
) -> torch.Tensor:
    """Get the imitation reward based on keypoint and wrist tracking."""
    # Get keypoint positions
    kpts_left = env.robots["left"].kpt_pos
    kpts_right = env.robots["right"].kpt_pos
    wrist_pose_left = env.robots["left"].wrist_pose
    wrist_pose_right = env.robots["right"].wrist_pose

    # Compute fingertip distances
    fingertip_dist_left = get_keypoint_dist(env, kpts_left, left_hand=True)
    fingertip_dist_right = get_keypoint_dist(env, kpts_right, left_hand=False)
    fingertip_dist = torch.mean(
        (fingertip_dist_left + fingertip_dist_right) / 2.0, dim=-1
    )  # (B, num_links) -> (B,)

    if exp_kpt_first:
        fingertip_rew_left = torch.exp(-imi_fingertip_beta * fingertip_dist_left)
        fingertip_rew_right = torch.exp(-imi_fingertip_beta * fingertip_dist_right)
        fingertip_rew = torch.mean(
            (fingertip_rew_left + fingertip_rew_right) / 2.0, dim=-1
        )  # (B, num_links) -> (B,)
    else:
        fingertip_rew = torch.exp(-imi_fingertip_beta * fingertip_dist)

    # Optionally add wrist reward
    if imi_wrist_weight > 0.0:
        wrist_rew_left, _, _ = get_wrist_reward(
            env,
            wrist_pose_left,
            side="left",
            wrist_rot_beta=imi_wrist_rot_beta,
            wrist_pos_beta=imi_wrist_pos_beta,
        )
        wrist_rew_right, _, _ = get_wrist_reward(
            env,
            wrist_pose_right,
            side="right",
            wrist_rot_beta=imi_wrist_rot_beta,
            wrist_pos_beta=imi_wrist_pos_beta,
        )
        wrist_rew = (wrist_rew_left + wrist_rew_right) / 2.0
        imi_rew = (
            imi_wrist_weight * wrist_rew + (1.0 - imi_wrist_weight) * fingertip_rew
        )
    else:
        imi_rew = fingertip_rew

    imi_rew -= 1.0
    imi_rew *= imi_rew_weight
    return imi_rew


def get_contact_reward(
    env: BaseEnv,
    contact_rew_weight: float = 1.0,
    contact_beta: float = 30.0,
    wrist_frame_contact: bool = True,
    mask_zero_contact: bool = True,
    multiply_frame_contact: bool = True,
    contact_rew_function: str = "exp",
) -> torch.Tensor:
    """Get the contact reward between the hands and the object."""
    if not hasattr(env, "contact_link_pos") or not hasattr(env, "contact_link_valid"):
        return torch.zeros(env.num_envs, device=env.device)

    # Get object pose
    obj_pos = env.objects[env.object_names[0]].entity.get_pos()
    obj_quat = env.objects[env.object_names[0]].entity.get_quat()
    obj_pose = torch.cat([obj_pos, obj_quat], dim=1)

    # Get demo object pose
    demo_pos = env.reward_module.match_demo_state("obj_pos", env.episode_length_buf)
    demo_quat = env.reward_module.match_demo_state("obj_quat", env.episode_length_buf)
    demo_obj_pose = torch.cat([demo_pos, demo_quat], dim=1)

    # Split contact data into left and right hands
    num_left_links = len(env.robots["left"].kpt_link_idxs)
    contact_link_pos_left = env.contact_link_pos[:, :, :num_left_links]
    contact_link_valid_left = env.contact_link_valid[:, :, :num_left_links]
    contact_link_pos_right = env.contact_link_pos[:, :, num_left_links:]
    contact_link_valid_right = env.contact_link_valid[:, :, num_left_links:]

    # Reshape contacts with labels
    left_reshaped, left_valid = reshape_contact_with_label(
        contact_link_pos_left, contact_link_valid_left
    )
    right_reshaped, right_valid = reshape_contact_with_label(
        contact_link_pos_right, contact_link_valid_right
    )

    # Get wrist poses
    wrist_pose_left = env.robots["left"].wrist_pose
    wrist_pose_right = env.robots["right"].wrist_pose

    # Compute contact reward for each hand
    contact_rew_left, _ = get_hand_contact_reward(
        env,
        left_reshaped,
        left_valid,
        wrist_pose_left,
        obj_pose,
        demo_obj_pose,
        side="left",
        beta=contact_beta,
        wrist_frame_contact=wrist_frame_contact,
        mask_zero_contact=mask_zero_contact,
        multiply_frame_contact=multiply_frame_contact,
        contact_rew_function=contact_rew_function,
    )
    contact_rew_right, _ = get_hand_contact_reward(
        env,
        right_reshaped,
        right_valid,
        wrist_pose_right,
        obj_pose,
        demo_obj_pose,
        side="right",
        beta=contact_beta,
        wrist_frame_contact=wrist_frame_contact,
        mask_zero_contact=mask_zero_contact,
        multiply_frame_contact=multiply_frame_contact,
        contact_rew_function=contact_rew_function,
    )

    # Average contact rewards from both hands
    contact_rew = (contact_rew_left + contact_rew_right) / 2.0
    contact_rew -= 1.0
    contact_rew *= contact_rew_weight

    return contact_rew


def get_reward(
    config: Config,
    env: BaseEnv,
    ref: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Non-terminal step reward for Genesis batched worlds.
    ref is a tuple (placeholder for compatibility)
    Returns (N,)
    """
    obj_pos_dist = get_obj_pos_dist(env)
    obj_quat_dist = get_obj_quat_dist(env)
    obj_dist = (obj_pos_dist + obj_quat_dist) / 2.0
    obj_dist = torch.clamp(obj_dist, min=0.0, max=1.0)

    # Get object articulation distance
    obj_arti_dist = get_obj_arti_dist(env)
    obj_arti_dist = torch.clamp(obj_arti_dist, min=0.0, max=1.0)

    # Base tracking reward
    obj_dist_rew = -obj_dist * 1.0
    obj_arti_rew = -obj_arti_dist * 1.0
    reward = obj_dist_rew + obj_arti_rew

    # Add imitation reward if enabled
    imi_rew_weight = getattr(config, "imi_rew_weight", 3.0)
    if imi_rew_weight > 0.0:
        imi_fingertip_beta = getattr(config, "imi_fingertip_beta", 1.0)
        imi_wrist_weight = getattr(config, "imi_wrist_weight", 0.0)
        imi_wrist_rot_beta = getattr(config, "imi_wrist_rot_beta", 1.0)
        imi_wrist_pos_beta = getattr(config, "imi_wrist_pos_beta", 1.0)
        exp_kpt_first = getattr(config, "exp_kpt_first", True)

        imi_rew = get_imitation_reward(
            env,
            imi_rew_weight=imi_rew_weight,
            imi_fingertip_beta=imi_fingertip_beta,
            imi_wrist_weight=imi_wrist_weight,
            imi_wrist_rot_beta=imi_wrist_rot_beta,
            imi_wrist_pos_beta=imi_wrist_pos_beta,
            exp_kpt_first=exp_kpt_first,
        )
    else:
        imi_rew = torch.zeros(env.num_envs, device=env.device)
    reward += imi_rew

    # Add contact reward if enabled
    contact_rew_weight = getattr(config, "contact_rew_weight", 0.0)
    if contact_rew_weight > 0.0:
        contact_beta = getattr(config, "contact_beta", 1.0)
        wrist_frame_contact = getattr(config, "wrist_frame_contact", True)
        mask_zero_contact = getattr(config, "mask_zero_contact", True)
        multiply_frame_contact = getattr(config, "multiply_frame_contact", True)
        contact_rew_function = getattr(config, "contact_rew_function", "exp")

        contact_rew = get_contact_reward(
            env,
            contact_rew_weight=contact_rew_weight,
            contact_beta=contact_beta,
            wrist_frame_contact=wrist_frame_contact,
            mask_zero_contact=mask_zero_contact,
            multiply_frame_contact=multiply_frame_contact,
            contact_rew_function=contact_rew_function,
        )
    else:
        contact_rew = torch.zeros(env.num_envs, device=env.device)
    reward += contact_rew

    info = {
        "obj_pos_dist": obj_pos_dist,
        "obj_quat_dist": obj_quat_dist,
        "obj_arti_dist": obj_arti_dist,
        "obj_dist_rew": obj_dist_rew,
        "obj_arti_rew": obj_arti_rew,
        "imi_rew": imi_rew,
        "contact_rew": contact_rew,
    }

    return reward, info


def get_terminal_reward(
    config: Config,
    env: BaseEnv,
    ref_slice: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Terminal reward focusing on object tracking."""
    rew, info = get_reward(config, env, ref_slice)
    return config.terminal_rew_scale * rew, info


def get_trace(config: Config, env: BaseEnv) -> torch.Tensor:
    """Return per-world trace points used for visualization.
    Returns shape (N, num_trace_points, 3)
    """
    # Get finger tip positions
    left_finger_tip_pos = env.robots["left"].entity.get_links_pos()[
        :, env.robots["left"].kpt_link_idxs, :
    ]
    right_finger_tip_pos = env.robots["right"].entity.get_links_pos()[
        :, env.robots["right"].kpt_link_idxs, :
    ]
    # Get object positions
    obj_pos = env.objects[env.object_names[0]].entity.get_pos()
    obj_link_pos = env.objects[env.object_names[0]].entity.get_links_pos()
    # Concatenate trace points
    trace = torch.cat(
        [obj_pos.unsqueeze(1), obj_link_pos, left_finger_tip_pos, right_finger_tip_pos],
        dim=1,
    )
    return trace


def get_qpos(env: BaseEnv) -> torch.Tensor:
    """Get the qpos of the environment.
    Returns object pos + object quat + object dof pos
    """
    obj_pos = env.objects[env.object_names[0]].entity.get_pos()
    obj_quat = env.objects[env.object_names[0]].entity.get_quat()
    obj_dofs_pos = env.objects[env.object_names[0]].entity.get_dofs_position()
    obj_qpos = torch.cat([obj_pos, obj_quat, obj_dofs_pos], dim=1)
    return obj_qpos


def save_state(env: BaseEnv):
    """Save the state of the environment."""
    # Record initial state
    initial_state = env.scene.get_state()
    initial_robot_episode_length_buf = {}
    initial_env_episode_length = env.episode_length_buf.clone()
    for k, robot in env.robots.items():
        initial_robot_episode_length_buf[k] = robot.episode_length_buf.clone()
    initial_obj_episode_length_buf = {}
    for k, obj in env.objects.items():
        initial_obj_episode_length_buf[k] = obj.episode_length_buf.clone()
    return (
        initial_state,
        initial_robot_episode_length_buf,
        initial_env_episode_length,
        initial_obj_episode_length_buf,
    )


def load_state(env: BaseEnv, state):
    """Load the state of the environment."""
    (
        initial_state,
        initial_robot_episode_length_buf,
        initial_env_episode_length,
        initial_obj_episode_length_buf,
    ) = state

    env.scene.reset(state=initial_state)
    env.episode_length_buf = initial_env_episode_length
    for k, robot in env.robots.items():
        robot.episode_length_buf = initial_robot_episode_length_buf[k]
    for k, obj in env.objects.items():
        obj.episode_length_buf = initial_obj_episode_length_buf[k]
    return env


def step_env(config: Config, env: BaseEnv, ctrl: torch.Tensor):
    """Step all worlds with provided controls of shape (N, nu)."""
    if ctrl.dim() == 1:
        ctrl = ctrl.unsqueeze(0).repeat(int(config.num_samples), 1)

    ctrl = ctrl * env.action_scale
    for k, robot in env.robots.items():
        idxs = env.action_idxs_to_robot[k]
        robot.step(ctrl[:, idxs], env._step_env_idxs)
    for obj in env.objects.values():
        obj.step()
    env.scene.step()
    env.episode_length_buf += 1
    env._compute_intermediate_values()
    return env


def save_env_params(config: Config, env: BaseEnv):
    """Save the current simulation parameters."""
    kp = env.objects[env.object_names[0]].entity.get_dofs_kp()
    kv = env.objects[env.object_names[0]].entity.get_dofs_kv()
    return {"kp": kp.clone(), "kv": kv.clone()}


def load_env_params(config: Config, env: BaseEnv, env_param: dict):
    """Load the simulation parameters.

    Parameters to be updated:
    - kp: object joint stiffness
    - kv: object joint damping
    """
    if "kp" in env_param:
        env.objects[env.object_names[0]].entity.set_dofs_kp(env_param["kp"])
    if "kv" in env_param:
        env.objects[env.object_names[0]].entity.set_dofs_kv(env_param["kv"])
    return env


def copy_sample_state(config: Config, env: BaseEnv, src_indices: torch.Tensor, dst_indices: torch.Tensor):
    """Copy simulation state from source samples to destination samples.

    Args:
        config: Config
        env: BaseEnv environment
        src_indices: Tensor of shape (n,) containing source sample indices
        dst_indices: Tensor of shape (n,) containing destination sample indices
    """
    # Convert to numpy for indexing
    src_idx = src_indices.cpu().numpy()
    dst_idx = dst_indices.cpu().numpy()

    # Get all entities dof positions and velocities
    obj_dofs_pos = env.objects[env.object_names[0]].entity.get_dofs_position()
    left_hand_dofs_pos = env.robots["left"].entity.get_dofs_position()
    right_hand_dofs_pos = env.robots["right"].entity.get_dofs_position()
    obj_dofs_vel = env.objects[env.object_names[0]].entity.get_dofs_velocity()
    left_hand_dofs_vel = env.robots["left"].entity.get_dofs_velocity()
    right_hand_dofs_vel = env.robots["right"].entity.get_dofs_velocity()

    # Copy from src to dst
    obj_dofs_pos[dst_idx] = obj_dofs_pos[src_idx]
    left_hand_dofs_pos[dst_idx] = left_hand_dofs_pos[src_idx]
    right_hand_dofs_pos[dst_idx] = right_hand_dofs_pos[src_idx]
    obj_dofs_vel[dst_idx] = obj_dofs_vel[src_idx]
    left_hand_dofs_vel[dst_idx] = left_hand_dofs_vel[src_idx]
    right_hand_dofs_vel[dst_idx] = right_hand_dofs_vel[src_idx]

    # Set all entities dof positions and velocities
    env.objects[env.object_names[0]].entity.set_dofs_position(obj_dofs_pos)
    env.robots["left"].entity.set_dofs_position(left_hand_dofs_pos)
    env.robots["right"].entity.set_dofs_position(right_hand_dofs_pos)
    env.objects[env.object_names[0]].entity.set_dofs_velocity(obj_dofs_vel)
    env.robots["left"].entity.set_dofs_velocity(left_hand_dofs_vel)
    env.robots["right"].entity.set_dofs_velocity(right_hand_dofs_vel)


def sync_env(config: Config, env: BaseEnv, mj_data=None):
    """Broadcast the state from first env to all envs.

    This function synchronizes states from the first environment to all environments.
    """
    # Get all entities dof positions and velocities
    obj_dofs_pos = env.objects[env.object_names[0]].entity.get_dofs_position()
    left_hand_dofs_pos = env.robots["left"].entity.get_dofs_position()
    right_hand_dofs_pos = env.robots["right"].entity.get_dofs_position()
    obj_dofs_vel = env.objects[env.object_names[0]].entity.get_dofs_velocity()
    left_hand_dofs_vel = env.robots["left"].entity.get_dofs_velocity()
    right_hand_dofs_vel = env.robots["right"].entity.get_dofs_velocity()

    # Broadcast to all envs
    obj_dofs_pos = obj_dofs_pos[:1].repeat(int(config.num_samples), 1)
    left_hand_dofs_pos = left_hand_dofs_pos[:1].repeat(int(config.num_samples), 1)
    right_hand_dofs_pos = right_hand_dofs_pos[:1].repeat(int(config.num_samples), 1)
    obj_dofs_vel = obj_dofs_vel[:1].repeat(int(config.num_samples), 1)
    left_hand_dofs_vel = left_hand_dofs_vel[:1].repeat(int(config.num_samples), 1)
    right_hand_dofs_vel = right_hand_dofs_vel[:1].repeat(int(config.num_samples), 1)

    # Set all entities dof positions and velocities
    env.objects[env.object_names[0]].entity.set_dofs_position(obj_dofs_pos)
    env.robots["left"].entity.set_dofs_position(left_hand_dofs_pos)
    env.robots["right"].entity.set_dofs_position(right_hand_dofs_pos)
    env.objects[env.object_names[0]].entity.set_dofs_velocity(obj_dofs_vel)
    env.robots["left"].entity.set_dofs_velocity(left_hand_dofs_vel)
    env.robots["right"].entity.set_dofs_velocity(right_hand_dofs_vel)

    return env
