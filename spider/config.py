# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Define the configuration for the optimizer.

Author: Chaoyi Pan
Date: 2025-08-10
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field, fields

import loguru
import mujoco
import numpy as np
import torch
from omegaconf import OmegaConf

import spider
from spider.io import get_processed_data_dir


@dataclass
class Config:
    # === TASK CONFIGURATION ===
    robot_type: str = "xhand"  # "inspire", "allegro", "g1"
    embodiment_type: str = "bimanual"  # "left", "right", "bimanual", "CMU"
    task: str = "pick_spoon_bowl"
    seed: int = 0

    # === SANITY CHECK CONFIGURATION ===
    # Holds controls at ctrl_ref[0] for sanity_check_seconds and verifies the
    # object does not drift more than the thresholds. Set seconds to 0.0 to
    # disable the check.
    sanity_check_seconds: float = 3.0
    sanity_check_pos_thresh: float = 0.01
    sanity_check_rot_thresh: float = 0.3

    # === DATASET CONFIGURATION ===
    dataset_dir: str = f"{spider.ROOT}/../example_datasets"
    dataset_name: str = "oakink"
    data_id: int = 0
    model_path: str = ""
    data_path: str = ""
    # Optional config loader (used by CLI runner)
    load_config_path: str = ""

    # === SIMULATOR CONFIGURATION ===
    simulator: str = "cpu"  # "cpu" | "mjwp" | "mjwp_cons" | "mjwp_eq" | "mjwp_cons_eq" | "isaac" | "mujoco" | "kinematic" | "chord" | ...
    device: str = "cuda:0"
    # Simulation timing
    sim_dt: float = 0.01  # simulation timestep
    ctrl_dt: float = 0.4  # control timestep
    ref_dt: float = 0.02  # reference data timestep
    render_dt: float = 0.02  # rendering timestep
    horizon: float = 1.6  # planning horizon
    knot_dt: float = 0.4  # knot point spacing
    max_sim_steps: int = -1  # maximum simulation steps (-1 for unlimited)
    # Simulation constraints
    nconmax_per_env: int = 100  # max contacts per environment
    njmax_per_env: int = 350  # max joints per environment
    # Simulation annealing
    num_dyn: int = (
        1  # number of environments for annealing, used for virtual contact constraint
    )
    # Domain randomization
    num_dr: int = (
        1  # number of domain randomization groups, used for domain randomization
    )
    pair_margin_range: tuple[float, float] = (-0.005, 0.005)
    xy_offset_range: tuple[float, float] = (-0.005, 0.005)
    perturb_force: float = 0.0
    perturb_torque: float = 0.0
    contact_guidance: bool = False
    object_pos_actuator_names: list[str] = field(
        default_factory=lambda: [
            "right_object_pos_x",
            "right_object_pos_y",
            "right_object_pos_z",
            "left_object_pos_x",
            "left_object_pos_y",
            "left_object_pos_z",
        ]
    )
    object_rot_actuator_names: list[str] = field(
        default_factory=lambda: [
            "right_object_rot_x",
            "right_object_rot_y",
            "right_object_rot_z",
            "left_object_rot_x",
            "left_object_rot_y",
            "left_object_rot_z",
        ]
    )
    object_action_dims: int = 0
    object_actuator_ids: list[int] = field(default_factory=list)
    object_actuator_names: list[str] = field(default_factory=list)
    init_pos_actuator_gain: float = 10.0
    init_pos_actuator_bias: float = 10.0
    init_rot_actuator_gain: float = 0.1
    init_rot_actuator_bias: float = 0.1
    guidance_decay_ratio: float = 0.5
    gibbs_sampling: bool = False

    # === OPTIMIZER CONFIGURATION ===
    # Sampling parameters
    num_samples: int = 2048
    temperature: float = 0.3
    max_num_iterations: int = 16
    improvement_threshold: float = 0.01
    improvement_check_steps: int = 1
    # Termination parameters
    terminate_resample: bool = False
    object_pos_threshold: float = 0.1
    object_rot_threshold: float = 0.3
    max_local_retries: int = 3
    max_revert_depth: int = 3
    base_pos_threshold: float = 0.5
    base_rot_threshold: float = 0.4
    # Compilation
    use_torch_compile: bool = True  # use torch.compile for acceleration
    # Noise scheduling
    first_ctrl_noise_scale: float = 0.5
    last_ctrl_noise_scale: float = 1.0
    final_noise_scale: float = 0.1
    exploit_ratio: float = 0.01
    exploit_noise_scale: float = 0.01
    # Noise scaling by component
    joint_noise_scale: float = 0.15
    pos_noise_scale: float = 0.03
    rot_noise_scale: float = 0.03
    # Reward mode
    use_rl_reward: bool = False  # use dexmachina RL training reward formulation
    # Reward scaling
    base_pos_rew_scale: float = 1.0
    base_rot_rew_scale: float = 0.3
    joint_rew_scale: float = 0.003
    pos_rew_scale: float = 1.0
    rot_rew_scale: float = 0.3
    vel_rew_scale: float = 0.0001
    terminal_rew_scale: float = 1.0
    contact_rew_scale: float = 0.0

    # === VISUALIZATION CONFIGURATION ===
    show_viewer: bool = True
    viewer: str = "viser"  # "mujoco" | "rerun" | "viser" | "isaac"
    wait_on_finish: bool = True  # block after optimization to keep viewer alive
    rerun_spawn: bool = False
    save_video: bool = True
    save_info: bool = True
    save_rerun: bool = False
    save_viser: bool = False
    save_metrics: bool = True
    save_config: bool = True

    # === TRACE RECORDING ===
    trace_dt: float = 1 / 50.0
    num_trace_uniform_samples: int = 4
    num_trace_topk_samples: int = 2
    trace_site_ids: list = field(default_factory=list)

    # === CONTACT GUIDANCE (DERIVED) ===
    contact_order: list = field(default_factory=list)
    hand_contact_site_ids: list = field(default_factory=list)
    right_contact_indices: list = field(default_factory=list)
    left_contact_indices: list = field(default_factory=list)
    right_pos_ctrl_ids: list = field(default_factory=list)
    left_pos_ctrl_ids: list = field(default_factory=list)
    contact_len: int = 0

    # === AUTOMATICALLY SET PROPERTIES ===
    # Computed timesteps
    horizon_steps: int = -1
    knot_steps: int = -1
    ref_steps: int = -1
    ctrl_steps: int = -1
    # Model dimensions
    nq_obj: int = -1  # object DOF
    nq: int = -1  # total position DOF
    nv: int = -1  # total velocity DOF
    nu: int = -1  # total control DOF
    npair: int = -1  # total pair DOF
    # Computed tensors
    noise_scale: torch.Tensor = field(default_factory=lambda: torch.ones(1))
    beta_traj: float = -1.0
    # Runtime state
    env_params_list: list = field(default_factory=list)
    viewer_body_entity_and_ids: list = field(default_factory=list)
    output_dir: str = ""


def resolve_object_actuator_ids(
    model: mujoco.MjModel,
    desired_names: list[str],
    object_action_dims: int,
) -> tuple[list[int], list[str]]:
    """Resolve object actuator ids by name, with a fallback to last N actuators."""
    resolved_ids: list[int] = []
    resolved_names: list[str] = []
    for name in desired_names:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid != -1:
            resolved_ids.append(int(aid))
            resolved_names.append(name)
    if resolved_ids:
        if len(resolved_ids) != len(desired_names):
            loguru.logger.info(
                "Resolved {} / {} object actuators by name.",
                len(resolved_ids),
                len(desired_names),
            )
        return resolved_ids, resolved_names

    obj_start = max(model.nu - max(object_action_dims, 0), 0)
    fallback_ids = list(range(obj_start, model.nu))
    fallback_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(aid))
        for aid in fallback_ids
    ]
    loguru.logger.warning(
        "No named object actuators found; falling back to last {} actuators.",
        len(fallback_ids),
    )
    return fallback_ids, fallback_names


def load_config_yaml(path: str) -> dict:
    """Load a config YAML into a plain dict, resolving OmegaConf types."""
    if not path:
        return {}
    abs_path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"Config file not found: {abs_path}")
    cfg = OmegaConf.load(abs_path)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise ValueError(f"Config at {abs_path} did not resolve to a mapping.")
    cfg_dict.pop("hydra", None)
    return cfg_dict


def filter_config_fields(config_dict: dict) -> dict:
    """Filter config dict keys to those defined in Config."""
    allowed = {field.name for field in fields(Config)}
    return {key: value for key, value in config_dict.items() if key in allowed}


def build_hand_contact_site_ids(
    mj_model: mujoco.MjModel, embodiment_type: str
) -> tuple[list[tuple[str, str]], list[int | None]]:
    contact_order = []
    if embodiment_type in ["bimanual", "right"]:
        contact_order.extend(
            [
                ("right", "thumb"),
                ("right", "index"),
                ("right", "middle"),
                ("right", "ring"),
                ("right", "pinky"),
            ]
        )
    if embodiment_type in ["bimanual", "left"]:
        contact_order.extend(
            [
                ("left", "thumb"),
                ("left", "index"),
                ("left", "middle"),
                ("left", "ring"),
                ("left", "pinky"),
            ]
        )

    site_ids: list[int | None] = [None] * len(contact_order)
    for sid in range(mj_model.nsite):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SITE, sid)
        if name is None:
            continue
        name_l = name.lower()
        if "track" not in name_l or "hand" not in name_l:
            continue
        for idx, (side, finger) in enumerate(contact_order):
            if side in name_l and finger in name_l:
                if site_ids[idx] is None:
                    site_ids[idx] = sid
                break

    missing = [i for i, sid in enumerate(site_ids) if sid is None]
    if missing:
        loguru.logger.warning(
            "Missing {} hand contact sites for guidance; indices: {}",
            len(missing),
            missing,
        )
    return contact_order, site_ids


def get_object_pos_ctrl_indices(config: Config) -> tuple[list[int], list[int]]:
    right_ids: list[int] = []
    left_ids: list[int] = []
    if config.object_actuator_ids and config.object_actuator_names:
        for aid, name in zip(
            config.object_actuator_ids, config.object_actuator_names, strict=False
        ):
            name_l = (name or "").lower()
            if "_pos_" not in name_l:
                continue
            if "right" in name_l:
                right_ids.append(int(aid))
            elif "left" in name_l:
                left_ids.append(int(aid))

    if right_ids or left_ids:
        return right_ids, left_ids

    obj_dims = int(config.object_action_dims) if config.object_action_dims > 0 else 0
    if obj_dims == 0:
        obj_dims = 12 if config.embodiment_type == "bimanual" else 6
    start = max(int(config.nu) - obj_dims, 0)
    if config.embodiment_type == "bimanual" and obj_dims >= 12:
        right_ids = list(range(start, start + 3))
        left_ids = list(range(start + 3, start + 6))
    else:
        if config.embodiment_type == "right":
            right_ids = list(range(start, start + 3))
        elif config.embodiment_type == "left":
            left_ids = list(range(start, start + 3))
        else:
            right_ids = list(range(start, start + 3))
    return right_ids, left_ids


def get_noise_scale(config: Config) -> torch.Tensor:
    """Get the noise scale for sampling.

    Args:
        config: Config

    Returns:
        Noise scale, shape (num_samples, knot_steps, nu)
    """
    noise_scale = torch.logspace(
        start=torch.log10(torch.tensor(config.first_ctrl_noise_scale)),
        end=torch.log10(torch.tensor(config.last_ctrl_noise_scale)),
        steps=int(round(config.horizon / config.knot_dt)),
        device=config.device,
        base=10,
    )[None, :, None]  # Shape: (1, num_knot_steps, 1)
    noise_scale = noise_scale.repeat(1, 1, config.nu)
    if config.embodiment_type in ["bimanual", "right", "left"]:
        object_action_dims = max(int(config.object_action_dims), 0)
        robot_nu = max(int(config.nu - object_action_dims), 0)
        noise_scale[:, :, :3] *= config.pos_noise_scale
        noise_scale[:, :, 3:6] *= config.rot_noise_scale
        if config.embodiment_type == "bimanual":
            half_dof = robot_nu // 2
            noise_scale[:, :, 6:half_dof] *= config.joint_noise_scale
            noise_scale[:, :, half_dof : half_dof + 3] *= config.pos_noise_scale
            noise_scale[:, :, half_dof + 3 : half_dof + 6] *= config.rot_noise_scale
            noise_scale[:, :, half_dof + 6 : robot_nu] *= config.joint_noise_scale
        elif config.embodiment_type in ["right", "left"]:
            noise_scale[:, :, 6:robot_nu] *= config.joint_noise_scale
    else:
        noise_scale *= config.joint_noise_scale
    if config.contact_guidance and config.object_actuator_ids:
        object_ids = torch.as_tensor(
            config.object_actuator_ids, device=config.device, dtype=torch.long
        )
        noise_scale[:, :, object_ids] *= 0.0
    # repeat to match num_samples; same samples used across DR groups
    noise_scale = noise_scale.repeat(config.num_samples, 1, 1)
    # set first sample to 0
    noise_scale[0] *= 0.0
    # set last few samples to exploit_noise_scale
    num_exploit_samples = int(config.num_samples * config.exploit_ratio)
    if num_exploit_samples > 0:  # [-0:] would be every sample
        noise_scale[-num_exploit_samples:] *= config.exploit_noise_scale
    return noise_scale


def compute_steps(config: Config):
    # make sure every dt can be divided by sim_dt
    config.horizon_steps = int(np.round(config.horizon / config.sim_dt))
    config.knot_steps = int(np.round(config.knot_dt / config.sim_dt))
    config.ref_steps = int(np.round(config.ref_dt / config.sim_dt))
    config.ctrl_steps = int(np.round(config.ctrl_dt / config.sim_dt))
    assert np.isclose(
        config.horizon - config.horizon_steps * config.sim_dt, 0, atol=1e-5
    ), "horizon must be divisible by sim_dt"
    assert np.isclose(
        config.ctrl_dt - config.ctrl_steps * config.sim_dt, 0, atol=1e-5
    ), "ctrl_dt must be divisible by sim_dt"
    assert np.isclose(
        config.knot_dt - config.knot_steps * config.sim_dt, 0, atol=1e-5
    ), "knot_dt must be divisible by sim_dt"
    return config


def compute_noise_schedule(config: Config) -> Config:
    config.noise_scale = get_noise_scale(config)
    if config.max_num_iterations > 0:
        config.beta_traj = config.final_noise_scale ** (1 / config.max_num_iterations)
    else:
        config.beta_traj = 1.0
    return config


def process_config(config: Config):
    """Process the configuration to fill in the missing fields."""
    config = compute_steps(config)
    trace_steps_tmp = int(np.round(config.trace_dt / config.sim_dt))
    assert np.isclose(
        config.trace_dt - trace_steps_tmp * config.sim_dt, 0, atol=1e-3
    ), "trace_dt must be divisible by sim_dt"

    # Set object DOF based on hand type
    if config.contact_guidance:
        config.nq_obj = {
            "bimanual": 12,
            "right": 6,
            "left": 6,
        }.get(config.embodiment_type, 0)
    else:
        config.nq_obj = {
            "bimanual": 14,
            "right": 7,
            "left": 7,
        }.get(config.embodiment_type, 0)

    # resolve processed directories for this trial
    dataset_dir_abs = os.path.abspath(config.dataset_dir)
    processed_dir_robot = get_processed_data_dir(
        dataset_dir=dataset_dir_abs,
        dataset_name=config.dataset_name,
        robot_type=config.robot_type,
        embodiment_type=config.embodiment_type,
        task=config.task,
        data_id=config.data_id,
    )
    # model and data within processed directory (scene_eq.xml support for annealing over equality constraints)
    if config.contact_guidance:
        scene_xml = "scene_act.xml"
    else:
        scene_xml = "scene.xml" if config.num_dyn == 1 else "scene_eq.xml"
    config.model_path = f"{processed_dir_robot}/../{scene_xml}"
    # default to MJWP retargeted trajectory if available
    if config.contact_guidance:
        config.data_path = f"{processed_dir_robot}/trajectory_kinematic_act.npz"
    else:
        config.data_path = f"{processed_dir_robot}/trajectory_kinematic.npz"

    # get model data
    if config.simulator == "mjwp":
        model = mujoco.MjModel.from_xml_path(config.model_path)
        config.nq = model.nq
        config.nv = model.nv
        config.nu = model.nu
        config.npair = model.npair
        if config.contact_guidance:
            if config.object_action_dims <= 0:
                config.object_action_dims = (
                    12 if config.embodiment_type == "bimanual" else 6
                )
            desired_names = (
                config.object_pos_actuator_names + config.object_rot_actuator_names
            )
            object_ids, object_names = resolve_object_actuator_ids(
                model,
                desired_names,
                config.object_action_dims,
            )
            config.object_actuator_ids = object_ids
            config.object_actuator_names = object_names
            config.right_pos_ctrl_ids, config.left_pos_ctrl_ids = (
                get_object_pos_ctrl_indices(config)
            )
            config.contact_order, config.hand_contact_site_ids = (
                build_hand_contact_site_ids(model, config.embodiment_type)
            )
            config.right_contact_indices = [
                idx
                for idx, (side, finger) in enumerate(config.contact_order)
                if (side == "right") and (finger in ["thumb"])
            ]
            config.left_contact_indices = [
                idx
                for idx, (side, finger) in enumerate(config.contact_order)
                if side == "left" and (finger in ["thumb"])
            ]

    # get noise scale
    config = compute_noise_schedule(config)

    # output dir: write artifacts alongside the trial
    config.output_dir = processed_dir_robot
    os.makedirs(config.output_dir, exist_ok=True)

    # read task info
    task_info_path = f"{processed_dir_robot}/../task_info.json"
    try:
        with open(task_info_path, encoding="utf-8") as f:
            task_info = json.load(f)
    except FileNotFoundError:
        loguru.logger.warning(
            f"task_info.json not found at {task_info_path}, using default values"
        )
        task_info = {}
    if "ref_dt" in task_info:
        config.ref_dt = task_info["ref_dt"]
        loguru.logger.info(f"overriding ref_dt: {config.ref_dt} from task_info.json")

    # override contact site ids
    if config.contact_rew_scale > 0.0:
        if "contact_site_ids" in task_info:
            config.contact_site_ids = task_info["contact_site_ids"]
            loguru.logger.info(
                f"overriding contact_site_ids: {config.contact_site_ids} from task_info.json"
            )
        else:
            raise ValueError(
                "contact_site_ids not found in task_info.json while contact_rew_scale > 0.0"
            )

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    random.seed(config.seed)

    return config
