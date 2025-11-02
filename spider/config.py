"""Define the configuration for the optimizer.

Author: Chaoyi Pan
Date: 2025-08-10
"""

import json
import os
from dataclasses import dataclass, field

import loguru
import mujoco
import numpy as np
import torch

import spider
from spider.io import get_processed_data_dir


@dataclass
class Config:
    # === TASK CONFIGURATION ===
    robot_type: str = "allegro"  # "inspire", "allegro", "g1"
    embodiment_type: str = "bimanual"  # "left", "right", "bimanual", "CMU"
    task: str = "pick_spoon_bowl"

    # === DATASET CONFIGURATION ===
    dataset_dir: str = f"{spider.ROOT}/../example_datasets"
    dataset_name: str = "oakink"
    data_id: int = 0
    model_path: str = ""
    data_path: str = ""

    # === SIMULATOR CONFIGURATION ===
    simulator: str = "isaac"  # "isaac" | "mujoco" | "mjwp" | "mjwp_cons" | "mjwp_eq" | "mjwp_cons_eq" | "kinematic"
    device: str = "cuda:0"
    # Simulation timing
    sim_dt: float = 0.01  # simulation timestep
    ctrl_dt: float = 0.2  # control timestep
    ref_dt: float = 1 / 50.0  # reference data timestep
    render_dt: float = 0.02  # rendering timestep
    horizon: float = 1.2  # planning horizon
    knot_dt: float = 0.4  # knot point spacing
    max_sim_steps: int = -1  # maximum simulation steps (-1 for unlimited)
    # Simulation constraints
    nconmax_per_env: int = 80  # max contacts per environment
    njmax_per_env: int = 300  # max joints per environment
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

    # === OPTIMIZER CONFIGURATION ===
    # Sampling parameters
    num_samples: int = 1024
    temperature: float = 1.0
    max_num_iterations: int = 16
    improvement_threshold: float = 0.01
    improvement_check_steps: int = 1
    num_resamples: int = 0  # if resample to get rid of bad samples
    resample_ratio: float = 0.2  # ratio of resamples to num_samples
    # Compilation
    use_torch_compile: bool = True  # use torch.compile for acceleration
    # Noise scheduling
    first_ctrl_noise_scale: float = 0.5
    last_ctrl_noise_scale: float = 1.0
    final_noise_scale: float = 0.1
    exploit_ratio: float = 0.01
    exploit_noise_scale: float = 0.01
    # Noise scaling by component
    joint_noise_scale: float = 0.3
    pos_noise_scale: float = 0.003
    rot_noise_scale: float = 0.003
    # Reward scaling
    base_pos_rew_scale: float = 0.03
    base_rot_rew_scale: float = 0.01
    joint_rew_scale: float = 0.003
    pos_rew_scale: float = 1.0
    rot_rew_scale: float = 0.1
    vel_rew_scale: float = 0.0001
    terminal_rew_scale: float = 1.0

    # === VISUALIZATION CONFIGURATION ===
    show_viewer: bool = True
    viewer: str = "mujoco"  # "mujoco" | "rerun" | "isaac"
    rerun_spawn: bool = False
    save_video: bool = True
    save_info: bool = True

    # === TRACE RECORDING ===
    trace_dt: float = 1 / 50.0
    num_trace_uniform_samples: int = 4
    num_trace_topk_samples: int = 2
    trace_site_ids: list = field(default_factory=list)

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
        noise_scale[:, :, :3] *= config.pos_noise_scale
        noise_scale[:, :, 3:6] *= config.rot_noise_scale
        if config.embodiment_type == "bimanual":
            half_dof = config.nu // 2
            noise_scale[:, :, 6:half_dof] *= config.joint_noise_scale
            noise_scale[:, :, half_dof : half_dof + 3] *= config.pos_noise_scale
            noise_scale[:, :, half_dof + 3 : half_dof + 6] *= config.rot_noise_scale
            noise_scale[:, :, half_dof + 6 :] *= config.joint_noise_scale
        elif config.embodiment_type in ["right", "left"]:
            noise_scale[:, :, 6:] *= config.joint_noise_scale
    else:
        noise_scale *= config.joint_noise_scale
    # repeat to match num_samples; same samples used across DR groups
    noise_scale = noise_scale.repeat(config.num_samples, 1, 1)
    # set first sample to 0
    noise_scale[0] *= 0.0
    # set last few samples to exploit_noise_scale
    num_exploit_samples = int(config.num_samples * config.exploit_ratio)
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
    scene_xml = "scene.xml" if config.num_dyn == 1 else "scene_eq.xml"
    config.model_path = f"{processed_dir_robot}/../{scene_xml}"
    # default to MJWP retargeted trajectory if available
    config.data_path = f"{processed_dir_robot}/trajectory_kinematic.npz"

    # get model data
    if config.simulator == "mjwp":
        model = mujoco.MjModel.from_xml_path(config.model_path)
        config.nq = model.nq
        config.nv = model.nv
        config.nu = model.nu
        config.npair = model.npair

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

    return config
