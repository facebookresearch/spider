"""
Define functions to load and save the data.

Author: Chaoyi Pan
Date: 2025-08-10
"""

from __future__ import annotations
import os
from typing import TYPE_CHECKING
import numpy as np
import loguru
import torch
from spider.interp import interp

if TYPE_CHECKING:
    from spider.config import Config


def load_data(
    config: Config,
    data_path: str = "../datasets/inspire/bimanual_wipe.npz",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load trajectory data from NPZ file."""
    raw_data = np.load(data_path)
    qpos_ref = raw_data["qpos"]
    qvel_ref = raw_data["qvel"]
    try:
        contact = raw_data["contact"]
    except:
        contact = np.zeros((qpos_ref.shape[0], 10))
        loguru.logger.warning("contact data not found")
    try:
        contact_pos = raw_data["contact_pos"]
    except:
        contact_pos = np.zeros((qpos_ref.shape[0], 10, 3))
        loguru.logger.warning("contact_pos data not found")
    if "ctrl" in raw_data:
        ctrl_ref = raw_data["ctrl"]
    else:
        # TODO: disable automatic reference control generation, instead, move it to preprocess
        # Fallback if 'ctrl' is not in the data file
        loguru.logger.warning(
            "ctrl data not found, using 'qpos' as a initial guess for control."
        )
        if config.hand_type in ["bimanual", "right", "left"]:
            ctrl_ref = qpos_ref[:, : -config.nq_obj]
        elif config.hand_type in ["CMU", "DanceDB"]:
            ctrl_ref = qpos_ref[:, 7:]
        else:
            raise ValueError(f"Invalid hand_type: {config.hand_type}")
    # move to device
    qpos_ref_torch = torch.from_numpy(qpos_ref).to(config.device).to(torch.float32)
    qvel_ref_torch = torch.from_numpy(qvel_ref).to(config.device).to(torch.float32)
    ctrl_ref_torch = torch.from_numpy(ctrl_ref).to(config.device).to(torch.float32)
    contact_ref_torch = torch.from_numpy(contact).to(config.device).to(torch.float32)
    contact_pos_ref_torch = (
        torch.from_numpy(contact_pos).to(config.device).to(torch.float32)
    )
    # interpolate to match sim_dt
    if config.ref_dt > config.sim_dt:
        qpos_ref_interp = interp(qpos_ref_torch.unsqueeze(0), config.ref_steps).squeeze(
            0
        )
        qvel_ref_interp = interp(qvel_ref_torch.unsqueeze(0), config.ref_steps).squeeze(
            0
        )
        ctrl_ref_interp = interp(ctrl_ref_torch.unsqueeze(0), config.ref_steps).squeeze(
            0
        )
        contact_ref_interp = interp(
            contact_ref_torch.unsqueeze(0), config.ref_steps
        ).squeeze(0)
        H, Nc, D = contact_pos_ref_torch.shape
        contact_pos_ref_flat = contact_pos_ref_torch.view(H, Nc * D)
        contact_pos_ref_flat_interp = interp(
            contact_pos_ref_flat.unsqueeze(0), config.ref_steps
        ).squeeze(0)
        contact_pos_ref_interp = contact_pos_ref_flat_interp.view(-1, Nc, D)
    else:
        # downsample
        downsample_factor = int(config.sim_dt / config.ref_dt)
        qpos_ref_interp = qpos_ref_torch[::downsample_factor]
        qvel_ref_interp = qvel_ref_torch[::downsample_factor]
        ctrl_ref_interp = ctrl_ref_torch[::downsample_factor]
        contact_ref_interp = contact_ref_torch[::downsample_factor]
        contact_pos_ref_interp = contact_pos_ref_torch[::downsample_factor]
    # repeat the last frame with extra config.horizon_steps
    for _ in range(config.horizon_steps + config.ctrl_steps):
        qpos_ref_interp = torch.cat([qpos_ref_interp, qpos_ref_interp[-1:]], dim=0)
        qvel_ref_interp = torch.cat([qvel_ref_interp, qvel_ref_interp[-1:]], dim=0)
        ctrl_ref_interp = torch.cat([ctrl_ref_interp, ctrl_ref_interp[-1:]], dim=0)
        contact_ref_interp = torch.cat(
            [contact_ref_interp, contact_ref_interp[-1:]], dim=0
        )
        contact_pos_ref_interp = torch.cat(
            [contact_pos_ref_interp, contact_pos_ref_interp[-1:]], dim=0
        )
    return (
        qpos_ref_interp,
        qvel_ref_interp,
        ctrl_ref_interp,
        contact_ref_interp,
        contact_pos_ref_interp,
    )


def get_processed_data_dir(
    dataset_dir: str,
    dataset_name: str,
    robot_type: str,
    hand_type: str,
    task: str,
    data_id: int,
) -> str:
    """Get the data directory from the data path."""
    return f"{dataset_dir}/processed/{dataset_name}/{robot_type}/{hand_type}/{task}/{data_id}"


def get_all_tasks(
    dataset_dir: str,
    dataset_name: str,
    robot_type: str,
    hand_type: str,
) -> list[str]:
    """Get all tasks from the data path."""
    return [
        d
        for d in os.listdir(
            f"{dataset_dir}/processed/{dataset_name}/{robot_type}/{hand_type}"
        )
        if os.path.isdir(
            f"{dataset_dir}/processed/{dataset_name}/{robot_type}/{hand_type}/{d}"
        )
    ]


def get_mesh_dir(dataset_dir: str, dataset_name: str, object_name: str) -> str:
    """Get the mesh directory from the data path."""
    return f"{dataset_dir}/processed/{dataset_name}/assets/objects/{object_name}"
