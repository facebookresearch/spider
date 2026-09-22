# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Licensed under the license in the repository root.

"""Read saved trajectories through the current SPIDER configuration and IO path."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

import mujoco
import numpy as np
import yaml

from spider.config import Config, process_config
from spider.io import get_processed_data_dir, load_data


@dataclass
class SavedTrajectory:
    """A validated saved rollout and the reference loaded by SPIDER."""

    config: Config
    model: mujoco.MjModel
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    times: np.ndarray
    reference: np.ndarray
    reference_source: np.ndarray
    metrics: dict


def load_saved_trajectory(
    dataset_dir: Path,
    dataset_name: str,
    robot_type: str,
    embodiment_type: str,
    task: str,
    data_id: int = 0,
    data_type: str = "mjwp",
) -> SavedTrajectory:
    """Load and validate a standard processed trial without a GPU or simulator loop."""
    root = dataset_dir.resolve()
    trial = Path(
        get_processed_data_dir(
            str(root),
            dataset_name,
            robot_type,
            embodiment_type,
            task,
            data_id,
        )
    )
    trial.resolve().relative_to(root)
    settings = {}
    config_path = trial / "config.yaml"
    if config_path.exists():
        allowed = {field.name for field in fields(Config)}
        settings.update(
            {
                k: v
                for k, v in yaml.safe_load(config_path.read_text()).items()
                if k in allowed
            }
        )
    info_path = trial.parent / "task_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        for key in ("ref_dt", "sim_dt"):
            if key in info:
                settings[key] = info[key]
    settings.update(
        dataset_dir=str(root),
        dataset_name=dataset_name,
        robot_type=robot_type,
        embodiment_type=embodiment_type,
        task=task,
        data_id=data_id,
        device="cpu",
        show_viewer=False,
        viewer="",
        save_video=True,
    )
    config = process_config(Config(**settings))
    model = mujoco.MjModel.from_xml_path(config.model_path)
    path = trial / f"trajectory_{data_type}.npz"
    with np.load(path, allow_pickle=False) as data:
        arrays = {}
        for key, width in (("qpos", model.nq), ("qvel", model.nv), ("ctrl", model.nu)):
            value = data[key]
            if value.ndim not in (2, 3) or value.shape[-1] != width:
                raise ValueError(
                    f"{path}: invalid {key} shape {value.shape}; expected width {width}"
                )
            arrays[key] = value.reshape(-1, width).copy()
            if not np.isfinite(arrays[key]).all():
                raise ValueError(f"{path}: nonfinite {key}")
        length = len(arrays["qpos"])
        if length < 2 or any(len(a) != length for a in arrays.values()):
            raise ValueError(f"{path}: empty or inconsistent trajectory lengths")
        times = (
            data["time"].reshape(-1).copy()
            if "time" in data
            else np.arange(length) * config.sim_dt
        )
    if (
        len(times) != length
        or not np.isfinite(times).all()
        or not np.allclose(np.diff(times), config.sim_dt, atol=1e-5)
    ):
        raise ValueError(f"{path}: invalid simulation time grid")
    with np.load(config.data_path, allow_pickle=False) as data:
        reference_source = data["qpos"].copy()
    if (
        reference_source.ndim != 2
        or reference_source.shape[-1] != model.nq
        or not np.isfinite(reference_source).all()
    ):
        raise ValueError(f"{config.data_path}: invalid reference")
    # Exercise the same loader and interpolation used by the optimizer.
    loaded_reference = load_data(config, config.data_path)
    if not all(np.isfinite(tensor.numpy()).all() for tensor in loaded_reference):
        raise ValueError(
            f"{config.data_path}: invalid interpolated reference/control/contact"
        )
    reference = loaded_reference[0].numpy()
    if len(reference) < length:
        reference = np.concatenate(
            [reference, np.repeat(reference[-1:], length - len(reference), axis=0)]
        )
    metrics_path = trial / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    return SavedTrajectory(
        config,
        model,
        **arrays,
        times=times,
        reference=reference[:length],
        reference_source=reference_source,
        metrics=metrics,
    )


def discover_trajectories(dataset_dir: Path, data_type: str = "mjwp") -> list[dict]:
    """List dataset/robot/embodiment/task/trial identifiers in the standard layout."""
    rows = []
    for path in sorted(
        dataset_dir.glob(f"processed/*/*/*/*/*/trajectory_{data_type}.npz")
    ):
        _, dataset, robot, side, task, trial, _ = path.relative_to(dataset_dir).parts
        if trial.isdecimal():
            rows.append(
                {
                    "dataset_name": dataset,
                    "robot_type": robot,
                    "embodiment_type": side,
                    "task": task,
                    "data_id": int(trial),
                }
            )
    return rows
