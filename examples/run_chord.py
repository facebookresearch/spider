# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Retarget a motion with SPIDER's sampling MPC in FlashCHORD's GPU environment.

The GPU twin of ``run_cpu.py``: the same loop, recovery, viewer and records
(``run_cpu.track``) on :mod:`spider.simulators.chord`, FlashCHORD's own
environment (its physics, control law and contact terms). Needs ``flash_chord``
and a GPU. ``spider.simulators.chord_export`` turns its scene into one for the
CPU workflow.

Example:
    python examples/run_chord.py "data_path='/path/to/robot_name=sharpa_wave'"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cpu  # noqa: E402

from spider.config import Config  # noqa: E402
from spider.io import get_processed_data_dir  # noqa: E402
from spider.simulators import chord  # noqa: E402


def resolve_paths(config: Config) -> Config:
    """The reference is ``data_path``; results go to the dataset layout's trial dir."""
    if not config.data_path:
        raise ValueError("run_chord.py needs data_path=<FlashCHORD reference>")
    trial_dir = get_processed_data_dir(
        dataset_dir=os.path.abspath(config.dataset_dir),
        dataset_name=config.dataset_name,
        robot_type=config.robot_type,
        embodiment_type=config.embodiment_type,
        task=config.task,
        data_id=config.data_id,
    )
    config.output_dir = str(Path(config.output_dir or trial_dir).expanduser().resolve())
    os.makedirs(config.output_dir, exist_ok=True)
    return config


def reference(config: Config, env: chord.ChordEnv) -> tuple[torch.Tensor, ...]:
    """The reference as ``run_cpu`` takes it: FlashCHORD's, in MuJoCo coordinates.

    The worlds keep their own reference clock, so only the ``qpos`` is real; the
    ``ctrl`` of a replay is a zero residual.
    """
    frames = len(env.mj_qpos_ref)
    return (
        torch.as_tensor(env.mj_qpos_ref, dtype=torch.float64),
        torch.zeros(frames, env.mj_model.nv, dtype=torch.float64),
        torch.zeros(frames, env.nu, device=config.device),
        torch.zeros(frames, 0),
        torch.zeros(frames, 0, 3),
    )


def retarget(
    config: Config,
    recovery: run_cpu.Recovery | None = None,
    num_threads: int = 0,
    panel: run_cpu.Panel | None = None,
):
    """Run sampling MPC over the reference and save the executed trajectory."""
    del num_threads  # the GPU runs every world at once
    config = resolve_paths(config)
    env = chord.setup_env(config)
    return run_cpu.track(config, recovery, chord, env, reference(config, env), panel)


@hydra.main(version_base=None, config_path="config", config_name="chord")
def run_main(cfg: DictConfig) -> None:
    """Hydra entry point."""
    config, recovery, num_threads, autostart = run_cpu.parse(cfg)
    run_cpu.main(config, recovery, num_threads, autostart, run=retarget)


if __name__ == "__main__":
    run_main()
