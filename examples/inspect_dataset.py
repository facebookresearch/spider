# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Licensed under the license in the repository root.

"""Inspect downloaded SPIDER rollouts in Viser, without running optimization."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import mujoco
import tyro

from spider.trajectory import discover_trajectories, load_saved_trajectory
from spider.viewers import viser_viewer


def main(
    dataset_dir: Path = Path("example_datasets"),
    dataset_name: str = "",
    robot_type: str = "",
    task: str = "",
    data_type: str = "mjwp",
    host: str = "0.0.0.0",
    port: int = 8080,
) -> None:
    """Browse trials, scrub frames, and compare saved simulation with its IK reference."""
    rows = [
        r
        for r in discover_trajectories(dataset_dir, data_type)
        if (not dataset_name or r["dataset_name"] == dataset_name)
        and (not robot_type or r["robot_type"] == robot_type)
        and (not task or r["task"] == task)
    ]
    if not rows:
        raise ValueError(
            f"No matching trajectory_{data_type}.npz files under {dataset_dir}"
        )
    labels = [
        f"{r['dataset_name']} / {r['robot_type']} / {r['embodiment_type']} / {r['task']} / {r['data_id']}"
        for r in rows
    ]
    trials_by_label = dict(zip(labels, rows, strict=True))
    server = viser_viewer.init_viser("SPIDER dataset inspector", host=host, port=port)
    server.scene.set_up_direction("+z")
    lock = threading.RLock()
    current = {}
    with server.gui.add_folder("Dataset"):
        choice = server.gui.add_dropdown("Trial", options=labels)
        details = server.gui.add_markdown("Loading…")
    with server.gui.add_folder("Playback"):
        playing = server.gui.add_checkbox("Play", initial_value=False)
        speed = server.gui.add_slider(
            "Speed", min=0.1, max=2.0, step=0.1, initial_value=1.0
        )
        frame = server.gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)

    def display() -> None:
        with lock:
            if not current:
                return
            run = current["run"]
            index = min(int(frame.value), len(run.qpos) - 1)
            data, ref = current["data"], current["ref"]
            data.qpos[:] = run.qpos[index]
            data.qvel[:] = run.qvel[index]
            data.ctrl[:] = run.ctrl[index]
            ref.qpos[:] = run.reference[index]
            mujoco.mj_forward(run.model, data)
            mujoco.mj_forward(run.model, ref)
            viser_viewer.update_frame(data, current["handles"], ref)

    def select() -> None:
        with lock:
            playing.value = False
            row = trials_by_label[choice.value]
            run = load_saved_trajectory(dataset_dir, **row, data_type=data_type)
            viser_viewer.reset_scene()
            spec = mujoco.MjSpec.from_file(run.config.model_path)
            handles = viser_viewer.build_and_log_scene_from_spec(
                spec,
                run.model,
                Path(run.config.model_path),
                build_ref=True,
            )
            current.update(
                run=run,
                handles=handles,
                data=mujoco.MjData(run.model),
                ref=mujoco.MjData(run.model),
            )
            frame.max = len(run.qpos) - 1
            frame.value = 0
            details.content = (
                f"**{row['dataset_name']} · {row['robot_type']}**\n\n"
                f"{len(run.qpos)} frames · {1 / run.config.sim_dt:g} Hz\n\n"
                "Robot = saved simulation; blue overlay = IK reference.\n\n"
                f"Recorded success: {run.metrics.get('success', 'unavailable')}\n\n"
                "State replay, not a new dynamics rollout."
            )
            display()

    @choice.on_update
    def _select(_) -> None:
        try:
            select()
        except Exception as error:
            playing.value = False
            details.content = f"**Cannot load trial:** {error}"

    @frame.on_update
    def _frame(_) -> None:
        display()

    @server.on_client_connect
    def _camera(client) -> None:
        client.camera.position = (0.7, -0.7, 0.55)
        client.camera.look_at = (0.0, 0.0, 0.15)

    select()
    print(
        f"Inspector ready: http://localhost:{server.get_port()} ({len(rows)} trials)",
        flush=True,
    )
    try:
        while True:
            with lock:
                if playing.value and current:
                    frame.value = (int(frame.value) + 1) % len(current["run"].qpos)
                delay = (
                    current["run"].config.sim_dt / float(speed.value)
                    if current
                    else 0.02
                )
            time.sleep(delay)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    tyro.cli(main)
