# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Retarget a motion with SPIDER's sampling MPC on the CPU (``simulator=cpu``).

The runner of SPIDER's CPU workflow: the protocol of ``run_mjwp.py`` (load the
reference, set up the worlds, optimise a receding-horizon plan, execute it,
broadcast the executed world) on :mod:`spider.simulators.cpu`, MuJoCo's CPU
batch API. Needs MuJoCo >= 3.17, NumPy and Torch; no GPU.

The plan is a residual on the reference; SPIDER samples it. When the object
slips, the run plans the frame again and then goes back to before the slip and
plans the stretch again (``recovery`` in ``config/cpu.yaml``).

With the Viser viewer (the default) the run is driven from the browser: pick a
task and the key settings, press Start, and watch the motion, the planned
traces and the metrics. ``autostart=true`` starts the configured task at once;
``show_viewer=false`` runs it without a viewer.

Examples:
    # a scene: example_datasets/processed/chord/sharpa/bimanual/<task>/<data_id>/
    python examples/run_cpu.py
    python examples/run_cpu.py task=s01-laptop-use data_id=2 show_viewer=false
"""

from __future__ import annotations

import copy
import json
import os
import random
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import NamedTuple

import hydra
import imageio
import loguru
import mujoco
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from spider.config import (
    Config,
    compute_noise_schedule,
    compute_steps,
    filter_config_fields,
)
from spider.interp import get_slice
from spider.io import get_processed_data_dir
from spider.optimizers.sampling import (
    make_optimize_fn,
    make_optimize_once_fn,
    make_rollout_fn,
)
from spider.postprocess.evaluate_cpu import evaluate, excursions, followed, summary
from spider.simulators import cpu
from spider.viewers import (
    log_frame,
    render_image,
    setup_renderer,
    setup_viewer,
    update_viewer,
)


@dataclass
class Recovery:
    """When the planner searches harder, and when the run goes back.

    Searching harder: when even the best sample of the last iteration expects
    the object further than ``retry_tolerance`` [m] from its reference over the
    horizon, or its worst body (a lid included) turned more than
    ``retry_rot_tolerance`` [rad] from it, the tick is planned again from the
    best plan so far with the noise scaled by ``retry_noise_gain`` per retry,
    up to ``max_retries`` times.

    Going back when it is lost: when the object has been more than
    ``stop_tolerance`` [m] off for ``stop_frames`` frames (lost), its worst body
    off or turned more than ``stop_rot_tolerance`` [rad] for as long, or a lid
    opening of the reference ended without being followed, the run returns to
    before that began and plans the stretch again.

    Going back while the object slips: when the executed object has been more
    than ``back_tolerance`` [m] from its reference, or its worst body turned more
    than ``back_rot_tolerance`` [rad] from it, for ``back_frames`` frames in a
    row, it is slipping out of the hands, and the run goes back before it has
    fallen. The slip began after the last frame at which the object was within
    ``calm_tolerance`` [m] (0: at the slip's first frame). These returns have
    tries of their own: the tries spent on a slip do not use up those of a loss.
    ``back_tolerance: 0`` turns this off.

    Where to: ``backtrack_seconds`` before the failure began, and
    ``backtrack_growth`` [s] further on each further try at the same place (a
    failure that begins within a second of the last one), up to
    ``max_backtracks`` tries. The stretch planned again, up to where it failed,
    gets the noise scaled by ``back_noise_gain`` per try, a horizon of
    ``back_horizon`` [s] and ``back_iterations`` iterations per frame (0: as the
    run): looking further ahead, the planner sees the failure coming.

    Stopping: a run that has lost the object with no tries left at that place
    stops and saves the motion up to there.
    """

    retry_tolerance: float = 0.04
    retry_rot_tolerance: float = 0.35
    max_retries: int = 2
    retry_noise_gain: float = 1.5
    back_tolerance: float = 0.05
    back_rot_tolerance: float = 0.5
    back_frames: int = 2
    calm_tolerance: float = 0.02
    back_noise_gain: float = 1.0
    back_horizon: float = 0.8
    back_iterations: int = 0
    max_backtracks: int = 3
    backtrack_seconds: float = 1.0
    backtrack_growth: float = 1.0
    stop_tolerance: float = 0.10
    stop_rot_tolerance: float = 0.5
    stop_frames: int = 40


def resolve_paths(config: Config) -> Config:
    """Scene, reference and output paths of the trial, in SPIDER's dataset layout."""
    trial_dir = get_processed_data_dir(
        dataset_dir=os.path.abspath(config.dataset_dir),
        dataset_name=config.dataset_name,
        robot_type=config.robot_type,
        embodiment_type=config.embodiment_type,
        task=config.task,
        data_id=config.data_id,
    )
    config.model_path = config.model_path or f"{trial_dir}/scene.xml"
    config.data_path = config.data_path or str(
        Path(config.model_path).with_suffix(".npz")
    )
    if not Path(config.model_path).exists():
        raise FileNotFoundError(
            f"no scene at {config.model_path}: get the example dataset, export one "
            "with spider.simulators.chord_export, or pass model_path=DIR/scene.xml"
        )
    config.output_dir = str(Path(config.output_dir or trial_dir).expanduser().resolve())
    os.makedirs(config.output_dir, exist_ok=True)
    return config


def process_config(config: Config, env) -> Config:
    """Fill the fields SPIDER derives from a scene, using the environment instead."""
    config.sim_dt = env.frame_dt
    config.nq, config.nv, config.nu = env.nq, env.nv, env.nu
    config.embodiment_type = "bimanual" if len(env.sides) == 2 else env.sides[0]
    config.object_action_dims = 0
    config = compute_steps(config)
    config = compute_noise_schedule(config)
    if config.max_sim_steps == -1:
        config.max_sim_steps = env.num_frames
    config.env_params_list = [
        [{}] * config.num_dr for _ in range(config.max_num_iterations)
    ]
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    random.seed(config.seed)
    return config


def scene_camera(model: mujoco.MjModel, trace_ref: np.ndarray) -> mujoco.MjvCamera:
    """A free camera on the object's path and the hands working on it.

    The scenes have no lights of their own, so the camera's headlight is turned up
    (a rendering setting of ``model``; the physics does not read it).

    Args:
        model: The scene.
        trace_ref: Reference positions of the object root and the wrists, (T, K, 3).
    """
    if model.nlight == 0:
        model.vis.headlight.ambient[:] = 0.35
        model.vis.headlight.diffuse[:] = 0.65
        model.vis.headlight.specular[:] = 0.15
    path, wrists = trace_ref[:, 0], trace_ref[:, 1:].reshape(-1, 3)
    centre = (path.min(axis=0) + path.max(axis=0)) / 2
    radius = float(np.linalg.norm(path - centre, axis=1).max())
    # Hands still far away, approaching, do not push the camera back.
    reach = np.linalg.norm(wrists - centre, axis=1)
    if (reach < 0.35).any():
        radius = max(radius, float(np.percentile(reach[reach < 0.35], 95)))
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.lookat[:] = centre
    camera.distance = float(np.clip(2.2 * radius + 0.3, 0.7, 1.6))
    camera.azimuth, camera.elevation = 135.0, -25.0
    return camera


def expected_errors(infos: dict[str, np.ndarray]) -> tuple[float, float]:
    """Object errors the best samples of the last iteration expect over the horizon.

    If even the best sample is far from the reference, the search has found no
    way to keep the object on its path (the position error [m]) or to turn it
    and its lid as the reference does (the rotation error of the worst object
    body [rad]).
    """
    last = int(np.asarray(infos["opt_steps"]).ravel()[0]) - 1
    errors = []
    for key in ("object_pos_err_min", "object_rot_err_min"):
        value = float(np.asarray(infos[key]).ravel()[last]) if key in infos else 0.0
        errors.append(value if np.isfinite(value) else float("inf"))
    return errors[0], errors[1]


def excess(recovery: Recovery, errors: tuple[float, float]) -> float:
    """Expected errors relative to the retry tolerances; above 1 no sample is good enough."""
    if recovery.retry_tolerance <= 0.0:
        return 0.0
    value = errors[0] / recovery.retry_tolerance
    if recovery.retry_rot_tolerance > 0.0:
        value = max(value, errors[1] / recovery.retry_rot_tolerance)
    return value


def slip_onset(frames: list[dict[str, np.ndarray]], calm_tolerance: float) -> int:
    """Index of the first frame after the last one with the object within ``calm_tolerance``.

    A diverged frame (NaN) is not calm.
    """
    onset = len(frames)
    while onset > 0 and not (
        float(frames[onset - 1]["object_pos_err"].max()) <= calm_tolerance
    ):
        onset -= 1
    return onset


def plan(
    config: Config, recovery: Recovery, env, optimize, ctrls: torch.Tensor, ref_slice
) -> tuple[torch.Tensor, dict[str, np.ndarray], int, tuple[float, float]]:
    """Optimise the plan, and search harder when no sample keeps the object.

    Returns:
        The best plan, its optimizer statistics, the number of passes and the
        object position [m] and rotation [rad] errors it expects.
    """
    retries = recovery.max_retries if recovery.retry_tolerance > 0.0 else 0
    noise_scale = config.noise_scale
    best = None
    for attempt in range(1 + max(retries, 0)):
        config.noise_scale = noise_scale * recovery.retry_noise_gain**attempt
        start = ctrls if best is None else best[0]
        candidate, infos = optimize(config, env, start, ref_slice)
        errors = expected_errors(infos)
        if best is None or excess(recovery, errors) < excess(recovery, best[3]):
            best = (candidate, infos, attempt + 1, errors)
        if excess(recovery, errors) <= 1.0:
            break
    config.noise_scale = noise_scale
    return best[0], best[1], attempt + 1, best[3]


class Checkpoint(NamedTuple):
    """What the run needs to return to the start of a replanning tick."""

    state: object  # the simulator's saved state of every world
    ctrls: torch.Tensor
    sim_step: int
    frames: int  # lengths of the run's records at that moment
    ticks: int
    images: int
    slip_frames: int
    slip_start: int
    lost_frames: int
    off_frames: int


class Backtracker:
    """Where to go back to when the object slips or its lid was not followed.

    A slipping object is rarely won back by planning on from where it is: the
    hands follow the demonstration and the sampling noise does not reach the
    object any more. A lid that stayed shut while the demonstration opened it
    is not opened later either. The state from just before that is still known,
    though, and planning the failed part again draws other samples. Slips and
    missed lids are counted apart: a lid that was given up must not use up the
    tries of a later slip.
    """

    # A failure that begins within this of the last one is the same failure.
    PLACE_SECONDS = 1.0

    def __init__(self, config: Config, recovery: Recovery, longest: int = 0):
        """Size the checkpoint store; ``longest`` is the longest lid opening [frames]."""
        self.steps = max(int(round(recovery.backtrack_seconds / config.sim_dt)), 1)
        self.growth = int(round(recovery.backtrack_growth / config.sim_dt))
        self.window = int(round(self.PLACE_SECONDS / config.sim_dt))
        self.max_level = recovery.max_backtracks
        self.enabled = (
            self.max_level > 0
            and recovery.backtrack_seconds > 0.0
            and config.max_num_iterations > 0
        )
        # A missed opening is only known when it ends, a loss some frames after
        # it began; ones longer than 10 s are returned to as far back as the
        # store reaches.
        longest = max(longest, recovery.stop_frames, recovery.back_frames)
        longest = min(longest, int(round(10.0 / config.sim_dt)))
        deepest = (self.max_level + 1) * (self.steps + self.max_level * self.growth)
        reach = longest + self.window + deepest
        self.points: deque[Checkpoint] = deque(
            maxlen=reach // max(config.ctrl_steps, 1) + 2
        )
        self.level = 0  # tries of the last call, for the log
        # Per kind of loss: tries at the current place, and the latest frame at
        # which a loss of that kind began.
        self.tries: dict[str, int] = {}
        self.furthest: dict[str, int] = {}
        self.log: list[tuple[int, int]] = []  # (frame given up at, frame returned to)

    def save(self, point: Checkpoint) -> None:
        """Remember the start of a tick."""
        self.points.append(point)

    def back(self, sim_step: int, since: int, kind: str = "slip") -> Checkpoint | None:
        """The checkpoint to return to, or None when this place has had its tries."""
        if not self.enabled or not self.points:
            return None
        onset = sim_step - since  # first frame of the failure
        furthest = self.furthest.get(kind, -(10**9))
        same_place = onset <= furthest + self.window
        self.level = self.tries.get(kind, 0) + 1 if same_place else 1
        self.tries[kind] = self.level
        self.furthest[kind] = max(furthest, onset)
        if self.level > self.max_level:
            return None
        target = onset - self.steps - (self.level - 1) * self.growth
        while len(self.points) > 1 and self.points[-1].sim_step > target:
            self.points.pop()
        point = self.points.pop()
        self.log.append((sim_step, point.sim_step))
        return point


class Openings:
    """The reference's lid openings, and whether the executed motion followed them.

    The same test as the ``art_events`` of ``evaluate_cpu``: an opening is
    followed if the simulated hinge goes at least half as far as the reference
    does at its peak. A run can only know that once the opening has ended.
    """

    def __init__(self, ref: np.ndarray, q_ids: np.ndarray):
        """Find the openings.

        Args:
            ref: Reference hinge angles of the frames the run executes, (T, H) [rad].
            q_ids: Index of each hinge in a recorded ``qpos``, (H,).
        """
        self.q_ids = q_ids
        self.start_angle = ref[0] if len(ref) else np.zeros(ref.shape[1])
        self.events = [
            (hinge, start, stop, peak)
            for hinge in range(ref.shape[1])
            for start, stop, peak in excursions(ref[:, hinge])
        ]

    def missed(self, frames: list[dict[str, np.ndarray]], since: int) -> int | None:
        """First frame of the earliest opening that ended unfollowed after frame ``since``."""
        starts = []
        for hinge, start, stop, peak in self.events:
            if since < stop <= len(frames):
                q_id = self.q_ids[hinge]
                sim = np.array([frame["qpos"][q_id] for frame in frames[start:stop]])
                if not followed(sim, self.start_angle[hinge], peak):
                    starts.append(start)
        return min(starts) if starts else None


class Panel:
    """The Viser viewer's controls: a task, the key settings, Start / Stop, metrics.

    Runs execute in the main thread (:meth:`serve`); the browser only sets flags.
    """

    # Recovery field, label, step and minimum of the number input.
    RECOVERY = (
        ("max_retries", "Plan a frame again, up to [times]", 1, 0),
        ("retry_noise_gain", "Noise gain per extra plan", 0.1, 1.0),
        ("back_tolerance", "Slipping: object off by [m] (0: off)", 0.005, 0.0),
        ("back_frames", "... for [frames] in a row: go back", 1, 1),
        (
            "calm_tolerance",
            "... it began when last within [m] (0: at the slip)",
            0.005,
            0.0,
        ),
        ("backtrack_seconds", "Go back to [s] before it began", 0.05, 0.05),
        ("max_backtracks", "Go back, up to [times] per slip (0: never)", 1, 0),
        ("backtrack_growth", "... each further try [s] further back", 0.05, 0.0),
        ("back_noise_gain", "Noise gain per return", 0.1, 1.0),
        ("back_horizon", "... planned with a horizon of [s] (0: as the run)", 0.2, 0.0),
        ("back_iterations", "... and [iterations] per frame (0: as the run)", 1, 0),
        ("stop_tolerance", "Lost: object off by [m] ...", 0.01, 0.0),
        ("stop_frames", "... for [frames]: go back, then stop", 10, 1),
    )
    # Config field, label, step and minimum of the number input.
    SETTINGS = (
        ("seed", "Seed", 1, 0),
        ("num_samples", "Samples", 16, 2),
        ("max_num_iterations", "Iterations per frame", 1, 0),
        ("temperature", "Temperature", 0.05, 0.01),
        ("pos_noise_scale", "Wrist noise [m]", 0.001, 0.0),
        ("rot_noise_scale", "Wrist rotation noise [rad]", 0.001, 0.0),
        ("joint_noise_scale", "Finger noise [rad]", 0.01, 0.0),
        ("horizon", "Horizon [s]", 0.2, 0.2),
        ("max_sim_steps", "Frames to run (-1: all)", 20, -1),
    )
    # Plot title, (curve label, colour) per curve; one colour per quantity.
    OBJECT, WRISTS, HINGE = "#2a78d6", "#eb6834", "#1baf7a"
    PLOTS = (
        ("Distance from the reference [cm]", (("object", OBJECT), ("wrists", WRISTS))),
        ("Angle from the reference [deg]", (("object", OBJECT), ("hinge", HINGE))),
        ("Planning time per frame [s]", (("planning", "#6b6a63"),)),
    )

    def __init__(
        self,
        config: Config,
        autostart: bool,
        recovery: Recovery | None = None,
        run=None,
    ):
        """Build the controls on the viewer's server; ``run`` is :func:`retarget`'s twin."""
        from spider.viewers import viser_viewer

        try:
            from viser import uplot
        except ImportError:  # an older Viser: the panel works without the plots
            uplot = None

        self.viewer, self.config = viser_viewer, config
        self.recovery = recovery or Recovery()
        self.run = run or retarget
        self.server = viser_viewer.init_viser(app_name="spider")
        self.start, self.stop, self.shown = autostart, False, None
        self.curves: list[list[float]] = []
        self.returns = 0  # times the current run went back
        gui = self.server.gui
        with gui.add_folder("Retargeting"):
            tasks = self.tasks()
            current = f"{config.task}/{config.data_id}"
            self.task = gui.add_dropdown(
                "Task", tasks, initial_value=current if current in tasks else tasks[0]
            )
            self.inputs = {
                name: gui.add_number(label, getattr(config, name), step=step, min=low)
                for name, label, step, low in self.SETTINGS
            }
            self.start_button = gui.add_button("Start", color="green")
            self.stop_button = gui.add_button("Stop", color="red", disabled=True)
            self.progress = gui.add_progress_bar(0.0)
            self.status = gui.add_markdown("Pick a task and press **Start**.")
        # What the run does when the object slips: plan the frame again, then go
        # back to an earlier frame and plan the stretch again (see Recovery).
        with gui.add_folder("Recovery: plan again, go back"):
            self.recovery_inputs = {
                name: gui.add_number(
                    label, getattr(self.recovery, name), step=step, min=low
                )
                for name, label, step, low in self.RECOVERY
            }
        with gui.add_folder("Metrics"):
            self.plots = [
                gui.add_uplot(
                    data=tuple(np.zeros(1) for _ in range(1 + len(curves))),
                    series=(
                        uplot.Series(label="time [s]"),
                        *[
                            uplot.Series(label=label, stroke=colour, width=2)
                            for label, colour in curves
                        ],
                    ),
                    scales={"x": uplot.Scale(time=False)},
                    title=title,
                    aspect=2.2,
                )
                for title, curves in (self.PLOTS if uplot is not None else ())
            ]
        self.controls = [
            self.task,
            *self.inputs.values(),
            *self.recovery_inputs.values(),
            self.start_button,
        ]
        self.start_button.on_click(lambda _: setattr(self, "start", True))
        self.stop_button.on_click(lambda _: setattr(self, "stop", True))

    def tasks(self) -> list[str]:
        """The scenes of the dataset as ``task/data_id``; one entry for a given scene."""
        config = self.config
        if config.simulator != "cpu" or config.model_path:
            return [f"{config.task}/{config.data_id}"]
        root = Path(
            get_processed_data_dir(
                os.path.abspath(config.dataset_dir),
                config.dataset_name,
                config.robot_type,
                config.embodiment_type,
                "",
                "",
            )
        )
        found = sorted(
            f"{p.parts[-3]}/{p.parts[-2]}" for p in root.glob("*/*/scene.xml")
        )
        return found or [f"{config.task}/{config.data_id}"]

    def run_config(self) -> Config:
        """The configuration of the run the controls describe."""
        config = copy.deepcopy(self.config)
        for name, handle in self.inputs.items():
            setattr(config, name, type(getattr(config, name))(handle.value))
        # The horizon is a whole number of knots.
        config.horizon = max(1, round(config.horizon / config.knot_dt)) * config.knot_dt
        if len(self.task.options) > 1:
            config.task, data_id = self.task.value.split("/")
            config.data_id = int(data_id)
            config.model_path = config.data_path = ""
        return config

    def run_recovery(self) -> Recovery:
        """The recovery settings the controls describe."""
        values = {
            name: type(getattr(self.recovery, name))(handle.value)
            for name, handle in self.recovery_inputs.items()
        }
        return replace(self.recovery, **values)

    def preview(self) -> None:
        """Show the selected task's scene at the first frame of its reference."""
        if self.config.simulator != "cpu" or self.shown == self.task.value:
            return
        self.shown = self.task.value
        config = resolve_paths(self.run_config())
        self.viewer.reset_scene()
        spec = mujoco.MjSpec.from_file(config.model_path)
        model = spec.compile()
        handles = self.viewer.build_and_log_scene_from_spec(
            spec, model, build_ref=False
        )
        data = mujoco.MjData(model)
        with np.load(config.data_path) as scene:
            qpos_ref, frame_dt = scene["qpos_ref"], float(scene["frame_dt"])
        data.qpos[:] = qpos_ref[0]
        mujoco.mj_kinematics(model, data)
        self.viewer.update_frame(data, handles)
        self.status.content = (
            f"**{self.shown}**: {len(qpos_ref)} frames "
            f"({len(qpos_ref) * frame_dt:.1f} s). Press **Start**."
        )

    def begin(self) -> None:
        """A run starts: clear the scene and the plots, lock the settings."""
        self.viewer.reset_scene()
        self.shown, self.stop, self.returns = self.task.value, False, 0
        self.curves = [[] for _ in range(6)]
        for control in self.controls:
            control.disabled = True
        self.stop_button.disabled = False
        self.progress.value = 0.0
        self.status.content = f"Running **{self.shown}** ..."

    def went_back(self, kept: int) -> None:
        """The run returned to an earlier frame: the timeline keeps ``kept`` frames."""
        self.returns += 1
        self.viewer.rewind(kept)

    def update(
        self,
        frames: list,
        ticks: list,
        config: Config,
        hinges: np.ndarray,
        qpos_ref: np.ndarray,
    ) -> None:
        """New frames were executed (or a stretch was given up): refresh the plots."""
        time_s, obj, wrist, rot, hinge, plan_time = self.curves
        for curve in self.curves:  # the run may have gone back
            del curve[len(frames) :]
        for i in range(len(time_s), len(frames)):
            frame = frames[i]
            gap = frame["qpos"][hinges] - qpos_ref[int(frame["frame"])][hinges]
            tick = ticks[min(i // config.ctrl_steps, len(ticks) - 1)]
            time_s.append((i + 1) * config.sim_dt)
            obj.append(float(frame["object_pos_err"].max()) * 100)
            wrist.append(float(frame["wrist_pos_err"].max()) * 100)
            rot.append(float(np.degrees(frame["object_rot_err"].max())))
            hinge.append(float(np.degrees(np.abs(gap).max())) if len(hinges) else 0.0)
            plan_time.append(float(tick["plan_time"]))
        self.progress.value = 100.0 * len(frames) / max(config.max_sim_steps, 1)
        if len(frames) % 5 == 0:
            self.draw()
            again = sum(int(tick["attempts"]) > 1 for tick in ticks)
            self.status.content = (
                f"Running **{self.shown}**: frame {len(frames)} of "
                f"{config.max_sim_steps}; planned {again} frames again, went back "
                f"{self.returns} times."
            )

    def draw(self) -> None:
        """Send the curves to the plots."""
        time_s, obj, wrist, rot, hinge, plan_time = (np.asarray(c) for c in self.curves)
        if self.plots and len(time_s):
            self.plots[0].data = (time_s, obj, wrist)
            self.plots[1].data = (time_s, rot, hinge)
            self.plots[2].data = (time_s, plan_time)

    def end(self, metrics: dict, stopped: bool) -> None:
        """A run is over: say how it went."""
        self.draw()
        for control in self.controls:
            control.disabled = False
        self.stop_button.disabled = True
        verdict = "passes" if metrics["success"] >= 1 else "fails"
        self.status.content = (
            f"**{self.shown}** {'stopped' if stopped else verdict} after "
            f"{metrics['frames']} frames, went back {self.returns} times: "
            f"{summary(metrics)}"
        )

    def serve(self, num_threads: int) -> None:
        """Run what the browser asks for, until Ctrl+C (or one run without wait_on_finish)."""
        loguru.logger.info(
            "Pick a task in the viewer and press Start (Ctrl+C to exit)."
        )
        try:
            while True:
                if self.start:
                    self.start = False
                    self.run(
                        self.run_config(), self.run_recovery(), num_threads, panel=self
                    )
                    if not self.config.wait_on_finish:
                        return
                self.preview()
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass


def replanning_config(config: Config, recovery: Recovery) -> Config:
    """The settings of a stretch planned again after going back (Recovery.back_*)."""
    again = copy.deepcopy(config)
    if recovery.back_horizon > 0.0:
        knots = max(1, round(recovery.back_horizon / config.knot_dt))
        again.horizon = knots * config.knot_dt
    if recovery.back_iterations > 0:
        again.max_num_iterations = recovery.back_iterations
        again.env_params_list = [[{}] * again.num_dr] * again.max_num_iterations
    again = compute_noise_schedule(compute_steps(again))
    return again


def fit(
    ctrls: torch.Tensor, length: int, ctrl_ref: torch.Tensor, sim_step: int
) -> torch.Tensor:
    """A plan of ``length`` frames from ``sim_step``: cut, or continued with the reference's."""
    if len(ctrls) < length:
        start = sim_step + len(ctrls)
        ctrls = torch.cat([ctrls, ctrl_ref[start : start + length - len(ctrls)]])
    return ctrls[:length]


def stack_ticks(values: list[np.ndarray]) -> np.ndarray:
    """Stack one statistic over ticks; ticks planned with fewer iterations are NaN-padded."""
    width = max((v.shape[0] for v in values if v.ndim), default=0)
    return np.stack(
        [
            np.concatenate([v, np.full((width - len(v), *v.shape[1:]), np.nan)])
            if v.ndim and len(v) < width
            else v
            for v in values
        ]
    )


def extend(ref_data: tuple[torch.Tensor, ...], length: int) -> tuple[torch.Tensor, ...]:
    """The reference held at its last frame up to ``length`` frames, as MJWP's is."""
    return tuple(
        torch.cat([r, r[-1:].expand(length - len(r), *r.shape[1:])])
        if len(r) < length
        else r
        for r in ref_data
    )


def retarget(
    config: Config,
    recovery: Recovery | None = None,
    num_threads: int = 0,
    panel: Panel | None = None,
) -> dict[str, np.ndarray]:
    """Run sampling MPC over the reference and save the executed trajectory."""
    config = resolve_paths(config)
    ref_data = cpu.load_data(config, config.data_path)
    env = cpu.setup_env(config, ref_data, num_threads=num_threads)
    return track(config, recovery, cpu, env, ref_data, panel)


def track(
    config: Config,
    recovery: Recovery | None,
    sim,
    env,
    ref_data: tuple[torch.Tensor, ...],
    panel: Panel | None = None,
) -> dict[str, np.ndarray]:
    """Plan and execute over the whole reference with simulator module ``sim``.

    Args:
        config: The run's settings; its paths are resolved.
        recovery: When to plan a frame again and when to go back.
        sim: A simulator with the interface of :mod:`spider.simulators.cpu`
            (MJWP's, plus ``run_arrays`` and ``read_step`` for the records).
        env: ``sim``'s worlds, at reference frame 0.
        ref_data: The reference, ``(qpos, qvel, ctrl, contact, contact_pos)``.
        panel: The viewer's controls, when it drives the run.
    """
    recovery = recovery or Recovery()
    output_dir = Path(config.output_dir)
    config = process_config(config, env)
    loguru.logger.info(
        "MPC: {} samples x {} iterations, horizon {} / replan {} / knot {} frames, "
        "{} frames to execute",
        config.num_samples,
        config.max_num_iterations,
        config.horizon_steps,
        config.ctrl_steps,
        config.knot_steps,
        config.max_sim_steps,
    )

    # A stretch planned again after going back may look further ahead.
    again = replanning_config(config, recovery)
    # The optimizer slices the reference one horizon ahead of every frame.
    longest_horizon = max(config.horizon_steps, again.horizon_steps)
    ref_data = extend(
        ref_data, config.max_sim_steps + longest_horizon + config.ctrl_steps + 1
    )
    ctrl_ref = ref_data[2]
    rollout = make_rollout_fn(
        sim.step_env,
        sim.save_state,
        sim.load_state,
        sim.get_reward,
        sim.get_terminal_reward,
        sim.get_terminate,
        sim.get_trace,
        sim.save_env_params,
        sim.load_env_params,
        sim.copy_sample_state,
    )
    optimize = make_optimize_fn(make_optimize_once_fn(rollout))

    # A single MuJoCo world for the viewer and the video.
    mj_model, qpos_ref = env.mj_model, env.mj_qpos_ref
    mj_data, mj_data_ref = mujoco.MjData(mj_model), mujoco.MjData(mj_model)
    mj_data.qpos[:] = mj_data_ref.qpos[:] = qpos_ref[0]
    mujoco.mj_kinematics(mj_model, mj_data)
    mujoco.mj_kinematics(mj_model, mj_data_ref)
    if panel is not None:
        panel.begin()
    run_viewer = setup_viewer(config, mj_model, mj_data)
    streaming = config.show_viewer and (
        "viser" in config.viewer or "rerun" in config.viewer
    )
    # The camera frames the object and the wrists (the fingertips are traced too).
    renderer, camera = (
        None,
        scene_camera(mj_model, env.trace_ref[:, : 1 + len(env.sides)]),
    )
    try:
        renderer = setup_renderer(config, mj_model)
    except Exception as error:  # no OpenGL context on this machine
        loguru.logger.warning("no video ({}); set MUJOCO_GL=egl or osmesa", error)

    arrays = sim.run_arrays(env)
    ctrls = ctrl_ref[: config.horizon_steps].clone()
    frames: list[dict[str, np.ndarray]] = []
    ticks: list[dict[str, np.ndarray]] = []
    images: list[np.ndarray] = []
    sim_step = 0
    slip_frames = 0  # consecutive executed frames with the object slipping
    slip_start = 0  # ... and the first of them
    lost_frames = 0  # ... further than stop_tolerance from its path
    off_frames = 0  # ... that far off, or turned / opened unlike the reference
    stopped_early = user_stopped = False
    # A stretch planned again after going back gets more noise, until the frame
    # at which it was given up.
    base_noise, boost_until, boost = config.noise_scale.clone(), -1, 1.0
    base_noise_again = again.noise_scale.clone()
    # Step i of the run targets reference frame i (the last one once it runs out).
    hinges = np.asarray(arrays["articulation_q_ids"], dtype=np.int64)
    steps = np.minimum(np.arange(config.max_sim_steps), len(qpos_ref) - 1)
    openings = Openings(qpos_ref[steps][:, hinges], hinges)
    longest = max((stop - start for _, start, stop, _ in openings.events), default=0)
    backtracker = Backtracker(config, recovery, longest)

    t_start = time.perf_counter()
    with run_viewer() as viewer:
        while sim_step < config.max_sim_steps and viewer.is_running():
            if backtracker.enabled:
                backtracker.save(
                    Checkpoint(
                        sim.save_state(env),
                        ctrls.clone(),
                        sim_step,
                        len(frames),
                        len(ticks),
                        len(images),
                        slip_frames,
                        slip_start,
                        lost_frames,
                        off_frames,
                    )
                )
            # A stretch planned again after going back, up to where it failed,
            # gets more noise and its own horizon and iterations.
            replanning = sim_step < boost_until
            planner = again if replanning else config
            planner.noise_scale = (base_noise_again if replanning else base_noise) * (
                boost if replanning else 1.0
            )
            ctrls = fit(ctrls, planner.horizon_steps, ctrl_ref, sim_step)
            t0 = time.perf_counter()
            ref_slice = get_slice(
                ref_data, sim_step + 1, sim_step + planner.horizon_steps + 1
            )
            attempts, expected = 0, (0.0, 0.0)
            if planner.max_num_iterations > 0:
                ctrls, infos, attempts, expected = plan(
                    planner, recovery, env, optimize, ctrls, ref_slice
                )
            else:
                infos = {"opt_steps": np.array([0])}
            t_plan = time.perf_counter() - t0
            horizon = np.minimum(
                sim_step + np.arange(planner.horizon_steps), len(qpos_ref) - 1
            )
            infos["trace_ref"] = env.trace_ref[horizon][None, None]  # (1, 1, H, K, 3)

            # Execute in every world, record one, then put all worlds on it.
            num_exec = min(config.ctrl_steps, config.max_sim_steps - sim_step)
            for i in range(num_exec):
                sim.step_env(config, env, ctrls[i])
                record = sim.read_step(config, env)
                record["ctrl"] = ctrls[i].detach().cpu().numpy()
                frames.append(record)
                sim_step += 1
                # A diverged world (NaN) counts as slipping and lost too.
                pos = float(record["object_pos_err"].max())
                rot = float(record["object_rot_err"].max())
                slipping = recovery.back_tolerance > 0.0 and (
                    not pos <= recovery.back_tolerance
                    or (
                        recovery.back_rot_tolerance > 0.0
                        and not rot <= recovery.back_rot_tolerance
                    )
                )
                if slipping and slip_frames == 0:
                    slip_start = sim_step - 1  # the frame just executed
                slip_frames = slip_frames + 1 if slipping else 0
                near = pos <= recovery.stop_tolerance
                turned = recovery.stop_rot_tolerance > 0.0 and not (
                    rot <= recovery.stop_rot_tolerance
                )
                lost_frames = 0 if near else lost_frames + 1
                off_frames = 0 if near and not turned else off_frames + 1
                mj_data.qpos[:] = np.nan_to_num(record["qpos"])
                mj_data_ref.qpos[:] = qpos_ref[int(record["frame"])]
                mj_data.time = sim_step * config.sim_dt
                if renderer is not None:
                    images.append(
                        render_image(
                            config, renderer, mj_model, mj_data, mj_data_ref, camera
                        )
                    )
                if streaming:
                    mujoco.mj_kinematics(mj_model, mj_data)
                    mujoco.mj_kinematics(mj_model, mj_data_ref)
                    log_frame(
                        mj_data,
                        sim_time=mj_data.time,
                        viewer_body_entity_and_ids=config.viewer_body_entity_and_ids,
                        data_ref=mj_data_ref,
                    )
            sim.sync_env(config, env, mj_data)
            update_viewer(config, viewer, mj_model, mj_data, mj_data_ref, infos)

            # Shift the plan; its new tail is the reference's (a zero residual).
            tail = sim_step + len(ctrls) - config.ctrl_steps
            ctrls = torch.cat(
                [ctrls[config.ctrl_steps :], ctrl_ref[tail : tail + config.ctrl_steps]]
            )
            tick = {k: np.asarray(v) for k, v in infos.items() if "trace" not in k}
            tick["plan_time"] = np.asarray(t_plan)
            tick["sim_step"] = np.asarray(sim_step)
            tick["attempts"] = np.asarray(attempts)
            ticks.append(tick)
            if panel is not None:
                panel.update(frames, ticks, config, hinges, qpos_ref)
                if panel.stop:  # the run ends here, and is scored up to here
                    user_stopped, config.max_sim_steps = True, sim_step
                    break
            last = frames[-1]
            print(
                f"frame {sim_step}/{config.max_sim_steps}  plan {t_plan:.2f}s "
                f"x{attempts}  object {last['object_pos_err'].max() * 100:.1f} cm "
                f"{np.degrees(last['object_rot_err'].max()):.0f} deg  "
                f"expects {expected[0] * 100:.1f} cm {np.degrees(expected[1]):.0f} deg",
                flush=True,
            )
            # A lost object decides where the run goes back to. Otherwise an object
            # that is on its path but turned does, or a lid opening of the reference
            # that ended in this tick without having been followed. Otherwise an
            # object that is slipping does: to before the slip began.
            lost = 0 < recovery.stop_frames <= lost_frames
            since = lost_frames if lost else off_frames
            if not lost and backtracker.enabled:
                missed = openings.missed(frames, sim_step - num_exec)
                if missed is not None:
                    since = max(since, recovery.stop_frames, sim_step - missed)
            kind = None
            if 0 < recovery.stop_frames <= since:
                kind = "lost" if lost else "turned"
            elif 0 < recovery.back_frames <= slip_frames:
                kind = "slip"
                if recovery.calm_tolerance > 0.0:
                    since = len(frames) - slip_onset(frames, recovery.calm_tolerance)
                else:
                    since = sim_step - slip_start
            if kind is not None:
                point = backtracker.back(sim_step, since, kind)
                if point is not None:
                    loguru.logger.warning(
                        "object {} since frame {}: going back to frame {} to plan the "
                        "stretch again (try {} of {} here)",
                        {"slip": "slipping", "lost": "lost"}.get(
                            kind, "or lid not following the reference"
                        ),
                        sim_step - since,
                        point.sim_step,
                        backtracker.level,
                        backtracker.max_level,
                    )
                    boost_until = sim_step
                    boost = recovery.back_noise_gain**backtracker.level
                    sim.load_state(env, point.state)
                    ctrls = point.ctrls
                    del frames[point.frames :]
                    del ticks[point.ticks :]
                    del images[point.images :]
                    if panel is not None:
                        panel.went_back(len(frames))
                        panel.update(frames, ticks, config, hinges, qpos_ref)
                    sim_step = point.sim_step
                    slip_frames, slip_start = point.slip_frames, point.slip_start
                    lost_frames, off_frames = point.lost_frames, point.off_frames
                    continue
            if lost:
                stopped_early = True
                loguru.logger.warning(
                    "stopping at frame {} of {}: the object has been more than {:.0f} "
                    "cm from its reference for {} frames and the planner finds no way "
                    "back",
                    sim_step,
                    config.max_sim_steps,
                    recovery.stop_tolerance * 100,
                    lost_frames,
                )
                break
    wall_time = time.perf_counter() - t_start
    config.noise_scale = base_noise

    trajectory = {k: np.stack([f[k] for f in frames]) for k in frames[0]}
    for k in ticks[0]:
        trajectory[f"opt_{k}"] = stack_ticks([np.asarray(t[k]) for t in ticks])
    trajectory.update(arrays)
    trajectory["wall_time"] = np.asarray(wall_time)
    trajectory["frames_planned"] = np.asarray(config.max_sim_steps)
    trajectory["stopped_early"] = np.asarray(int(stopped_early))
    # Frames at which the run gave a stretch up, and the frames it returned to.
    trajectory["backtrack_from"] = np.asarray(
        [given_up for given_up, _ in backtracker.log], dtype=np.int64
    )
    trajectory["backtrack_to"] = np.asarray(
        [returned for _, returned in backtracker.log], dtype=np.int64
    )
    metrics = evaluate(trajectory)
    if config.save_info:
        path = output_dir / f"trajectory_{config.simulator}.npz"
        np.savez(path, **trajectory)
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
        loguru.logger.info("Saved {}", path)
    if config.save_video and images:
        video_path = output_dir / f"visualization_{config.simulator}.mp4"
        imageio.mimsave(video_path, images, fps=int(round(1.0 / config.sim_dt)))
        loguru.logger.info("Saved video to {}", video_path)
    loguru.logger.info("Done in {:.0f} s: {}", wall_time, summary(metrics))
    if panel is not None:
        panel.end(metrics, user_stopped)
    return trajectory


def main(
    config: Config,
    recovery: Recovery | None = None,
    num_threads: int = 0,
    autostart: bool = False,
    run=None,
) -> None:
    """Retarget the configured task, or serve the Viser viewer's control panel.

    ``run`` replaces :func:`retarget` (``run_chord.py`` passes its GPU twin).
    """
    recovery, run = recovery or Recovery(), run or retarget
    if config.show_viewer and "viser" in config.viewer:
        Panel(config, autostart, recovery, run).serve(num_threads)
    else:
        run(config, recovery, num_threads)


def parse(cfg: DictConfig) -> tuple[Config, Recovery, int, bool]:
    """The run's Config, and the keys of the YAML that are not part of it."""
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    recovery = Recovery(**config_dict.pop("recovery", {}))
    num_threads = int(config_dict.pop("num_threads", 0))
    autostart = bool(config_dict.pop("autostart", False))
    config_dict = filter_config_fields(config_dict)
    if config_dict.get("noise_scale") is None:
        config_dict.pop("noise_scale", None)
    return Config(**config_dict), recovery, num_threads, autostart


@hydra.main(version_base=None, config_path="config", config_name="cpu")
def run_main(cfg: DictConfig) -> None:
    """Hydra entry point."""
    main(*parse(cfg))


if __name__ == "__main__":
    run_main()
