# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests of SPIDER's CPU simulator and its runner on a small synthetic scene.

Self-contained: the scene is written here in the form the CPU workflow reads
(a floating one-finger hand, a box with a passive lid held by friction and a
support fixed to the world). ``chord_export_test.py`` checks that the export of
FlashCHORD scenes writes this form. The environment tests need ``mujoco.batch``
(MuJoCo >= 3.17)::

    pytest spider/simulators/cpu_test.py
"""

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import mujoco
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

import spider
from spider.config import Config, compute_noise_schedule, filter_config_fields
from spider.simulators import cpu

NUM_WORLDS = 6
NUM_FRAMES = 12
EXAMPLES = Path(spider.ROOT).parent / "examples"
# The lid weighs 0.05 kg, its centre of mass 3 cm out and 3 mm up from the hinge;
# its friction holds 1.5 times that torque.
LID_HOLD = 1.5 * 0.05 * 9.81 * float(np.hypot(0.03, 0.003))
SCENE_XML = f"""
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.01"/>
  <worldbody>
    <body name="hand" pos="0 0 0.3" gravcomp="1">
      <joint name="wrist_x" type="slide" axis="1 0 0" armature="0.01"/>
      <joint name="wrist_y" type="slide" axis="0 1 0" armature="0.01"/>
      <joint name="wrist_z" type="slide" axis="0 0 1" armature="0.01"/>
      <joint name="wrist_rot" type="ball" damping="1" armature="0.01"/>
      <geom type="box" size="0.03 0.03 0.01" mass="0.5"/>
      <body name="finger" pos="0.03 0 0" gravcomp="1">
        <joint name="finger" type="hinge" axis="0 1 0" armature="0.01"/>
        <geom type="capsule" fromto="0 0 0 0.04 0 0" size="0.008" mass="0.05"/>
      </body>
    </body>
    <body name="box" pos="0 0 0.07">
      <joint name="box_free" type="free"/>
      <geom type="box" size="0.03 0.03 0.03" mass="0.2"/>
      <body name="lid" pos="-0.03 0 0.03">
        <joint name="lid_hinge" type="hinge" axis="0 1 0" frictionloss="{LID_HOLD!r}"
               range="-2 0" armature="0.001"/>
        <geom type="box" pos="0.03 0 0.003" size="0.03 0.03 0.003" mass="0.05"/>
      </body>
    </body>
    <body name="support" pos="0 0 0.02">
      <geom type="cylinder" size="0.1 0.02" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <general joint="wrist_x" biastype="affine" gainprm="200" biasprm="0 -200"/>
    <general joint="wrist_x" biastype="affine" gainprm="15" biasprm="0 0 -15"/>
    <general joint="wrist_y" biastype="affine" gainprm="200" biasprm="0 -200"/>
    <general joint="wrist_y" biastype="affine" gainprm="15" biasprm="0 0 -15"/>
    <general joint="wrist_z" biastype="affine" gainprm="200" biasprm="0 -200"/>
    <general joint="wrist_z" biastype="affine" gainprm="15" biasprm="0 0 -15"/>
    <general joint="finger" biastype="affine" gainprm="2" biasprm="0 -2"/>
    <general joint="finger" biastype="affine" gainprm="0.1" biasprm="0 0 -0.1"/>
    <general joint="lid_hinge" biastype="affine" gainprm="0" biasprm="0 0 0"/>
    <general joint="lid_hinge" biastype="affine" gainprm="0" biasprm="0 0 0"/>
  </actuator>
</mujoco>
"""


def scene_tables() -> dict[str, np.ndarray]:
    """The tables of ``scene.npz`` for :data:`SCENE_XML`: the hand rises 2 cm."""
    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    qpos_ref = np.tile(data.qpos, (NUM_FRAMES, 1))
    qpos_ref[:, 2] = np.linspace(0.0, 0.02, NUM_FRAMES)
    targets = np.zeros((NUM_FRAMES, model.nv))
    targets[:, 2] = qpos_ref[:, 2]
    bodies = [model.body("box").id, model.body("lid").id]
    pose = np.concatenate([data.xpos[bodies], data.xquat[bodies][:, [1, 2, 3, 0]]], 1)
    return {
        "frame_dt": 0.05,
        "substeps": 5,
        "sides": np.array(["right"]),
        "body_names": np.array(["box", "lid"]),
        "targets0": targets,
        # Position actuators and the dof whose target each tracks; the others
        # are velocity actuators, which hold zero.
        "position_actuator": np.array([0, 2, 4, 6, 8]),
        "position_actuator_dof": np.array([0, 1, 2, 6, 13]),
        "wrist_pos_dof": np.array([[0, 1, 2]]),
        "wrist_ori_dof": np.array([[3, 4, 5]]),
        "finger_dof": np.array([[6]]),
        "wrist_qpos": np.array([3]),
        "wrist_kp": np.array([25.0]),
        "wrist_tau_max": np.array([45.0]),
        "action_clip": np.array([0.2] * 3 + [1.0] * 4),
        "action_ema": 0.3,
        "qpos_ref": qpos_ref,
        "object_pose_ref": np.tile(pose, (NUM_FRAMES, 1, 1)),
        "object_pose_reset": pose,
        "object_body": np.array(bodies),
        "wrist_body": np.array([model.body("hand").id]),
        "scalar_q": np.array([0, 1, 2, 7, 8, 9, 10, 15]),
        "scalar_group": np.array([0, 0, 0, 1, 2, 2, 2, 3]),
        "quat_q": np.array([3, 11]),
        "quat_group": np.array([0, 1]),
        "vel_q": np.array([0, 1, 2, 7, 15]),
        "vel_dof": np.array([0, 1, 2, 6, 13]),
        "articulation_q": np.array([15]),
    }


def _write_scene(directory) -> None:
    (directory / "scene.xml").write_text(SCENE_XML)
    np.savez(directory / "scene.npz", **scene_tables())


def _config(directory, **overrides) -> Config:
    """The defaults of examples/config/cpu.yaml on the synthetic scene."""
    values = OmegaConf.to_container(OmegaConf.load(EXAMPLES / "config" / "cpu.yaml"))
    values = filter_config_fields(values)
    values.update(
        model_path=str(directory / "scene.xml"),
        data_path=str(directory / "scene.npz"),
        output_dir=str(directory),
        num_samples=NUM_WORLDS,
        show_viewer=False,
        save_video=False,
    )
    values.update(overrides)
    values.pop("noise_scale", None)
    return Config(**values)


def _env(directory, threads: int = 2):
    pytest.importorskip("mujoco.batch")
    directory.mkdir(exist_ok=True)
    _write_scene(directory)
    config = _config(directory)
    ref_data = cpu.load_data(config, config.data_path)
    return config, cpu.setup_env(config, ref_data, num_threads=threads)


@pytest.fixture
def env(tmp_path):
    """A few CPU worlds of the synthetic scene at frame 0."""
    return _env(tmp_path)


def _plans(env, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    scale = torch.tensor([0.02] * 3 + [0.1] * 4)
    return torch.randn(env.num_worlds, 8, env.nu, generator=generator) * scale


def _rollout(config, env, plans: torch.Tensor) -> torch.Tensor:
    rewards = []
    for t in range(plans.shape[1]):
        cpu.step_env(config, env, plans[:, t])
        rewards.append(cpu.get_reward(config, env)[0])
    return torch.stack(rewards)


def test_quaternion_helpers_match_scipy():
    """Products, rotation vectors and roll-pitch-yaw agree with SciPy (xyzw)."""
    a = Rotation.random(16, random_state=0)
    b = Rotation.random(16, random_state=1)
    product = cpu._quat_mul(a.as_quat(), b.as_quat())
    np.testing.assert_allclose(
        cpu._quat_to_rotvec(product), (a * b).as_rotvec(), atol=1e-9
    )
    rotvec = a.as_rotvec()
    back = cpu._quat_to_rotvec(cpu._rotvec_to_quat(rotvec))
    np.testing.assert_allclose(back, rotvec, atol=1e-9)
    rpy = np.random.default_rng(2).uniform(-1.0, 1.0, (16, 3))
    expected = Rotation.from_euler("xyz", rpy).as_rotvec()
    np.testing.assert_allclose(
        cpu._quat_to_rotvec(cpu._quat_rpy(rpy)), expected, atol=1e-9
    )


def test_support_is_static_and_holds_the_box(env):
    """The support has no coordinates left and the box rests on it, lid shut."""
    config, env = env
    assert (env.nq, env.nv, env.nu) == (16, 14, 7)
    start = env.xpos[:, env.scene["object_body"]].copy()
    for _ in range(10):
        env.timestep[:] = 0
        cpu.step_env(config, env, torch.zeros(env.nu))
    drift = np.linalg.norm(env.xpos[:, env.scene["object_body"]] - start, axis=-1)
    assert drift.max() < 2e-3
    assert not env.failed.any()


def test_zero_control_follows_the_reference(env):
    """Zero residual tracks the reference targets and advances the clock."""
    config, env = env
    for frame in range(NUM_FRAMES):
        cpu.step_env(config, env, torch.zeros(env.nu))
        assert (env.frame == frame).all()
    assert (env.timestep == NUM_FRAMES - 1).all()
    assert np.abs(env.qpos[:, 2] - env.mj_qpos_ref[-1, 2]).max() < 5e-3
    reward, info = cpu.get_reward(config, env)
    assert (reward <= 0).all() and info["object_pos_err"].max() < 5e-3
    record = cpu.read_step(config, env)
    assert record["object_pose"].shape == (2, 7)
    assert record["wrist_pos_err"].shape == (1,) and record["wrist_pos_err"][0] < 5e-3


def test_control_is_in_physical_units(env):
    """A 1 cm control commands a 1 cm residual before the EMA."""
    config, env = env
    ctrl = torch.zeros(env.nu)
    ctrl[0] = 0.01
    cpu.step_env(config, env, ctrl)
    assert env.filtered[:, 0] == pytest.approx(0.7 * 0.01)
    assert np.abs(env.filtered[:, 1:]).max() == 0.0
    position = env.scene["position_actuator"][0]
    assert env.ctrl[:, position] == pytest.approx(0.7 * 0.01)


def test_wrist_torque_drives_toward_the_target(env):
    """An orientation residual turns the ball wrist toward the commanded rotation."""
    config, env = env
    ctrl = torch.zeros(env.nu)
    ctrl[5] = 0.3  # yaw
    for _ in range(NUM_FRAMES):
        cpu.step_env(config, env, ctrl)
    quat = cpu._wxyz_to_xyzw(env.qpos[:, 3:7])
    yaw = cpu._quat_to_rotvec(quat)[:, 2]
    assert (yaw > 0.1).all() and (yaw < 0.35).all()


def test_worlds_reruns_and_restores_are_bitwise_identical(env):
    """Identical inputs give identical states, across worlds and after a restore."""
    config, env = env
    plans = _plans(env)
    same = plans[:1].expand(env.num_worlds, -1, -1)
    saved = cpu.save_state(env)
    first = _rollout(config, env, same)
    assert torch.equal(first, first[:, :1].expand_as(first))
    state = env.batch.state.copy()
    cpu.load_state(env, saved)
    assert np.array_equal(env.batch.state, saved["state"])
    second = _rollout(config, env, same)
    assert torch.equal(first, second)
    assert np.array_equal(env.batch.state, state)


def test_different_plans_give_different_rewards(env):
    """Per-world plans are applied per world."""
    config, env = env
    rewards = _rollout(config, env, _plans(env)).mean(dim=0)
    assert rewards.std().item() > 0.0


def test_sync_and_copy(env):
    """Sync broadcasts one world; copy overwrites the selected worlds."""
    config, env = env
    _rollout(config, env, _plans(env))
    cpu.copy_sample_state(config, env, torch.tensor([1]), torch.tensor([4]))
    assert np.array_equal(env.batch.state[4], env.batch.state[1])
    assert np.array_equal(env.filtered[4], env.filtered[1])
    cpu.sync_env(config, env, None)  # as MJWP: the first world to every world
    assert (env.batch.state == env.batch.state[0]).all()
    assert (env.filtered == env.filtered[0]).all()


def test_thread_count_does_not_change_results(tmp_path):
    """One thread and several threads give bitwise identical rollouts."""
    config, one = _env(tmp_path / "one", threads=1)
    _, many = _env(tmp_path / "many", threads=3)
    plans = _plans(one)
    assert torch.equal(_rollout(config, one, plans), _rollout(config, many, plans))
    assert np.array_equal(one.batch.state, many.batch.state)


def test_traces_are_the_object_root_the_wrists_and_the_fingertips(env):
    """The viewer's traces: the object first, one point per hand, one per finger."""
    config, env = env
    cpu.step_env(config, env, torch.zeros(env.nu))
    trace = cpu.get_trace(config, env)
    assert trace.shape == (NUM_WORLDS, 3, 3)
    assert env.trace_ref.shape == (NUM_FRAMES, 3, 3)
    assert np.allclose(trace[0, 0].numpy(), env.xpos[0, env.scene["object_body"][0]])
    assert np.allclose(env.trace_ref[0, 0], env.scene["object_pose_ref"][0, 0, :3])
    # The finger is a capsule from its joint to 4 cm along x, 8 mm thick: its tip
    # is the far end, carried by the finger's body.
    finger = env.mj_model.body("finger").id
    assert env.tip_body.tolist() == [finger]
    assert np.allclose(env.tip_local, [[0.048, 0.0, 0.0]])
    tip = np.zeros(3)
    mujoco.mju_rotVecQuat(tip, env.tip_local[0], env.xquat[0, finger])
    assert np.allclose(trace[0, 2].numpy(), env.xpos[0, finger] + tip, atol=1e-6)
    hand = env.mj_model.body("hand").id
    assert np.allclose(trace[0, 1].numpy(), env.xpos[0, hand], atol=1e-6)


def _runner():
    """``examples/run_cpu.py`` as a module (it is a script, not a package)."""
    if "run_cpu" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "run_cpu", EXAMPLES / "run_cpu.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["run_cpu"] = module  # dataclasses look their module up
        spec.loader.exec_module(module)
    return sys.modules["run_cpu"]


@pytest.mark.parametrize("iterations", [0, 2])
def test_the_runner_plans_scores_and_saves(tmp_path, iterations):
    """A replay and a short planned run of examples/run_cpu.py on the scene."""
    pytest.importorskip("mujoco.batch")
    _write_scene(tmp_path)
    config = _config(
        tmp_path, simulator="cpu", num_samples=16, max_num_iterations=iterations
    )
    trajectory = _runner().retarget(config, num_threads=2)
    assert trajectory["qpos"].shape == (NUM_FRAMES, 16)
    assert trajectory["opt_plan_time"].shape == (NUM_FRAMES,)
    saved = np.load(tmp_path / "trajectory_cpu.npz")
    assert np.array_equal(saved["qpos"], trajectory["qpos"])
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["frames"] == NUM_FRAMES and metrics["stopped_early"] == 0
    assert metrics["obj_pos_cm"] < 0.5 and metrics["success"] == 1.0


@pytest.mark.parametrize("longer", [False, True])
def test_a_slip_sends_the_run_back_before_it_began(tmp_path, longer):
    """With a tolerance nothing meets, the run goes back as soon as it can tell.

    ``longer``: the stretch planned again looks further ahead, with more iterations.
    """
    pytest.importorskip("mujoco.batch")
    _write_scene(tmp_path)
    config = _config(tmp_path, num_samples=16, max_num_iterations=1)
    recovery = _runner().Recovery(
        back_tolerance=1e-9,
        back_rot_tolerance=0.0,
        back_frames=4,
        calm_tolerance=0.0,  # the slip begins at its first frame
        max_backtracks=2,
        backtrack_seconds=0.1,
        back_noise_gain=2.0,
        back_horizon=0.8 if longer else 0.0,
        back_iterations=2 if longer else 0,
    )
    trajectory = _runner().retarget(config, recovery, num_threads=2)
    # Ticks of the stretch planned again ran two iterations, the others one.
    rewards = trajectory["opt_rew_max"]
    assert rewards.shape == (NUM_FRAMES, 2 if longer else 1)
    assert np.isnan(rewards[:, -1]).any() == longer
    # Slipping from the first frame on, noticed at frame 4: back to before
    # frame 0 (the oldest checkpoint), twice; then the run goes on to the end.
    assert trajectory["backtrack_from"].tolist() == [4, 4]
    assert trajectory["backtrack_to"].tolist() == [0, 0]
    assert len(trajectory["qpos"]) == NUM_FRAMES and not trajectory["stopped_early"]
    # The boost does not outlive the run: the noise is the schedule of the settings.
    expected = compute_noise_schedule(copy.deepcopy(config)).noise_scale
    assert torch.equal(config.noise_scale, expected)


def test_a_slip_began_after_the_object_was_last_calm():
    """The run returns to before the excursion that led to the slip."""
    runner = _runner()
    errors = [0.01, 0.015, 0.03, 0.025, 0.04, 0.06, 0.07]  # metres, frame by frame
    frames = [{"object_pos_err": np.array([e, 0.0])} for e in errors]
    assert runner.slip_onset(frames, 0.02) == 2  # last calm frame: 1 (1.5 cm)
    assert runner.slip_onset(frames, 0.05) == 5
    assert runner.slip_onset(frames, 0.1) == len(frames)  # calm now: no excursion
    frames[1]["object_pos_err"][1] = np.nan  # a diverged frame is not calm
    assert runner.slip_onset(frames, 0.02) == 1


def test_the_runner_finds_the_scene_in_the_dataset_layout(tmp_path):
    """Without model_path the scene is looked up as SPIDER lays datasets out."""
    trial = tmp_path / "processed" / "chord" / "sharpa" / "bimanual" / "box" / "1"
    trial.mkdir(parents=True)
    _write_scene(trial)
    config = _config(
        trial,
        dataset_dir=str(tmp_path),
        model_path="",
        data_path="",
        task="box",
        data_id=1,
    )
    config.output_dir = ""
    config = _runner().resolve_paths(config)
    assert config.model_path == str(trial / "scene.xml")
    assert config.data_path == str(trial / "scene.npz")
    assert Path(config.output_dir) == trial.resolve()


def _panel(tmp_path, monkeypatch):
    """The viewer's panel over two scenes of a dataset, on a stand-in server."""
    from spider.viewers import viser_viewer

    root = tmp_path / "processed" / "chord" / "sharpa" / "bimanual" / "box"
    for take in ("1", "2"):
        (root / take).mkdir(parents=True)
        _write_scene(root / take)
    server = MagicMock()
    server.gui.add_button.side_effect = lambda label, **_: MagicMock()
    server.gui.add_number.side_effect = lambda label, initial_value, **_: (
        SimpleNamespace(value=initial_value, disabled=False)
    )
    server.gui.add_dropdown.side_effect = lambda label, options, initial_value, **_: (
        SimpleNamespace(value=initial_value, options=tuple(options), disabled=False)
    )
    state = viser_viewer._ViserState(server=server)
    monkeypatch.setattr(viser_viewer, "_STATE", state)
    monkeypatch.setattr(viser_viewer, "threading", MagicMock())  # no playback thread
    config = _config(
        root / "1",
        dataset_dir=str(tmp_path),
        model_path="",
        data_path="",
        output_dir="",
        show_viewer=True,
        viewer="viser",
        num_samples=16,
        max_num_iterations=1,
    )
    return _runner().Panel(config, autostart=False)


def test_the_panel_runs_the_task_and_the_settings_it_shows(tmp_path, monkeypatch):
    """The panel lists the dataset's scenes; a run uses its inputs and fills it."""
    pytest.importorskip("mujoco.batch")
    panel = _panel(tmp_path, monkeypatch)
    assert panel.task.options == ("box/1", "box/2") and panel.task.value == "box/1"
    panel.preview()
    assert "12 frames" in panel.status.content
    panel.task.value = "box/2"
    panel.inputs["seed"].value = 3.0
    panel.inputs["max_sim_steps"].value = 6.0
    config = panel.run_config()
    assert (config.task, config.data_id) == ("box", 2)
    assert (config.seed, config.max_sim_steps) == (3, 6)
    # The recovery settings: shown with the runner's values, and used as entered.
    assert panel.recovery_inputs["max_backtracks"].value == 3
    panel.recovery_inputs["max_backtracks"].value = 0.0
    panel.recovery_inputs["backtrack_seconds"].value = 2.0
    recovery = panel.run_recovery()
    assert (recovery.max_backtracks, recovery.backtrack_seconds) == (0, 2.0)
    assert recovery.stop_frames == _runner().Recovery().stop_frames
    assert _runner().Backtracker(config, recovery).enabled is False
    trajectory = _runner().retarget(config, num_threads=2, panel=panel)
    assert len(trajectory["qpos"]) == 6
    assert [len(curve) for curve in panel.curves] == [6] * 6
    assert panel.status.content.startswith("**box/2** passes after 6 frames")
    assert not panel.start_button.disabled and panel.stop_button.disabled
    assert (tmp_path / "processed/chord/sharpa/bimanual/box/2/metrics.json").exists()


def test_stop_ends_the_run_and_scores_what_was_done(tmp_path, monkeypatch):
    """Pressing Stop keeps the frames executed so far."""
    pytest.importorskip("mujoco.batch")
    panel = _panel(tmp_path, monkeypatch)
    update = panel.update

    def update_then_stop(frames, *args):
        update(frames, *args)
        panel.stop = len(frames) >= 3

    panel.update = update_then_stop
    trajectory = _runner().retarget(panel.run_config(), num_threads=2, panel=panel)
    assert len(trajectory["qpos"]) == 3 and trajectory["frames_planned"] == 3
    assert "stopped after 3 frames" in panel.status.content


def test_going_back_rewinds_the_viewer(tmp_path, monkeypatch):
    """A return to an earlier frame is counted and rewinds the viewer's timeline."""
    panel = _panel(tmp_path, monkeypatch)
    rewound = []
    monkeypatch.setattr(panel.viewer, "rewind", rewound.append)
    panel.begin()
    panel.went_back(4)
    panel.went_back(2)
    assert rewound == [4, 2] and panel.returns == 2


# --
# The runner's recovery logic (no simulator needed)
# --


def _fake_optimize(errors: list[float], noise_seen: list[float]):
    """An optimizer whose successive passes expect the given object errors."""
    remaining = iter(errors)

    def optimize(config, env, ctrls, ref_slice):
        noise_seen.append(float(config.noise_scale))
        infos = {
            "opt_steps": np.array([2]),
            "object_pos_err_min": np.array([0.0, next(remaining)]),
            "object_rot_err_min": np.array([0.0, 0.5]),
        }
        return ctrls + 1.0, infos

    return optimize


def test_plan_searches_harder_until_a_sample_keeps_the_object():
    """Planning repeats with more noise while no sample keeps the object, and keeps the best."""
    runner = _runner()
    recovery = runner.Recovery(retry_rot_tolerance=0.0)
    config = Config()
    config.noise_scale = torch.ones(1)
    start = torch.zeros(2)

    def plan(optimize):
        return runner.plan(config, recovery, None, optimize, start, None)

    noise: list[float] = []
    ctrls, _, attempts, expected = plan(_fake_optimize([0.12, 0.08, 0.02], noise))
    assert (attempts, expected) == (3, (pytest.approx(0.02), 0.5))
    assert noise == [1.0, 1.5, 2.25]
    assert float(ctrls[0]) == 3.0  # every pass continued from the best plan so far
    assert float(config.noise_scale) == 1.0

    # Retries that do not help keep the first plan.
    ctrls, _, attempts, expected = plan(_fake_optimize([0.12, 0.20, 0.15], []))
    assert (attempts, expected[0], float(ctrls[0])) == (3, pytest.approx(0.12), 1.0)

    # A plan that keeps the object is not planned again.
    assert plan(_fake_optimize([0.01], []))[2] == 1
    # With a rotation tolerance, a lid no sample follows (0.5 rad off) also means
    # planning again.
    recovery.retry_rot_tolerance = 0.35
    assert plan(_fake_optimize([0.01, 0.01, 0.01], []))[2] == 3
    # A tolerance of zero never retries.
    recovery.retry_rot_tolerance = recovery.retry_tolerance = 0.0
    assert plan(_fake_optimize([0.50], []))[2] == 1


def test_expected_errors_read_the_last_iteration_that_ran():
    """The expectation is the last real iteration's best sample; a diverged one counts as lost."""
    runner = _runner()
    infos = {
        "opt_steps": np.array([2]),
        "object_pos_err_min": np.array([0.5, 0.03, 0.0]),
        "object_rot_err_min": np.array([1.0, 0.2, 0.0]),
    }
    assert runner.expected_errors(infos) == (pytest.approx(0.03), pytest.approx(0.2))
    infos["object_pos_err_min"] = np.array([0.5, np.nan, 0.0])
    assert runner.expected_errors(infos)[0] == float("inf")
    assert runner.expected_errors({"opt_steps": np.array([0])}) == (0.0, 0.0)
    # Relative to the tolerances: the position alone unless a rotation tolerance is set.
    recovery = runner.Recovery(retry_tolerance=0.04, retry_rot_tolerance=0.0)
    assert runner.excess(recovery, (0.02, 0.8)) == pytest.approx(0.5)
    recovery.retry_rot_tolerance = 0.4
    assert runner.excess(recovery, (0.02, 0.8)) == pytest.approx(2.0)
    recovery.retry_tolerance = 0.0
    assert runner.excess(recovery, (0.5, 0.8)) == 0.0


def _ticks(runner, backtracker, first: int, last: int, ctrl_steps: int = 4) -> None:
    """Checkpoints of the replanning ticks that start at frames ``first`` .. ``last``."""
    for step in range(first, last + 1, ctrl_steps):
        backtracker.save(runner.Checkpoint(None, None, step, step, 0, 0, 0, 0, 0, 0))


def test_backtracker_returns_to_just_before_the_failure():
    """Every try at a failure returns the same short way before it; then the place is given up."""
    runner = _runner()
    recovery = runner.Recovery(
        stop_frames=40, max_backtracks=2, backtrack_seconds=1.0, backtrack_growth=0.0
    )
    config = Config(max_num_iterations=4)
    config.sim_dt, config.ctrl_steps = 0.05, 4
    backtracker = runner.Backtracker(config, recovery)
    # The object leaves at frame 100; the run notices 40 frames later.
    _ticks(runner, backtracker, 0, 136)
    assert backtracker.back(140, 40).sim_step == 80  # 1 s (20 frames) before
    # Planned again from frame 80 and failing at about the same place: the same
    # distance before that failure, not further.
    _ticks(runner, backtracker, 80, 140)
    assert backtracker.back(144, 40).sim_step == 84
    # A third failure here is one too many.
    _ticks(runner, backtracker, 84, 136)
    assert backtracker.back(140, 40) is None
    assert backtracker.log == [(140, 80), (144, 84)]

    # The default: 1 s before the failure, then 2 s, then 3 s.
    default = runner.Backtracker(config, runner.Recovery(stop_frames=40))
    _ticks(runner, default, 0, 136)
    assert default.back(140, 40).sim_step == 80
    _ticks(runner, default, 80, 140)
    assert default.back(144, 40).sim_step == 64

    # Growing returns: each further try at the place goes a little further back.
    growing = runner.Backtracker(
        config, runner.Recovery(backtrack_seconds=0.5, backtrack_growth=0.5)
    )
    _ticks(runner, growing, 0, 136)
    assert growing.back(140, 40).sim_step == 88  # 0.5 s before frame 100
    _ticks(runner, growing, 88, 140)
    assert growing.back(144, 40).sim_step == 84  # 1 s before frame 104

    # One step before the failure: the shortest return.
    short = runner.Backtracker(config, runner.Recovery(backtrack_seconds=0.05))
    _ticks(runner, short, 0, 136)
    assert (
        short.steps == 1 and short.back(140, 40).sim_step == 96
    )  # the tick before 100
    # A loss more than a second further on is a new place and counts from one again.
    _ticks(runner, backtracker, 140, 236)
    assert backtracker.back(240, 40).sim_step == 180

    # Checkpoints only reach so far back: the oldest one kept is used then.
    backtracker = runner.Backtracker(config, recovery)
    _ticks(runner, backtracker, 0, 396)
    assert backtracker.points[0].sim_step > 0
    oldest = backtracker.points[0].sim_step
    assert backtracker.back(400, 399).sim_step == oldest

    # A lid that was given up does not use up the tries of a slip later on, and a
    # lid still shut after the return is not tried again.
    backtracker = runner.Backtracker(config, recovery)
    _ticks(runner, backtracker, 0, 136)
    assert backtracker.back(140, 40, "lid").sim_step == 80
    _ticks(runner, backtracker, 80, 136)
    assert backtracker.back(140, 40, "lid").sim_step == 80
    _ticks(runner, backtracker, 80, 136)
    assert backtracker.back(140, 40, "lid") is None
    _ticks(runner, backtracker, 140, 196)
    assert backtracker.back(200, 40, "slip").sim_step == 140
    _ticks(runner, backtracker, 140, 156)
    assert backtracker.back(160, 60, "lid") is None

    # Off (no tries, or a replay that plans nothing): a lost object ends the run.
    for off in (
        runner.Backtracker(config, runner.Recovery(max_backtracks=0)),
        runner.Backtracker(Config(max_num_iterations=0, sim_dt=0.05), recovery),
    ):
        assert not off.enabled and off.back(140, 40) is None


def test_openings_find_the_lid_opening_that_was_not_followed():
    """An opening counts as missed once it has ended without the lid going half way."""
    runner = _runner()
    ref = np.zeros((100, 2))
    ref[10:30, 1] = 1.0  # the second hinge opens twice by 1 rad
    ref[50:80, 1] = 1.0
    openings = runner.Openings(ref, np.array([3, 5]))
    assert [(h, a, b) for h, a, b, _ in openings.events] == [(1, 10, 30), (1, 50, 80)]

    def frames(
        count: int, opened: tuple[int, int] | None
    ) -> list[dict[str, np.ndarray]]:
        qpos = np.zeros((count, 8))
        if opened:
            qpos[opened[0] : opened[1], 5] = 0.6
        return [{"qpos": row} for row in qpos]

    # The first opening ends at frame 30: nothing to say before that.
    assert openings.missed(frames(28, None), 24) is None
    assert openings.missed(frames(32, None), 28) == 10
    # Followed more than half way.
    assert openings.missed(frames(32, (12, 20)), 28) is None
    # An opening that ended in an earlier tick is not judged again.
    assert openings.missed(frames(36, None), 32) is None
    assert openings.missed(frames(80, (12, 20)), 76) == 50
    # No hinges, no openings.
    assert runner.Openings(np.zeros((100, 0)), np.zeros(0, dtype=np.int64)).events == []


def test_the_default_timing_is_whole_control_frames():
    """The default replanning period, knot spacing and horizon are whole control frames."""
    settings = OmegaConf.to_container(OmegaConf.load(EXAMPLES / "config" / "cpu.yaml"))
    for name in ("ctrl_dt", "knot_dt", "horizon"):
        steps = settings[name] / settings["sim_dt"]
        assert abs(steps - round(steps)) < 1e-9 and round(steps) >= 1, name
    assert settings["horizon"] >= settings["knot_dt"] >= settings["ctrl_dt"]
    # The recovery section names exactly the runner's settings, all of them: Hydra
    # only takes overrides of keys the file has.
    recovery = _runner().Recovery(**settings["recovery"])
    assert sorted(settings["recovery"]) == sorted(vars(recovery))
    assert vars(recovery) == vars(_runner().Recovery())  # and the same defaults


def test_a_single_elite_gets_all_the_weight():
    """Fewer than twenty samples leave one elite: its weight is 1, not NaN."""
    from spider.optimizers.sampling import _compute_weights_impl

    weights, _ = _compute_weights_impl(torch.tensor([0.1, 0.7, 0.3]), 3, 0.3)
    assert torch.equal(weights, torch.tensor([0.0, 1.0, 0.0]))


def test_few_samples_keep_their_noise():
    """With too few samples for an exploit share, the samples still explore."""
    from spider.config import get_noise_scale

    config = Config(
        num_samples=64,
        horizon=0.6,
        knot_dt=0.2,
        device="cpu",
        embodiment_type="bimanual",
    )
    config.nu, config.object_action_dims = 56, 0
    config.exploit_ratio, config.exploit_noise_scale = 0.01, 0.01
    config.pos_noise_scale = config.rot_noise_scale = 0.01
    config.joint_noise_scale = 0.1
    few = get_noise_scale(config)
    assert few.shape == (64, 3, 56)
    assert float(few[0].abs().max()) == 0.0  # the first sample is the plan itself
    config.num_samples = 200  # two exploit samples at the end
    many = get_noise_scale(config)
    torch.testing.assert_close(few[1:], many[1:64])
    torch.testing.assert_close(many[-2:], many[1:3] * 0.01)
    assert float(many[-3].max()) == float(many[1].max())
