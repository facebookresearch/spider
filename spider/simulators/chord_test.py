# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests of the FlashCHORD GPU backend.

Needs FlashCHORD, a CUDA device and a reference::

    CHORD_TEST_PARQUET=REFERENCE_DIR pytest spider/simulators/chord_test.py

The runner's logic and the CPU backend are tested without FlashCHORD in
``cpu_test.py`` and ``chord_export_test.py``.
"""

import importlib.util
import os

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from spider.config import Config

chord = pytest.importorskip("spider.simulators.chord")  # needs Warp

NUM_WORLDS = 32
HORIZON = 16


def test_quat_error_matches_scipy():
    """The xyzw rotation error equals SciPy's relative rotation vector."""
    rng = np.random.default_rng(0)
    sim = Rotation.random(64, random_state=rng)
    ref = Rotation.random(64, random_state=rng)
    expected = (sim * ref.inv()).as_rotvec()
    error = chord._quat_error_xyzw(
        torch.as_tensor(sim.as_quat()), torch.as_tensor(ref.as_quat())
    )
    np.testing.assert_allclose(error.numpy(), expected, atol=1e-6)
    # A quaternion and its negation are the same rotation.
    flipped = chord._quat_error_xyzw(
        -torch.as_tensor(sim.as_quat()), torch.as_tensor(ref.as_quat())
    )
    np.testing.assert_allclose(flipped.numpy(), expected, atol=1e-6)


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    """A small batch of FlashCHORD worlds with the module's recipe."""
    parquet = os.environ.get("CHORD_TEST_PARQUET", "")
    if not parquet:
        pytest.skip("set CHORD_TEST_PARQUET to a FlashCHORD reference directory")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    if importlib.util.find_spec("flash_chord") is None:
        pytest.skip("needs FlashCHORD")
    config = Config(
        simulator="chord",
        data_path=parquet,
        num_samples=NUM_WORLDS,
        device="cuda:0",
        output_dir=str(tmp_path_factory.mktemp("chord")),
        show_viewer=False,
        base_pos_rew_scale=1.0,
        base_rot_rew_scale=0.3,
        joint_rew_scale=0.01,
    )
    return config, chord.setup_env(config)


@pytest.fixture
def env(setup):
    """The shared worlds, reset to reference frame 0."""
    config, env = setup
    return config, chord.reset_env(config, env, 0)


def _plans(env, seed: int = 0) -> torch.Tensor:
    """A different smooth residual plan per world, (N, HORIZON, nu)."""
    generator = torch.Generator().manual_seed(seed)
    half = env.nu // max(len(env.sides), 1)
    scale = torch.full((half,), 0.15)
    scale[:6] = 0.03
    knots = torch.randn(env.num_worlds, HORIZON // 4, env.nu, generator=generator)
    knots = knots * scale.repeat(len(env.sides))
    return knots.repeat_interleave(4, dim=1).to(env.device)


def _rollout(config, env, plans: torch.Tensor) -> torch.Tensor:
    """Mean reward per world of a plan batch."""
    rewards = []
    for t in range(plans.shape[1]):
        chord.step_env(config, env, plans[:, t])
        rewards.append(chord.get_reward(config, env)[0])
    return torch.stack(rewards).mean(dim=0)


def test_layout_and_recipe(env):
    """State and action dimensions match the scene, and the recipe's gains are in."""
    config, env = env
    assert env.state["joint_q"].shape == (NUM_WORLDS, env.nq)
    assert env.state["joint_qd"].shape == (NUM_WORLDS, env.nv)
    hands = env.fc.scene.layout.hands
    assert env.nu == sum(6 + len(hand.finger_dof_ids) for hand in hands)
    assert not chord.get_terminate(config, env).any()
    assert float(env.cfg.task.motion_speed) == 1.0  # FlashCHORD trains at another speed
    assert float(env.fc.action.buffers.ema) == pytest.approx(chord.ACTION_EMA)
    assert float(env.fc.solver.mjw_data.njmax) == chord.NJMAX
    gains = env.mj_model.actuator_gainprm[:, 0]
    assert chord.WRIST_KP in gains and chord.FINGER_KP in gains


def test_reset_state_is_the_reference(env):
    """The reset state is the reference frame it was reset to, and the clock runs."""
    config, env = env
    qpos_dist, _ = chord.spider_cost(config, env)
    assert qpos_dist.max().item() < 1e-4
    zero = torch.zeros(env.nu, device=env.device)
    for frame in range(3):
        chord.step_env(config, env, zero)
        assert (env.frame == frame).all()
    assert (env.state["timestep"] == 3).all()


def test_control_is_in_physical_units(env):
    """A 1 cm control commands a 1 cm wrist residual before the EMA."""
    config, env = env
    ctrl = torch.zeros(env.nu, device=env.device)
    ctrl[0] = 0.01  # 1 cm on the first wrist axis of the first action side
    chord.step_env(config, env, ctrl)
    filtered = env.state["filtered_action"][0]
    assert filtered[0].item() == pytest.approx(
        (1.0 - chord.ACTION_EMA) * 0.01, rel=1e-4
    )
    assert filtered[1:].abs().max().item() == 0.0


def test_reward_is_tracking_cost_plus_contact_terms(env):
    """The reward is SPIDER's cost plus the positively weighted CHORD contact terms."""
    config, env = env
    chord.step_env(config, env, _plans(env)[:, 0])
    reward, info = chord.get_reward(config, env)
    qpos_dist, qvel_dist = chord.spider_cost(config, env)
    contact = (env.terms * env.weights).sum(dim=1)
    expected = -qpos_dist - config.vel_rew_scale * qvel_dist
    assert torch.allclose(reward, expected + chord.CONTACT_SCALE * contact, atol=1e-5)
    assert (env.weights >= 0).all() and (contact >= 0).all()
    assert torch.allclose(info["qpos_dist"], qpos_dist)


def test_save_load_state_is_exact(env):
    """Loading a saved state restores every state array exactly."""
    config, env = env
    saved = chord.save_state(env)
    _rollout(config, env, _plans(env))
    assert not torch.equal(env.state["joint_q"], saved["joint_q"])
    chord.load_state(env, saved)
    for name, view in env.state.items():
        assert torch.equal(view, saved[name]), name


def test_rollout_repeats_after_load(env):
    """A plan replayed from a restored state ends in nearly the same state."""
    config, env = env
    plans = _plans(env)
    saved = chord.save_state(env)
    _rollout(config, env, plans)
    first = env.state["joint_q"].clone()
    chord.load_state(env, saved)
    _rollout(config, env, plans)
    second = env.state["joint_q"].clone()
    # GPU physics is not bitwise repeatable; a replay must still end far closer
    # to the first run than the states that different plans lead to.
    replay = (first - second).abs().max(dim=1).values.median().item()
    spread = (first - first.roll(1, dims=0)).abs().max(dim=1).values.median().item()
    assert replay < 0.05 * spread, (replay, spread)


def test_sync_and_copy(env):
    """Sync broadcasts one world; copy overwrites the selected worlds."""
    config, env = env
    _rollout(config, env, _plans(env))
    src = torch.tensor([1, 2], device=env.device)
    dst = torch.tensor([7, 9], device=env.device)
    chord.copy_sample_state(config, env, src, dst)
    for view in env.state.values():
        assert torch.equal(view[dst], view[src])
    source = {name: view[0].clone() for name, view in env.state.items()}
    chord.sync_env(config, env, None)  # as MJWP: the first world to every world
    for name, view in env.state.items():
        assert torch.equal(view, source[name].expand_as(view)), name


def test_kinematic_armature_fix(env):
    """The fix replaces the 1e10 armature and rescales the solver statistic."""
    import warp as wp

    _, env = env
    solver = env.fc.solver
    if not env.kinematic_dofs:
        pytest.skip("the scene has no kinematic body")
    armature = wp.to_torch(solver.mjw_model.dof_armature)[..., list(env.kinematic_dofs)]
    assert torch.all(armature == chord.KINEMATIC_ARMATURE)
    # Without the fix the statistic is dominated by the 1e10 armature.
    assert wp.to_torch(solver.mjw_model.stat.meaninertia).max().item() < 100.0


def test_records_are_in_mujoco_coordinates(env):
    """What a run saves addresses the compiled MuJoCo model, as the CPU backend's does."""
    config, env = env
    arrays = chord.run_arrays(env)
    nq = env.mj_model.nq
    assert arrays["qpos_ref"].shape == (env.num_frames, nq)
    # Traces: the object root, the wrists, and five fingertips per hand.
    hands = len(env.sides)
    assert len(env.tip_body_ids) == 5 * hands
    assert env.trace_ref.shape == (env.num_frames, 1 + 6 * hands, 3)
    assert chord.get_trace(config, env).shape == (NUM_WORLDS, 1 + 6 * hands, 3)
    # The fingertips are where plain MuJoCo puts them on the compiled model.
    import mujoco

    from spider.simulators.cpu import _rotate_wxyz, fingertips

    to_newton = env.fc.solver.mjc_body_to_newton.numpy()[0]
    to_mujoco = {int(body): i for i, body in enumerate(to_newton) if body >= 0}
    links, local = fingertips(
        env.mj_model, [to_mujoco[int(body)] for body in env.wrist_body_ids]
    )
    data = mujoco.MjData(env.mj_model)
    data.qpos[:] = env.mj_qpos_ref[0]
    mujoco.mj_kinematics(env.mj_model, data)
    expected = data.xpos[links] + _rotate_wxyz(data.xquat[links], local)
    assert np.allclose(env.trace_ref[0, 1 + hands :], expected, atol=1e-4)
    fingers = arrays["finger_q_ids"]
    assert len(set(fingers)) == len(fingers) == env.nu - 6 * len(env.sides)
    bodies = len(arrays["body_names"])
    assert arrays["object_pose_ref"].shape == (env.num_frames, bodies, 7)
    assert arrays["object_pose_reset"].shape == (bodies, 7)
    # The wrist quaternions of the reference are unit quaternions at their addresses.
    for q in arrays["wrist_quat_q_ids"]:
        norm = np.linalg.norm(arrays["qpos_ref"][:, q : q + 4], axis=1)
        np.testing.assert_allclose(norm, 1.0, atol=1e-4)
    chord.step_env(config, env, torch.zeros(env.nu, device=env.device))
    record = chord.read_step(config, env)
    assert record["qpos"].shape == (nq,) and int(record["frame"]) == 0
    assert record["object_pose"].shape == (bodies, 7)
    assert record["object_pos_err"].shape == (bodies,)
    assert record["wrist_pos_err"].shape == (len(env.sides),)
    # One frame of zero control stays on the reference.
    assert np.abs(record["qpos"] - arrays["qpos_ref"][0]).max() < 0.05


def test_export_gives_a_scene_plain_mujoco_compiles(setup, tmp_path):
    """The exported scene has the support fixed and tables that fit its model."""
    import mujoco

    from spider.simulators.chord_export import export_scene

    config, env = setup
    out = export_scene(config.data_path, tmp_path, device=config.device)
    model = mujoco.MjModel.from_xml_path(str(out / "scene.xml"))
    scene = np.load(out / "scene.npz")
    fixed = len(env.kinematic_dofs)
    assert model.nv == env.mj_model.nv - fixed
    assert scene["qpos_ref"].shape == (env.num_frames, model.nq)
    assert scene["targets0"].shape == (env.num_frames, model.nv)
    assert model.opt.timestep == pytest.approx(
        float(scene["frame_dt"]) / int(scene["substeps"])
    )
    assert float(scene["action_ema"]) == pytest.approx(chord.ACTION_EMA)
    assert len(scene["action_clip"]) == env.nu
    # The reference's first frame puts the object bodies where FlashCHORD has them.
    data = mujoco.MjData(model)
    data.qpos[:] = scene["qpos_ref"][0]
    mujoco.mj_kinematics(model, data)
    np.testing.assert_allclose(
        data.xpos[scene["object_body"]], scene["object_pose_reset"][:, :3], atol=1e-4
    )
