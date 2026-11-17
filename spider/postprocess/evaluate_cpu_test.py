# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests of the CPU workflow's metrics on synthetic trajectories (no simulator needed)."""

import json

import numpy as np
from scipy.spatial.transform import Rotation

from spider.postprocess import evaluate_cpu

FRAMES = 40


def _trajectory(
    object_shift=0.0, hinge_shift=0.0, finger_shift=0.0, reference_offset=0.0
):
    """A one-body articulated object and two fingers that follow the reference.

    qpos layout: finger 0, finger 1, hinge. ``reference_offset`` puts the
    reference's body frame that far along the body's x axis from the link frame.
    """
    time = np.linspace(0.0, 1.0, FRAMES)
    yaw = Rotation.from_rotvec(np.outer(0.5 * time, [0.0, 0.0, 1.0]))
    link_pos = np.stack([0.1 * time, np.zeros(FRAMES), 0.2 * time], axis=1)
    ref_pos = link_pos + yaw.apply([reference_offset, 0.0, 0.0])
    ref_pose = np.concatenate([ref_pos, yaw.as_quat()], axis=1)[:, None]  # (T, 1, 7)
    sim_pose = np.concatenate([link_pos, yaw.as_quat()], axis=1)[:, None]
    reset_pose = sim_pose[0].copy()  # the simulated body starts on the reference
    sim_pose[..., 2] += object_shift
    qpos_ref = np.stack([0.3 * time, -0.2 * time, time], axis=1)
    qpos = qpos_ref + np.array([finger_shift, finger_shift, hinge_shift])
    return {
        "frame": np.arange(FRAMES),
        "qpos": qpos.astype(np.float32),
        "qpos_ref": qpos_ref.astype(np.float32),
        "object_pose": sim_pose.astype(np.float32),
        "object_pose_ref": ref_pose,
        "object_pose_reset": reset_pose.astype(np.float32),
        "wrist_pos_err": np.full((FRAMES, 2), 0.01, dtype=np.float32),
        "wrist_rot_err": np.full((FRAMES, 2), np.radians(2.0), dtype=np.float32),
        "reward": np.zeros(FRAMES, dtype=np.float32),
        "wall_time": np.asarray(1.0),
        "opt_plan_time": np.ones(FRAMES // 4),
        "finger_q_ids": np.array([0, 1]),
        "wrist_quat_q_ids": np.zeros(0, dtype=np.int64),
        "articulation_q_ids": np.array([2]),
        "body_names": np.array(["bottom"]),
        "frame_dt": np.asarray(0.05),
        "frames_planned": np.asarray(FRAMES),
        "stopped_early": np.asarray(0),
        "backtrack_from": np.zeros(0, dtype=np.int64),
    }


def _evaluate(tmp_path, **changes):
    np.savez(tmp_path / "trajectory_cpu.npz", **_trajectory(**changes))
    return evaluate_cpu.evaluate(tmp_path)


def test_a_run_on_the_reference_has_no_error(tmp_path):
    """Following the reference exactly scores zero object, hinge and finger error."""
    metrics = _evaluate(tmp_path)
    assert metrics["obj_pos_cm"] < 1e-4
    assert metrics["obj_rot_deg"] < 1e-3
    assert metrics["art_err_deg"] < 1e-4
    assert metrics["finger_deg"] < 1e-4
    assert metrics["success"] == 1.0
    assert metrics["frac_ok"] == 1.0
    assert abs(metrics["wrist_pos_cm"] - 1.0) < 1e-5


def test_errors_are_reported_in_their_units(tmp_path):
    """A 5 cm object shift, 0.1 rad hinge error and 0.05 rad finger error are measured as such."""
    metrics = _evaluate(tmp_path, object_shift=0.05, hinge_shift=0.1, finger_shift=0.05)
    assert abs(metrics["obj_pos_cm"] - 5.0) < 1e-4
    assert abs(metrics["art_err_deg"] - np.degrees(0.1)) < 1e-3
    assert abs(metrics["finger_deg"] - np.degrees(0.05)) < 1e-3
    assert metrics["success"] == 0.0
    assert metrics["frac_ok"] == 0.0


def test_reference_frame_offset_is_not_an_error(tmp_path):
    """A reference frame placed elsewhere on the body than the link frame costs nothing."""
    metrics = _evaluate(tmp_path, reference_offset=0.08)
    assert metrics["obj_pos_cm"] < 1e-4
    assert metrics["success"] == 1.0


def test_main_writes_metrics(tmp_path, capsys):
    """The command-line entry writes metrics.json and prints one summary line."""
    np.savez(tmp_path / "trajectory_cpu.npz", **_trajectory())
    evaluate_cpu.main((str(tmp_path),))
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["frames"] == FRAMES
    assert "object 0.00 cm" in capsys.readouterr().out


def test_an_object_far_below_its_reference_has_left_the_support(tmp_path):
    """A slip of a few centimetres is not a fall; a drop far below the reference is."""
    assert _evaluate(tmp_path, object_shift=-0.05)["fallen_frames"] == 0
    assert _evaluate(tmp_path, object_shift=-0.5)["fallen_frames"] == FRAMES


def test_a_lid_that_never_opens_is_a_failure(tmp_path):
    """Object tracking alone does not pass a run whose hinge misses its opening."""
    opened = _evaluate(tmp_path)
    assert (opened["art_events"], opened["art_events_hit"]) == (1, 1)
    trajectory = _trajectory()
    trajectory["qpos"][:, 2] = 0.0  # the hinge stays where it started
    np.savez(tmp_path / "trajectory_cpu.npz", **trajectory)
    shut = evaluate_cpu.evaluate(tmp_path)
    assert (shut["art_events"], shut["art_events_hit"]) == (1, 0)
    assert shut["success_object"] == 1.0
    assert shut["success"] == 0.0
    assert shut["frac_ok"] < 1.0


def test_articulation_events_count_each_opening():
    """Two openings of different size are two events; a hinge that follows one hits one."""
    time = np.linspace(0.0, 4.0 * np.pi, 400)
    ref = np.abs(np.sin(time / 2.0)) * np.where(time < 2.0 * np.pi, 1.0, 0.6)
    assert evaluate_cpu.articulation_events(ref, ref) == (2, 2)
    first_only = np.where(time < 2.0 * np.pi, ref, 0.0)
    assert evaluate_cpu.articulation_events(first_only, ref) == (2, 1)
    assert evaluate_cpu.articulation_events(np.zeros_like(ref), ref) == (2, 0)
    # A hinge that barely moves in the whole clip has no events to follow.
    assert evaluate_cpu.articulation_events(0.1 * ref, 0.1 * ref) == (0, 0)


def test_a_run_stopped_early_is_a_failure(tmp_path):
    """Frames a stopped run never reached count against it."""
    trajectory = _trajectory()
    trajectory["stopped_early"] = np.asarray(1)
    trajectory["frames_planned"] = np.asarray(2 * FRAMES)
    np.savez(tmp_path / "trajectory_cpu.npz", **trajectory)
    metrics = evaluate_cpu.evaluate(tmp_path)
    assert metrics["stopped_early"] == 1
    assert metrics["success"] == 0.0
    assert metrics["frac_ok"] == 0.5


def _hand_motion(wobble=0.0, wobble_hz=5.0, smoothing=1):
    """Ten seconds of one wrist (position, quaternion) and one finger at 20 Hz.

    The reference sways at 0.5 Hz with a little 4 Hz on top; the simulated hand
    adds ``wobble`` at ``wobble_hz`` or averages over ``smoothing`` frames.
    """
    frame_dt, frames = 0.05, 200
    time = np.arange(frames) * frame_dt
    sway = np.sin(2 * np.pi * 0.5 * time) + 0.02 * np.sin(2 * np.pi * 4.0 * time)

    def layout(signal):
        position = np.outer(0.1 * signal, [1.0, 0.5, 0.2])
        turn = Rotation.from_rotvec(np.outer(0.3 * signal, [0.0, 0.0, 1.0]))
        quat = turn.as_quat()[:, [3, 0, 1, 2]]  # wxyz
        return np.concatenate([position, quat, 0.5 * signal[:, None]], axis=1)

    signal = sway + wobble * np.sin(2 * np.pi * wobble_hz * time)
    if smoothing > 1:
        signal = np.convolve(signal, np.ones(smoothing) / smoothing, mode="same")
    return {
        "frame": np.arange(frames),
        "frame_dt": np.asarray(frame_dt),
        "qpos": layout(signal),
        "qpos_ref": layout(sway),
        "wrist_quat_q_ids": np.array([3]),
        "finger_q_ids": np.array([7]),
    }


def test_shake_is_one_on_the_reference_and_grows_with_a_wobble():
    """Fast motion the demonstration does not have raises the shake; smoothing lowers it."""
    same = evaluate_cpu.shake(_hand_motion())
    assert set(same) == {"wrist_shake", "wrist_rot_shake", "finger_shake"}
    for value in same.values():
        np.testing.assert_allclose(value, 1.0, rtol=1e-6)
    shaky = evaluate_cpu.shake(_hand_motion(wobble=0.05))
    assert all(value > 2.0 for value in shaky.values())
    # A wobble below the cutoff is slow motion, not shake.
    slow = evaluate_cpu.shake(_hand_motion(wobble=0.05, wobble_hz=1.0))
    assert all(value < 1.3 for value in slow.values())
    smooth = evaluate_cpu.shake(_hand_motion(smoothing=5))
    assert all(value < 0.8 for value in smooth.values())


def test_shake_needs_two_seconds():
    """Runs shorter than two seconds report no shake."""
    short = {
        name: value[:30] if np.ndim(value) and len(value) == 200 else value
        for name, value in _hand_motion().items()
    }
    assert evaluate_cpu.shake(short) == {}


def test_evaluate_takes_the_arrays_of_a_run():
    """The runner scores a trajectory it holds in memory."""
    metrics = evaluate_cpu.evaluate(_trajectory())
    assert metrics["success"] == 1.0
