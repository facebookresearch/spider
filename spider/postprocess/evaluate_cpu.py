# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Metrics of the runs of the CPU workflow and of its GPU twin.

The runs of ``examples/run_cpu.py`` and ``examples/run_chord.py``. Reads
``RUN_DIR/trajectory_cpu.npz`` (or ``trajectory_chord.npz``: the executed world
the runner recorded; nothing is re-simulated) and writes ``RUN_DIR/metrics.json``. Two things are
measured against the reference frame each step targeted: how well the object
follows it and how close the hands stay to it. Coordinates are MuJoCo's
(quaternions wxyz in ``qpos``, xyzw in body poses).

Object:

- ``obj_pos_cm`` / ``obj_rot_deg``: mean over frames of the worst object body.
  A reference may put a body's frame elsewhere on the body than the simulated
  link frame (a lid referenced at its centre, simulated at its hinge), so the
  constant body-frame offset measured at reset is applied before comparing.
- ``art_err_deg``: mean absolute error of the object's articulation (hinge).
- ``art_events`` / ``art_events_hit``: the reference's openings (excursions of
  the hinge from where it starts) and how many of them the simulated hinge
  follows at least half of the way. A lid that never opens misses its events
  even when the clip is long enough to hide that in the mean error.
- ``success``: 1 if every object body's mean error is under ``pos_tolerance``
  and ``rot_tolerance``, the mean hinge error is under ``art_tolerance``, every
  opening is followed and the run was not stopped early. ``success_object`` is
  the object part alone. ``frac_ok``: share of the reference's frames with the
  object within both tolerances and every hinge within ``rot_tolerance``.
- ``fallen_frames``: frames with the object more than 10 cm below its lowest
  reference height, i.e. off the support surface it should rest on.
- ``backtracks``: how often the run lost the object, went back and planned the
  stretch again (the saved trajectory is the last try of every stretch).

Hands:

- ``wrist_pos_cm`` / ``wrist_rot_deg``: mean over frames of the worse wrist.
- ``finger_deg``: mean absolute finger-joint distance from the reference.
- ``wrist_shake`` / ``wrist_rot_shake`` / ``finger_shake``: how shaky the motion
  is. The speed of the wrist positions (wrist rotations, finger joints) at 3 Hz
  and above, as a multiple of the same quantity of the reference: 1 is as
  smooth as the demonstration, 2 is a hand base that visibly trembles.

Example:
    python -m spider.postprocess.evaluate_cpu RUN_DIR
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tyro
from scipy.spatial.transform import Rotation


def object_errors(trajectory: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame, per-body object error at the reference's body frames.

    Args:
        trajectory: Arrays of a run's ``trajectory_<simulator>.npz``.

    Returns:
        Position error [m] and rotation error [rad], each of shape (T, B).
    """
    sim = trajectory["object_pose"].astype(np.float64)  # (T, B, 7) position + xyzw
    ref = trajectory["object_pose_ref"][trajectory["frame"]]
    reset = trajectory["object_pose_reset"].astype(np.float64)
    ref0 = trajectory["object_pose_ref"][0]
    pos = np.zeros(sim.shape[:2])
    rot = np.zeros(sim.shape[:2])
    for b in range(sim.shape[1]):
        r_reset = Rotation.from_quat(reset[b, 3:])
        # Where the reference puts this body's frame, expressed in the link frame.
        offset = r_reset.inv().apply(ref0[b, :3] - reset[b, :3])
        twist = r_reset.inv() * Rotation.from_quat(ref0[b, 3:])
        r_sim = Rotation.from_quat(sim[:, b, 3:])
        pos[:, b] = np.linalg.norm(
            sim[:, b, :3] + r_sim.apply(offset) - ref[:, b, :3], axis=1
        )
        rot[:, b] = (
            (r_sim * twist) * Rotation.from_quat(ref[:, b, 3:]).inv()
        ).magnitude()
    return pos, rot


def fallen(trajectory: dict[str, np.ndarray], margin: float = 0.10) -> np.ndarray:
    """Frames in which an object body is far below anything the reference does.

    An object that slips normally lands on the support it started on. One that
    drops more than ``margin`` below the lowest height its reference ever
    reaches has left the support: the surface under it is smaller than the
    table the motion was recorded on.

    Args:
        trajectory: Arrays of a run's ``trajectory_<simulator>.npz``.
        margin: Drop below the reference's lowest height that counts [m].

    Returns:
        Boolean array of shape (T,).
    """
    sim = trajectory["object_pose"][..., 2].astype(np.float64)  # (T, B)
    ref = trajectory["object_pose_ref"][..., 2].astype(np.float64)
    start = trajectory["object_pose_reset"][:, 2].astype(np.float64)
    lowest = (ref - ref[0]).min(axis=0)  # the reference's largest descent, per body
    return ((sim - start) < lowest - margin).any(axis=1)


def excursions(
    ref: np.ndarray, floor: float = 15.0, share: float = 0.3
) -> list[tuple[int, int, float]]:
    """The reference's hinge excursions (lid openings).

    An excursion is a run of frames in which the reference is further from its
    starting angle than ``floor`` degrees and ``share`` of its largest
    excursion.

    Args:
        ref: Reference hinge angle per frame [rad], shape (T,).
        floor: Smallest excursion that counts [deg].
        share: Smallest excursion that counts, as a share of the largest.

    Returns:
        ``(start, stop, peak)`` per excursion: its frames ``[start, stop)`` and
        the signed angle [deg] from the starting angle at its peak.
    """
    away = np.degrees(ref - ref[0])
    peak = np.abs(away).max() if len(away) else 0.0
    active = np.abs(away) > max(floor, share * peak)
    if not active.any():
        return []
    edges = np.flatnonzero(np.diff(np.r_[0, active.astype(np.int8), 0]))
    found = []
    for start, stop in zip(edges[::2], edges[1::2], strict=True):
        top = start + int(np.abs(away[start:stop]).argmax())
        found.append((int(start), int(stop), float(away[top])))
    return found


def followed(sim: np.ndarray, start_angle: float, peak: float) -> bool:
    """Whether a simulated hinge follows an excursion at least half of the way.

    Args:
        sim: Simulated hinge angle over the excursion's frames [rad].
        start_angle: Reference hinge angle at the first frame of the clip [rad].
        peak: Signed angle of the excursion's peak from ``start_angle`` [deg].

    Returns:
        True if at some frame the hinge has gone at least half as far in the
        same direction as the reference goes at its peak.
    """
    along = np.degrees(sim - start_angle) * np.sign(peak)
    return bool(len(along) and along.max() >= 0.5 * abs(peak))


def articulation_events(
    sim: np.ndarray, ref: np.ndarray, floor: float = 15.0, share: float = 0.3
) -> tuple[int, int]:
    """Count the reference's hinge excursions and those the simulation follows.

    Args:
        sim: Simulated hinge angle per frame [rad], shape (T,).
        ref: Reference hinge angle of the same frames [rad], shape (T,).
        floor: Smallest excursion that counts [deg], see :func:`excursions`.
        share: Smallest excursion that counts, as a share of the largest.

    Returns:
        Number of excursions and number followed (:func:`followed`).
    """
    events = excursions(ref, floor, share)
    hits = sum(followed(sim[start:stop], ref[0], peak) for start, stop, peak in events)
    return len(events), hits


def _fast_part(speed: np.ndarray, frame_dt: float, cutoff_hz: float) -> float:
    """Size of the part of ``speed`` (T, D) at and above ``cutoff_hz``."""
    spectrum = np.abs(np.fft.rfft(speed * np.hanning(len(speed))[:, None], axis=0))
    fast = np.fft.rfftfreq(len(speed), frame_dt) >= cutoff_hz
    return float(np.sqrt(np.sum(spectrum[fast] ** 2)))


def _turn_rate(quat: np.ndarray, frame_dt: float) -> np.ndarray:
    """Angular velocity between consecutive wxyz quaternions (T, 4), (T - 1, 3)."""
    rotation = Rotation.from_quat(quat[:, [1, 2, 3, 0]])
    return (rotation[:-1].inv() * rotation[1:]).as_rotvec() / frame_dt


def shake(
    trajectory: dict[str, np.ndarray], cutoff_hz: float = 3.0
) -> dict[str, float]:
    """How shaky the hands are: their fast motion as a multiple of the reference's.

    A planner that corrects the hands at every tick adds motion the
    demonstration does not have, above the few hertz a hand moves at. The
    speed of the wrist positions, of the wrist rotations and of the finger
    joints is split at ``cutoff_hz``, and the part at and above it is divided by
    the same part of the reference. 1 is as smooth as the demonstration.

    Args:
        trajectory: Arrays of a run's ``trajectory_<simulator>.npz``.
        cutoff_hz: Where fast motion begins.

    Returns:
        ``wrist_shake``, ``wrist_rot_shake`` and ``finger_shake`` (those the
        trajectory has the layout for); empty for runs under two seconds.
    """
    frame_dt = float(trajectory["frame_dt"])
    sim = np.asarray(trajectory["qpos"], dtype=np.float64)
    ref = np.asarray(trajectory["qpos_ref"], dtype=np.float64)[trajectory["frame"]]
    valid = ~np.isnan(sim).any(axis=1)
    sim, ref = sim[valid], ref[valid]
    if len(sim) * frame_dt < 2.0:
        return {}

    def ratio(sim_speed: np.ndarray, ref_speed: np.ndarray) -> float:
        reference = _fast_part(ref_speed, frame_dt, cutoff_hz)
        return _fast_part(sim_speed, frame_dt, cutoff_hz) / max(reference, 1e-9)

    out: dict[str, float] = {}
    wrists = np.asarray(trajectory["wrist_quat_q_ids"]).ravel()
    if len(wrists):
        # A wrist is three position coordinates followed by its quaternion.
        position = np.concatenate([np.arange(q - 3, q) for q in wrists])
        out["wrist_shake"] = ratio(
            np.diff(sim[:, position], axis=0) / frame_dt,
            np.diff(ref[:, position], axis=0) / frame_dt,
        )
        out["wrist_rot_shake"] = ratio(
            np.concatenate(
                [_turn_rate(sim[:, q : q + 4], frame_dt) for q in wrists], axis=1
            ),
            np.concatenate(
                [_turn_rate(ref[:, q : q + 4], frame_dt) for q in wrists], axis=1
            ),
        )
    fingers = trajectory["finger_q_ids"]
    if len(fingers):
        out["finger_shake"] = ratio(
            np.diff(sim[:, fingers], axis=0) / frame_dt,
            np.diff(ref[:, fingers], axis=0) / frame_dt,
        )
    return out


def trajectory_file(run: str | Path) -> Path:
    """The trajectory a run directory holds: the CPU workflow's, else the GPU's."""
    for name in ("trajectory_cpu.npz", "trajectory_chord.npz"):
        if (Path(run) / name).exists():
            return Path(run) / name
    raise FileNotFoundError(f"no trajectory_cpu.npz or trajectory_chord.npz in {run}")


def evaluate(
    run: str | Path | dict[str, np.ndarray],
    pos_tolerance: float = 0.03,
    rot_tolerance: float = 30.0,
    art_tolerance: float = 10.0,
) -> dict[str, float | int | str]:
    """Compute the metrics of one run.

    Args:
        run: Output directory of ``examples/run_cpu.py`` (or ``run_chord.py``),
            or the arrays of its ``trajectory_<simulator>.npz``.
        pos_tolerance: Object position tolerance [m] of ``success`` / ``frac_ok``.
        rot_tolerance: Object rotation tolerance [deg] of ``success`` / ``frac_ok``.
        art_tolerance: Tolerance of the mean hinge error [deg] of ``success``.

    Returns:
        Metric name to value.
    """
    if isinstance(run, dict):
        t = run
    else:
        with np.load(trajectory_file(run), allow_pickle=False) as data:
            t = {name: data[name] for name in data.files}
    frame = t["frame"]
    pos, rot = object_errors(t)  # (T, B)
    worst_pos, worst_rot = pos.max(axis=1), np.degrees(rot.max(axis=1))
    body_ok = (pos.mean(axis=0) < pos_tolerance) & (
        np.degrees(rot.mean(axis=0)) < rot_tolerance
    )
    frame_ok = (worst_pos < pos_tolerance) & (worst_rot < rot_tolerance)
    stopped, planned = bool(t["stopped_early"]), int(t["frames_planned"])
    metrics: dict[str, float | int | str] = {
        "frames": len(frame),
        "frames_planned": planned,
        "stopped_early": int(stopped),
        "backtracks": len(t["backtrack_from"]),
        "obj_pos_cm": float(worst_pos.mean() * 100),
        "obj_pos_cm_max": float(worst_pos.max() * 100),
        "obj_rot_deg": float(worst_rot.mean()),
        "success_object": float(body_ok.all()),
        "wrist_pos_cm": float(t["wrist_pos_err"].max(axis=1).mean() * 100),
        "wrist_pos_cm_max": float(t["wrist_pos_err"].max() * 100),
        "wrist_rot_deg": float(np.degrees(t["wrist_rot_err"].max(axis=1).mean())),
        "reward": float(t["reward"].mean()),
        "wall_time_s": float(t["wall_time"]),
        "plan_time_s": float(t["opt_plan_time"].mean()),
        "nan_frames": int(np.isnan(t["qpos"]).any(axis=1).sum()),
    }
    metrics["fallen_frames"] = int(fallen(t).sum())
    metrics.update(shake(t))
    fingers = t["finger_q_ids"]
    if len(fingers):
        gap = np.degrees(
            np.abs(t["qpos"][:, fingers] - t["qpos_ref"][frame][:, fingers])
        )
        metrics["finger_deg"] = float(gap.mean())
        metrics["finger_deg_p95"] = float(np.percentile(gap, 95))
    hinges = t["articulation_q_ids"]
    hinge_ok = True
    if len(hinges):
        sim_q, ref_q = t["qpos"][:, hinges], t["qpos_ref"][frame][:, hinges]
        gap = np.degrees(np.abs(sim_q - ref_q))  # (T, hinges)
        counts = [
            articulation_events(sim_q[:, k], ref_q[:, k]) for k in range(len(hinges))
        ]
        events, hits = (int(sum(c[k] for c in counts)) for k in (0, 1))
        metrics["art_err_deg"] = float(gap.mean())
        metrics["art_max_deg"] = float(np.degrees(sim_q.max()))
        metrics["art_ref_max_deg"] = float(np.degrees(t["qpos_ref"][:, hinges].max()))
        metrics["art_events"], metrics["art_events_hit"] = events, hits
        hinge_ok = bool((gap.mean(axis=0) < art_tolerance).all()) and hits == events
        frame_ok &= (gap < rot_tolerance).all(axis=1)
    metrics["success"] = float(body_ok.all() and hinge_ok and not stopped)
    # Frames a stopped run never reached count as out of tolerance.
    metrics["frac_ok"] = float(frame_ok.sum() / max(planned, len(frame)))
    for b, name in enumerate(t["body_names"]):
        metrics[f"obj_pos_cm_{name}"] = float(pos[:, b].mean() * 100)
        metrics[f"obj_rot_deg_{name}"] = float(np.degrees(rot[:, b].mean()))
    return metrics


def summary(metrics: dict[str, float | int | str]) -> str:
    """One line with the numbers to look at first."""
    hinge = ""
    if "art_err_deg" in metrics:
        hinge = (
            f" hinge {metrics['art_err_deg']:.1f} deg openings "
            f"{metrics['art_events_hit']}/{metrics['art_events']}"
        )
    # Runs shorter than two seconds have no shake.
    shake = (
        f"shake x{metrics['wrist_shake']:.2f} | " if "wrist_shake" in metrics else ""
    )
    return (
        f"object {metrics['obj_pos_cm']:.2f} cm "
        f"{metrics['obj_rot_deg']:.1f} deg{hinge} success {metrics['success']:.0f} "
        f"frac_ok {metrics['frac_ok']:.2f} | wrist {metrics['wrist_pos_cm']:.2f} cm "
        f"{metrics['wrist_rot_deg']:.1f} deg fingers "
        f"{metrics.get('finger_deg', float('nan')):.1f} deg | "
        f"{shake}wall {metrics['wall_time_s']:.0f} s"
    )


def main(
    runs: tyro.conf.Positional[tuple[str, ...]],
    pos_tolerance: float = 0.03,
    rot_tolerance: float = 30.0,
    art_tolerance: float = 10.0,
) -> None:
    """Write ``metrics.json`` into each run directory and print a summary line.

    Args:
        runs: Output directories of ``examples/run_cpu.py`` or ``run_chord.py``.
        pos_tolerance: Object position tolerance [m].
        rot_tolerance: Object rotation tolerance [deg].
        art_tolerance: Tolerance of the mean hinge error [deg].
    """
    for run in runs:
        metrics = evaluate(run, pos_tolerance, rot_tolerance, art_tolerance)
        (Path(run) / "metrics.json").write_text(json.dumps(metrics, indent=1))
        print(f"{run}: {summary(metrics)}")
        if metrics["backtracks"]:
            print(
                f"  lost the object {metrics['backtracks']} times, went back and "
                "planned the stretch again"
            )
        if metrics["stopped_early"]:
            print(
                f"  stopped early after {metrics['frames']} of "
                f"{metrics['frames_planned']} frames"
            )
        if metrics["fallen_frames"]:
            print(
                "  the object is more than 10 cm below its lowest reference height in "
                f"{metrics['fallen_frames']} frames: it left its support surface"
            )


if __name__ == "__main__":
    tyro.cli(main)
