# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""SPIDER's CPU simulator: sampling MPC on MuJoCo's CPU batch API.

Self-contained: MuJoCo (>= 3.17, for ``mujoco.batch``), NumPy and Torch. The
worlds are one ``mujoco.batch.Batch``: double precision, a thread pool, and
results that do not depend on the thread count.

It has the interface of :mod:`spider.simulators.mjwp` (``setup_mj_model``,
``load_data``, ``setup_env``, ``step_env``, ``get_reward``, ``save_state`` /
``load_state``, ``sync_env``, ``get_qpos``, ...), so SPIDER's optimizer and the
MJWP runner's protocol drive it unchanged; ``examples/run_cpu.py`` is its runner.
Like the HDMI and DexMachina simulators, the environment keeps the reference
clock: ``get_reward`` scores each world against the reference frame it is at,
and the per-step ``ref`` the optimizer passes is not needed.

A scene is a directory with two files, in SPIDER's dataset layout
(``processed/<dataset>/<robot>/<embodiment>/<task>/<data_id>/``):

- ``scene.xml``: the MuJoCo model, ready to simulate.
- ``scene.npz``: the reference motion and the tables of the control law
  (written for scenes of FlashCHORD by :mod:`spider.simulators.chord_export`).

The control law is FlashCHORD's, in NumPy. The control is a residual on the
reference in physical units: per hand 3 wrist position residuals [m], 3 wrist
roll/pitch/yaw residuals [rad] and the finger joint residuals [rad]. It is
filtered (EMA) and clipped, then

- wrist translations and fingers track ``reference + residual`` through the
  model's position and velocity actuators (held for one control frame);
- each ball wrist gets a clamped proportional torque towards
  ``reference * rpy(residual)``, recomputed every physics substep.

A zero control replays the reference (its ``ctrl`` reference is zero). One step
is one control frame, and the reward is SPIDER's tracking cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import loguru
import mujoco
import numpy as np
import torch

from spider.config import Config

# Coordinate groups of SPIDER's tracking cost in ``scene.npz``.
GROUP_WRIST_POS, GROUP_FINGER, GROUP_OBJECT_POS, GROUP_ARTICULATION = 0, 1, 2, 3
GROUP_WRIST_ROT, GROUP_OBJECT_ROT = 0, 1


@dataclass
class CpuEnv:
    """A batch of CPU worlds plus the controller state MuJoCo does not hold."""

    batch: Any  # mujoco.batch.Batch
    mj_model: mujoco.MjModel
    device: str
    num_worlds: int
    num_frames: int
    frame_dt: float
    substeps: int
    nq: int
    nv: int
    nu: int  # residual action size
    sides: tuple[str, ...]
    scene: dict[str, np.ndarray]  # the tables of scene.npz
    qpos: np.ndarray  # (N, nq) live
    qvel: np.ndarray  # (N, nv) live
    ctrl: np.ndarray  # (N, nu_mujoco) live
    qfrc: np.ndarray  # (N, nv) live applied generalized forces
    xpos: np.ndarray  # (N, nbody, 3) read-only, current after each frame
    xquat: np.ndarray  # (N, nbody, 4) wxyz
    filtered: np.ndarray  # (N, nu) EMA-filtered residual
    timestep: np.ndarray  # (N,) reference frame of the next transition
    frame: np.ndarray  # (N,) reference frame of the last transition
    target_quat: np.ndarray  # (N, H, 4) xyzw wrist orientation targets
    failed: np.ndarray  # (N,) worlds whose simulation raised
    mj_qpos_ref: np.ndarray  # (T, nq) reference
    qvel_ref: np.ndarray  # (T, V) finite-difference reference velocities
    wrist_pos_ref: np.ndarray  # (T, H, 3) wrist body positions of the reference
    wrist_quat_ref: np.ndarray  # (T, H, 4) xyzw
    trace_body: np.ndarray  # bodies drawn as traces: the object root, then the wrists
    tip_body: np.ndarray  # (F,) last link of every finger, drawn as traces too
    tip_local: np.ndarray  # (F, 3) the fingertip in that link's frame
    trace_ref: np.ndarray  # (T, K, 3) reference of the trace points, K = 1 + H + F


# --
# Quaternions (xyzw unless named otherwise)
# --


def _wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    return np.concatenate([quat[..., 1:], quat[..., :1]], axis=-1)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    out = np.empty(np.broadcast_shapes(a.shape, b.shape))
    out[..., 0] = aw * bx + bw * ax + (ay * bz - az * by)
    out[..., 1] = aw * by + bw * ay + (az * bx - ax * bz)
    out[..., 2] = aw * bz + bw * az + (ax * by - ay * bx)
    out[..., 3] = aw * bw - (ax * bx + ay * by + az * bz)
    return out


def _quat_conj(quat: np.ndarray) -> np.ndarray:
    return np.concatenate([-quat[..., :3], quat[..., 3:]], axis=-1)


def _quat_to_rotvec(quat: np.ndarray) -> np.ndarray:
    quat = np.where(quat[..., 3:] < 0.0, -quat, quat)
    norm = np.sqrt((quat[..., :3] * quat[..., :3]).sum(-1, keepdims=True))
    angle = 2.0 * np.arctan2(norm, quat[..., 3:])
    return quat[..., :3] * (angle / np.maximum(norm, 1e-12))


def _rotvec_to_quat(rotvec: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    scale = np.where(angle > 1e-9, np.sin(0.5 * angle) / np.maximum(angle, 1e-12), 0.5)
    return np.concatenate([rotvec * scale, np.cos(0.5 * angle)], axis=-1)


def _quat_rpy(rpy: np.ndarray) -> np.ndarray:
    """Roll about x, then pitch about y, then yaw about z (fixed axes)."""
    half = 0.5 * rpy
    cr, cp, cy = np.cos(half[..., 0]), np.cos(half[..., 1]), np.cos(half[..., 2])
    sr, sp, sy = np.sin(half[..., 0]), np.sin(half[..., 1]), np.sin(half[..., 2])
    return np.stack(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        axis=-1,
    )


def _rotate_wxyz(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate vectors by MuJoCo (wxyz) quaternions, broadcasting."""
    axis = quat[..., 1:]
    twice = 2.0 * np.cross(axis, vec)
    return vec + quat[..., :1] * twice + np.cross(axis, twice)


# --
# Setup
# --


def fingertips(
    model: mujoco.MjModel, wrist_body: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The last link of every finger, and the fingertip in that link's frame.

    A finger ends in a body without children below a wrist. Its tip is the point
    of its geometry furthest from the body's origin (the last joint): a mesh
    vertex, or for a primitive the far point of its bounding sphere.

    Args:
        model: The scene.
        wrist_body: Body ids of the hand bases.

    Returns:
        Body ids, (F,), and the tips in those bodies' frames, (F, 3).
    """
    wrists = {int(body) for body in wrist_body}
    parents = set(model.body_parentid[1:].tolist())
    bodies, tips = [], []
    for body in range(1, model.nbody):
        root = body
        while root != 0 and root not in wrists:
            root = int(model.body_parentid[root])
        if root == 0 or body in wrists or body in parents:
            continue
        points = [np.zeros((1, 3))]
        first = model.body_geomadr[body]
        for geom in range(first, first + model.body_geomnum[body]):
            pos, mat = model.geom_pos[geom], np.zeros(9)
            mujoco.mju_quat2Mat(mat, model.geom_quat[geom])
            mesh = model.geom_dataid[geom]
            if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH and mesh >= 0:
                start = model.mesh_vertadr[mesh]
                vertices = model.mesh_vert[start : start + model.mesh_vertnum[mesh]]
                points.append(vertices @ mat.reshape(3, 3).T + pos)
            else:
                reach = pos / max(np.linalg.norm(pos), 1e-9) * model.geom_rbound[geom]
                points.append((pos + reach)[None])
        points = np.concatenate(points)
        bodies.append(body)
        tips.append(points[np.linalg.norm(points, axis=1).argmax()])
    return np.asarray(bodies, dtype=np.int64), np.asarray(tips).reshape(-1, 3)


def _tracking_weights(config: Config, scene: dict[str, np.ndarray]) -> None:
    """SPIDER's reward scales for each tracked coordinate."""
    scalar = {
        GROUP_WRIST_POS: config.base_pos_rew_scale,
        GROUP_FINGER: config.joint_rew_scale,
        GROUP_OBJECT_POS: config.pos_rew_scale,
        GROUP_ARTICULATION: config.rot_rew_scale,
    }
    quat = {
        GROUP_WRIST_ROT: config.base_rot_rew_scale,
        GROUP_OBJECT_ROT: config.rot_rew_scale,
    }
    scene["scalar_w"] = np.array([scalar[int(g)] for g in scene["scalar_group"]])
    scene["quat_w"] = np.array([quat[int(g)] for g in scene["quat_group"]])


def setup_mj_model(config: Config) -> mujoco.MjModel:
    """The scene's MuJoCo model (``config.model_path``), for the viewer and the video."""
    return mujoco.MjModel.from_xml_path(str(Path(config.model_path).expanduser()))


def _reference(scene: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """The reference ``qpos`` and its finite-difference ``qvel`` on the tracked dofs."""
    qpos_ref = scene["qpos_ref"]
    nxt = np.concatenate([qpos_ref[1:], qpos_ref[-1:]])
    qvel_ref = (nxt[:, scene["vel_q"]] - qpos_ref[:, scene["vel_q"]]) / float(
        scene["frame_dt"]
    )
    return qpos_ref, qvel_ref


def load_data(config: Config, data_path: str | None = None) -> tuple[torch.Tensor, ...]:
    """The reference of a scene, as :func:`spider.io.load_data` gives it for MJWP.

    Returns ``(qpos_ref, qvel_ref, ctrl_ref, contact, contact_pos)``: ``qpos``
    (T, nq) and ``qvel`` (T, nv) in the scene's coordinates, the ``ctrl`` of a
    replay (T, nu), which is a zero residual, and no contacts (T, 0), (T, 0, 3).

    Args:
        config: ``device``, and ``data_path`` by default.
        data_path: The scene's ``scene.npz``.
    """
    path = Path(data_path or config.data_path).expanduser()
    with np.load(path) as archive:
        scene = {name: archive[name] for name in archive.files}
    qpos_ref, qvel_tracked = _reference(scene)
    qvel_ref = np.zeros(scene["targets0"].shape)  # (T, nv)
    qvel_ref[:, scene["vel_dof"]] = qvel_tracked
    frames, device = len(qpos_ref), config.device
    return (
        torch.from_numpy(qpos_ref.astype(np.float64)).to(device),
        torch.from_numpy(qvel_ref).to(device),
        torch.zeros(frames, len(scene["action_clip"]), device=device),
        torch.zeros(frames, 0, device=device),
        torch.zeros(frames, 0, 3, device=device),
    )


def setup_env(
    config: Config,
    ref_data: tuple[torch.Tensor, ...] | None = None,
    num_threads: int = 0,
) -> CpuEnv:
    """Build ``config.num_samples`` worlds of the scene at reference frame 0.

    Args:
        config: ``model_path`` is the scene's ``scene.xml`` and ``data_path``
            its ``scene.npz``.
        ref_data: The reference of :func:`load_data`; its ``qpos`` and ``qvel``
            replace the scene's when given.
        num_threads: Threads of the batch (0: one per CPU).
    """
    try:
        from mujoco import batch
    except ImportError as error:
        raise ImportError(
            f"simulator=cpu needs mujoco.batch (MuJoCo >= 3.17); found "
            f"{mujoco.__version__}"
        ) from error
    model = mujoco.MjModel.from_xml_path(str(Path(config.model_path).expanduser()))
    with np.load(Path(config.data_path).expanduser()) as archive:
        scene = {name: archive[name] for name in archive.files}
    _tracking_weights(config, scene)
    frame_dt, substeps = float(scene["frame_dt"]), int(scene["substeps"])

    num_worlds = int(config.num_samples)
    worlds = batch.Batch(model, num_worlds, nthread=int(num_threads))
    num_hands = len(scene["sides"])
    qpos_ref, qvel_ref = _reference(scene)
    if ref_data is not None:
        qpos_ref = ref_data[0].detach().cpu().numpy().astype(np.float64)
        qvel_ref = ref_data[1].detach().cpu().numpy()[:, scene["vel_dof"]]

    # Body positions of the reference: the wrists for the report, and the
    # object root, the wrists and the fingertips for the viewer's traces.
    trace_body = np.concatenate([scene["object_body"][:1], scene["wrist_body"]])
    tip_body, tip_local = fingertips(model, scene["wrist_body"])
    data = mujoco.MjData(model)
    wrist_pos_ref = np.zeros((len(qpos_ref), num_hands, 3))
    wrist_quat_ref = np.zeros((len(qpos_ref), num_hands, 4))
    trace_ref = np.zeros((len(qpos_ref), len(trace_body) + len(tip_body), 3))
    for frame, qpos in enumerate(qpos_ref):
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        wrist_pos_ref[frame] = data.xpos[scene["wrist_body"]]
        wrist_quat_ref[frame] = _wxyz_to_xyzw(data.xquat[scene["wrist_body"]])
        trace_ref[frame, : len(trace_body)] = data.xpos[trace_body]
        trace_ref[frame, len(trace_body) :] = data.xpos[tip_body] + _rotate_wxyz(
            data.xquat[tip_body], tip_local
        )

    env = CpuEnv(
        batch=worlds,
        mj_model=model,
        device="cpu",
        num_worlds=num_worlds,
        num_frames=len(qpos_ref),
        frame_dt=frame_dt,
        substeps=substeps,
        nq=model.nq,
        nv=model.nv,
        nu=len(scene["action_clip"]),
        sides=tuple(str(s) for s in scene["sides"]),
        scene=scene,
        qpos=worlds.bind("qpos"),
        qvel=worlds.bind("qvel"),
        ctrl=worlds.bind("ctrl"),
        qfrc=worlds.bind("qfrc_applied"),
        xpos=worlds.bind("xpos"),
        xquat=worlds.bind("xquat"),
        filtered=np.zeros((num_worlds, len(scene["action_clip"]))),
        timestep=np.zeros(num_worlds, dtype=np.int64),
        frame=np.zeros(num_worlds, dtype=np.int64),
        target_quat=np.zeros((num_worlds, num_hands, 4)),
        failed=np.zeros(num_worlds, dtype=bool),
        mj_qpos_ref=qpos_ref,
        qvel_ref=qvel_ref,
        wrist_pos_ref=wrist_pos_ref,
        wrist_quat_ref=wrist_quat_ref,
        trace_body=trace_body,
        tip_body=tip_body,
        tip_local=tip_local,
        trace_ref=trace_ref,
    )
    reset_env(config, env, 0)
    loguru.logger.info(
        "cpu: {} worlds on {} threads, {} frames at {:.0f} Hz x {} substeps, "
        "nq {} nv {} nu {}, MuJoCo {}",
        num_worlds,
        worlds.nthread,
        env.num_frames,
        1.0 / frame_dt,
        substeps,
        env.nq,
        env.nv,
        env.nu,
        mujoco.__version__,
    )
    return env


def reset_env(config: Config, env: CpuEnv, frame_id: int = 0) -> CpuEnv:
    """Put every world on reference frame ``frame_id`` at rest."""
    del config
    env.batch.reset(stage=int(mujoco.mjtStage.mjSTAGE_NONE))
    env.qpos[:] = env.mj_qpos_ref[frame_id]
    env.qvel[:] = 0.0
    env.batch.forward(stage=int(mujoco.mjtStage.mjSTAGE_POS))
    env.filtered[:] = 0.0
    env.timestep[:] = frame_id
    env.frame[:] = frame_id
    env.failed[:] = False
    return env


# --
# Stepping and state
# --


def _command(env: CpuEnv, ctrl: np.ndarray) -> None:
    """Residual action -> filtered residual -> actuator targets and wrist targets."""
    scene = env.scene
    ema = float(scene["action_ema"])
    clip = scene["action_clip"]
    env.filtered[:] = np.clip(ema * env.filtered + (1.0 - ema) * ctrl, -clip, clip)
    env.frame[:] = np.clip(env.timestep, 0, env.num_frames - 1)
    target = scene["targets0"][env.frame]  # (N, nv), a copy
    start = 0
    for hand in range(len(env.sides)):
        fingers = scene["finger_dof"][hand]
        residual = env.filtered[:, start : start + 6 + len(fingers)]
        orientation = scene["wrist_ori_dof"][hand]
        reference = _rotvec_to_quat(target[:, orientation])
        env.target_quat[:, hand] = _quat_mul(reference, _quat_rpy(residual[:, 3:6]))
        target[:, scene["wrist_pos_dof"][hand]] += residual[:, :3]
        target[:, fingers] += residual[:, 6:]
        start += 6 + len(fingers)
    env.ctrl[:] = 0.0
    env.ctrl[:, scene["position_actuator"]] = target[:, scene["position_actuator_dof"]]


def _wrist_torque(env: CpuEnv) -> None:
    """Clamped proportional torque of each ball wrist toward its target, in the child frame."""
    scene = env.scene
    # Both wrists at once: this runs every substep.
    address = scene["wrist_qpos"].astype(np.int64)[:, None] + np.array([1, 2, 3, 0])
    current = env.qpos[:, address]  # (N, hands, 4) as xyzw, a copy
    current /= np.sqrt((current * current).sum(-1, keepdims=True))
    error = _quat_to_rotvec(_quat_mul(_quat_conj(current), env.target_quat))
    limit = np.asarray(scene["wrist_tau_max"], dtype=np.float64)[:, None]
    gain = np.asarray(scene["wrist_kp"], dtype=np.float64)[:, None]
    torque = np.clip(gain * error, -limit, limit)
    dofs = np.asarray(scene["wrist_ori_dof"]).reshape(-1)
    env.qfrc[:, dofs] = torque.reshape(len(torque), -1)


def step_env(config: Config, env: CpuEnv, ctrl: torch.Tensor) -> None:
    """Advance one control frame with residual controls of shape (N, nu) or (nu,)."""
    del config
    from mujoco import batch

    command = ctrl.detach().cpu().numpy().astype(np.float64)
    if command.ndim == 1:
        command = np.broadcast_to(command, (env.num_worlds, env.nu))
    _command(env, command)
    for substep in range(env.substeps):
        _wrist_torque(env)
        # Body poses are wanted at the new state after the last substep only.
        last = substep == env.substeps - 1
        stage = mujoco.mjtStage.mjSTAGE_POS if last else mujoco.mjtStage.mjSTAGE_NONE
        try:
            env.batch.step(nstep=1, stage=int(stage))
        except batch.SimulationError as error:
            env.failed[np.asarray(error.ids, dtype=np.int64)] = True
    env.timestep[:] = np.minimum(env.timestep + 1, env.num_frames - 1)


def _state_arrays(env: CpuEnv) -> dict[str, np.ndarray]:
    return {
        "state": env.batch.state,
        "filtered": env.filtered,
        "timestep": env.timestep,
        "failed": env.failed,
    }


def save_state(env: CpuEnv) -> dict[str, np.ndarray]:
    """Copy the state of every world."""
    return {name: array.copy() for name, array in _state_arrays(env).items()}


def load_state(env: CpuEnv, state: dict[str, np.ndarray]) -> CpuEnv:
    """Restore a state saved by :func:`save_state`."""
    for name, array in _state_arrays(env).items():
        array[:] = state[name]
    return env


def sync_env(
    config: Config, env: CpuEnv, mj_data: mujoco.MjData | None = None
) -> CpuEnv:
    """Broadcast the first world (the executed one) to every world, as MJWP does.

    ``mj_data`` (the viewer's copy of that world) is not needed: the worlds also
    hold the controller's state, which it does not have.
    """
    del config, mj_data
    for array in _state_arrays(env).values():
        array[:] = array[:1].copy()
    return env


def copy_sample_state(
    config: Config,
    env: CpuEnv,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
) -> None:
    """Copy the state of worlds ``src_indices`` onto worlds ``dst_indices``."""
    del config
    src, dst = src_indices.cpu().numpy(), dst_indices.cpu().numpy()
    for array in _state_arrays(env).values():
        array[dst] = array[src]


def save_env_params(config: Config, env: CpuEnv) -> dict:
    """No per-rollout parameters."""
    del config, env
    return {}


def load_env_params(config: Config, env: CpuEnv, env_param: dict) -> CpuEnv:
    """No per-rollout parameters."""
    del config, env_param
    return env


# --
# Rewards
# --


def spider_cost(config: Config, env: CpuEnv) -> tuple[np.ndarray, np.ndarray]:
    """SPIDER's weighted pose distance and joint-velocity distance per world."""
    del config
    scene = env.scene
    ref = env.mj_qpos_ref[env.frame]
    scalar = (env.qpos[:, scene["scalar_q"]] - ref[:, scene["scalar_q"]]) * scene[
        "scalar_w"
    ]
    quat_ids = scene["quat_q"][:, None] + np.arange(4)[None, :]
    quat = _wxyz_to_xyzw(env.qpos[:, quat_ids])
    rot = _quat_to_rotvec(_quat_mul(quat, _quat_conj(_wxyz_to_xyzw(ref[:, quat_ids]))))
    rot = rot * scene["quat_w"][None, :, None]
    weighted = np.concatenate([scalar, rot.reshape(len(rot), -1)], axis=1)
    qpos_dist = np.linalg.norm(weighted, axis=1)
    qvel = env.qvel[:, scene["vel_dof"]] - env.qvel_ref[env.frame]
    return qpos_dist, np.linalg.norm(qvel, axis=1)


def _pose_errors(
    pos: np.ndarray, quat: np.ndarray, ref_pos: np.ndarray, ref_quat: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Distance [m] and angle [rad] between poses (..., 3) and xyzw (..., 4)."""
    rot = _quat_to_rotvec(_quat_mul(quat, _quat_conj(ref_quat)))
    return np.linalg.norm(pos - ref_pos, axis=-1), np.linalg.norm(rot, axis=-1)


def _object_errors(env: CpuEnv) -> tuple[np.ndarray, np.ndarray]:
    """Position [m] and rotation [rad] error of every object body, (N, B)."""
    ref = env.scene["object_pose_ref"][env.frame]
    bodies = env.scene["object_body"]
    return _pose_errors(
        env.xpos[:, bodies],
        _wxyz_to_xyzw(env.xquat[:, bodies]),
        ref[..., :3],
        ref[..., 3:],
    )


def get_reward(
    config: Config, env: CpuEnv, ref: tuple[torch.Tensor, ...] | None = None
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Per-world SPIDER tracking reward of the last transition."""
    del ref
    qpos_dist, qvel_dist = spider_cost(config, env)
    reward = -qpos_dist - config.vel_rew_scale * qvel_dist
    reward[env.failed] = -np.inf
    pos, rot = _object_errors(env)
    tensor = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float()  # noqa: E731
    info = {
        "qpos_dist": tensor(qpos_dist),
        "object_pos_err": tensor(pos.max(axis=1)),
        "object_rot_err": tensor(rot.max(axis=1)),
    }
    return tensor(reward), info


def get_terminal_reward(
    config: Config, env: CpuEnv, ref: tuple[torch.Tensor, ...] | None = None
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Reward of the last rollout step."""
    reward, info = get_reward(config, env, ref)
    return config.terminal_rew_scale * reward, info


def get_terminate(
    config: Config, env: CpuEnv, ref: tuple[torch.Tensor, ...] | None = None
) -> torch.Tensor:
    """Worlds whose simulation failed."""
    del config, ref
    return torch.from_numpy(env.failed.copy())


# --
# Readouts
# --


def get_qpos(config: Config, env: CpuEnv) -> torch.Tensor:
    """MuJoCo ``qpos`` of every world, (N, nq)."""
    del config
    return torch.from_numpy(env.qpos.copy())


def set_qpos(config: Config, env: CpuEnv, qpos: torch.Tensor) -> CpuEnv:
    """Put every world on ``qpos`` (nq,) or (N, nq), at rest, and update its poses."""
    del config
    env.qpos[:] = qpos.detach().cpu().numpy()
    env.qvel[:] = 0.0
    env.batch.forward(stage=int(mujoco.mjtStage.mjSTAGE_POS))
    return env


def get_qvel(config: Config, env: CpuEnv) -> torch.Tensor:
    """MuJoCo ``qvel`` of every world, (N, nv)."""
    del config
    return torch.from_numpy(env.qvel.copy())


def get_trace(config: Config, env: CpuEnv) -> torch.Tensor:
    """Positions of the object root, the wrists and the fingertips, (N, K, 3)."""
    del config
    tips = env.xpos[:, env.tip_body] + _rotate_wxyz(
        env.xquat[:, env.tip_body], env.tip_local
    )
    trace = np.concatenate([env.xpos[:, env.trace_body], tips], axis=1)
    return torch.from_numpy(trace).float()


def run_arrays(env: CpuEnv) -> dict[str, np.ndarray]:
    """What a saved trajectory needs besides its frames: the reference and the layout."""
    scene = env.scene
    return {
        "qpos_ref": env.mj_qpos_ref,
        "object_pose_ref": scene["object_pose_ref"],
        "object_pose_reset": scene["object_pose_reset"],
        "articulation_q_ids": scene["articulation_q"],
        "finger_q_ids": scene["scalar_q"][scene["scalar_group"] == GROUP_FINGER],
        "wrist_quat_q_ids": scene["quat_q"][scene["quat_group"] == GROUP_WRIST_ROT],
        "body_names": scene["body_names"],
        "sides": scene["sides"],
        "frame_dt": np.asarray(env.frame_dt),
    }


def read_step(config: Config, env: CpuEnv, world: int = 0) -> dict[str, np.ndarray]:
    """Record of the last transition in one world."""
    reward, _ = get_reward(config, env)
    scene, frame = env.scene, env.frame[world]
    pos, rot = _object_errors(env)
    wrists = scene["wrist_body"]
    wrist_pos, wrist_rot = _pose_errors(
        env.xpos[world, wrists],
        _wxyz_to_xyzw(env.xquat[world, wrists]),
        env.wrist_pos_ref[frame],
        env.wrist_quat_ref[frame],
    )
    objects = scene["object_body"]
    return {
        "frame": np.asarray(frame),
        "qpos": env.qpos[world].copy(),
        "object_pose": np.concatenate(
            [env.xpos[world, objects], _wxyz_to_xyzw(env.xquat[world, objects])],
            axis=-1,
        ),
        "reward": reward[world].numpy(),
        "object_pos_err": pos[world],
        "object_rot_err": rot[world],
        "wrist_pos_err": wrist_pos,
        "wrist_rot_err": wrist_rot,
    }
