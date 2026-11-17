# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Sampling MPC directly on FlashCHORD's environment (GPU).

Nothing is ported: the worlds are a ``flash_chord.envs.base.BaseEnv`` (Newton
``SolverMuJoCo`` physics, FlashCHORD's control law and reference handling), and
SPIDER's sampler drives it through the usual simulator interface.
:mod:`spider.simulators.cpu` runs the same scene on the CPU without
FlashCHORD.

Conventions:

- One SPIDER simulation step is one FlashCHORD control frame (``config.sim_dt``
  is the environment's ``frame_dt``; the physics substeps run inside the frame).
- The control is FlashCHORD's residual hand-pose action in physical units: per
  hand (action side order), 3 wrist position residuals [m], 3 wrist
  roll/pitch/yaw residuals [rad] and the finger joint residuals [rad], before
  the environment's EMA filter and clip. Zero control replays the reference.
- The environment owns the reference clock, so reference slices passed by the
  optimizer are ignored.
- The reward is SPIDER's tracking cost plus ``CONTACT_SCALE`` times CHORD's
  contact terms.
- What a run records is in the compiled MuJoCo model's coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import loguru
import numpy as np
import torch
import warp as wp

from spider.config import Config
from spider.simulators.chord_export import colour_mjcf, hold_hinge, scene_colours
from spider.simulators.cpu import fingertips

# FlashCHORD experiment the scene, embodiment and action space come from.
EXPERIMENT = "sharpa_flash_sac"
# Hand gains. FlashCHORD's own (wrists 25 N/m, fingers 1.7 N m/rad) are tuned
# for learning with a virtual wrench carrying the object; hands this stiff
# carry it themselves and go where the plan sends them. The wrist damping is
# twice what keeps FlashCHORD's damping ratio, and the action filter keeps 0.5
# of the previous action instead of 0.3: both steady the hand base.
WRIST_KP, WRIST_KD = 1600.0, 84.8528  # N/m, N s/m
WRIST_ROT_KP, WRIST_ROT_KD = 100.0, 6.0  # N m/rad, N m s/rad
FINGER_KP, FINGER_KD = 40.0, 0.6  # N m/rad, N m s/rad
ACTION_EMA = 0.5
# Solver room per world; FlashCHORD's training sizes (512 / 256) overflow on
# mesh-heavy objects, and rows that do not fit are dropped.
NJMAX, NCONMAX = 1024, 512
# Armature of the dofs of kinematic bodies (the support table), see
# set_kinematic_armature.
KINEMATIC_ARMATURE = 1.0e4
# Object hinges get this many times the dry friction that holds the part they
# carry against gravity (FlashCHORD: 0.1 N m for every hinge).
HINGE_HOLD = 1.5
# Weight of CHORD's contact terms in the reward. Its keypoint and joint
# tracking terms and its penalties are left out: SPIDER's cost does the tracking.
CONTACT_SCALE = 0.005
_TRACKING_TERMS = ("object_keypoints", "hand_keypoints", "hand_joint_pos")
_FAILURE_TERMS = (
    "wrist_position",
    "wrist_orientation",
    "object_position",
    "object_orientation",
)
_LOST = 10.0  # object error [m] reported for a world whose simulation diverged


@dataclass
class TrackIndex:
    """Coordinates and weights of SPIDER's tracking cost in Newton ``joint_q``."""

    scalar_q: torch.Tensor  # (S,) scalar coordinates
    scalar_w: torch.Tensor  # (S,)
    quat_q: torch.Tensor  # (Q, 4) xyzw quaternion coordinates
    quat_w: torch.Tensor  # (Q,)
    vel_q: torch.Tensor  # (V,) scalar coordinates with a finite-difference velocity
    vel_dof: torch.Tensor  # (V,) their joint_qd indices


@dataclass
class ChordEnv:
    """FlashCHORD environment plus the torch views SPIDER reads and writes."""

    fc: Any  # flash_chord.envs.base.BaseEnv
    cfg: Any  # composed FlashCHORD config
    setup: Any  # flash_chord SceneSetup
    mj_model: Any  # the compiled single-world mujoco.MjModel
    device: str
    num_worlds: int
    num_frames: int
    frame_dt: float
    nq: int
    nv: int
    nu: int
    sides: tuple[str, ...]  # action side order
    action_scale: torch.Tensor  # (nu,) physical residual per unit of raw action
    action_torch: torch.Tensor  # (N * nu,) raw action buffer shared with action_wp
    action_wp: wp.array
    state: dict[str, torch.Tensor]  # persistent per-world state, (N, k) views
    terms: torch.Tensor  # (N, K) CHORD objective terms of the last transition
    weights: torch.Tensor  # (K,) their weights in the reward
    errors: dict[str, torch.Tensor]  # FlashCHORD tracking errors of the last transition
    frame: torch.Tensor  # (N,) reference frame of the last transition
    qpos_ref: torch.Tensor  # (T, nq) reference joint_q
    qvel_ref: torch.Tensor  # (T, V) finite-difference reference of track.vel_dof
    track: TrackIndex
    object_body_ids: torch.Tensor  # (B,) per-world body ids of the object bodies
    wrist_body_ids: torch.Tensor  # (H,) per-world body ids of the wrist frames
    trace_body_ids: torch.Tensor  # the object root, then the wrists
    tip_body_ids: torch.Tensor  # (F,) last link of every finger, traced too
    tip_local: torch.Tensor  # (F, 3) the fingertip in that link's frame
    kinematic_dofs: tuple[int, ...]  # MuJoCo dofs of kinematic bodies
    mj_qpos: torch.Tensor  # (N, nq_mj) state in the MuJoCo model's coordinates
    mj_qpos_ref: np.ndarray | None = None  # (T, nq_mj) reference
    trace_ref: np.ndarray | None = None  # (T, K, 3) reference of the trace points
    object_pose_reset: np.ndarray | None = None  # (B, 7) object bodies at frame 0


# --
# Setup
# --


def compose_chord_config(config: Config):
    """Compose FlashCHORD's training config for sampling MPC.

    As FlashCHORD's evaluation: no virtual object control, no failure
    terminations; plus this module's gains and solver sizes.
    """
    from flash_chord.configuration import compose_config
    from hydra.core.global_hydra import GlobalHydra

    if not config.data_path:
        raise ValueError("data_path must point to a FlashCHORD reference parquet")
    parquet = Path(config.data_path).expanduser().resolve()
    wrists = []
    for side, y in (("left", -0.25), ("right", 0.25)):
        for axis, rest in (("x", 0.0), ("y", y), ("z", 0.5)):
            wrists.append(
                f"{side}_wrist_{axis}:{{kp:{WRIST_KP},kd:{WRIST_KD},"
                f"effort_limit:45.0,default_pos:{rest}}}"
            )
        wrists.append(
            f"{side}_wrist_rotvec:{{mode:effort,kp:{WRIST_ROT_KP},kd:{WRIST_ROT_KD},"
            "effort_limit:45.0}"
        )
    overrides = [
        f"experiment={EXPERIMENT}",
        f"task.parquet='{parquet}'",
        # The whole reference at the speed it was recorded.
        "task.motion_speed=1.0",
        "task.motion_start_frame=0",
        "task.motion_end_frame=-1",
        f"scene.world_count={int(config.num_samples)}",
        "action.input_mode=raw",
        "env.voc_scale=0.0",
        "env.voc.max_force=0.0",
        "env.voc.max_torque=0.0",
        f"sim.njmax={NJMAX}",
        f"sim.nconmax={NCONMAX}",
        *[f"termination.{name}.enabled=false" for name in _FAILURE_TERMS],
        "+embodiment.config.joint_overrides={" + ",".join(wrists) + "}",
        f"action.wrist_kp={WRIST_ROT_KP}",
        "+embodiment.config.default_joint={_target_:"
        f"flash_chord.embodiments.base.JointControlSpec,kp:{FINGER_KP},kd:{FINGER_KD}}}",
        f"action.ema={ACTION_EMA}",
    ]
    # SPIDER runners are Hydra applications; FlashCHORD composes its own config.
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    return compose_config("train_flash_sac", overrides)


def _view(array: wp.array, num_worlds: int) -> torch.Tensor:
    """Zero-copy (N, k) torch view of a per-world Warp array."""
    return wp.to_torch(array).view(num_worlds, -1)


def _state_views(fc, num_worlds: int) -> dict[str, torch.Tensor]:
    """Everything a FlashCHORD transition carries from one frame to the next."""
    state, action = fc.state_0, fc.action.buffers
    arrays = {
        "joint_q": state.joint_q,
        "joint_qd": state.joint_qd,
        "body_q": state.body_q,
        "body_qd": state.body_qd,
        "timestep": fc.timestep,
        "episode_step": fc.episode_step,
        "episode_start_frame": fc.episode_start_frame,
        "terminated": fc.terminated,
        "truncated": fc.truncated,
        "raw_action": action.raw_action,
        "filtered_action": action.filtered_action,
        "processed_target": action.processed_target,
        "joint_target": action.joint_target,
        "action_l2": action.action_l2,
        "action_rate_l2": action.action_rate_l2,
        # MuJoCo's constraint solver starts from the previous acceleration.
        "qacc_warmstart": fc.solver.mjw_data.qacc_warmstart,
    }
    return {name: _view(array, num_worlds) for name, array in arrays.items()}


def hands_in_action_order(fc) -> list:
    """Hand layouts in the action's side order, the order of every per-hand array."""
    by_side = {hand.side: hand for hand in fc.scene.layout.hands}
    return [by_side[side] for side in fc.action.sides]


def mujoco_qpos_address(fc, q_ids) -> np.ndarray:
    """MuJoCo ``qpos`` addresses of Newton ``joint_q`` coordinates of one world.

    Scalar and position coordinates map one to one. A quaternion keeps its four
    addresses, but Newton stores xyzw where MuJoCo stores wxyz.
    """
    joints = fc.model.joint_count // fc.world_count
    newton_start = fc.model.joint_q_start.numpy()[:joints]
    mujoco_start = fc.solver.mj_q_start.numpy()[:joints]
    q_ids = np.atleast_1d(np.asarray(q_ids, dtype=np.int64))
    joint = np.searchsorted(newton_start, q_ids, side="right") - 1
    return mujoco_start[joint] + q_ids - newton_start[joint]


def _track_index(config: Config, fc) -> TrackIndex:
    """SPIDER's weighted qpos tracking, expressed on the FlashCHORD scene layout."""
    device = config.device
    scalar_q: list[int] = []
    scalar_w: list[float] = []
    quat_q: list[int] = []
    quat_w: list[float] = []
    vel_q: list[int] = []
    vel_dof: list[int] = []
    for hand in hands_in_action_order(fc):
        scalar_q += list(hand.wrist_pos_q_ids)
        scalar_w += [config.base_pos_rew_scale] * len(hand.wrist_pos_q_ids)
        quat_q.append(int(hand.wrist_orient_q_id))
        quat_w.append(config.base_rot_rew_scale)
        scalar_q += list(hand.finger_q_ids)
        scalar_w += [config.joint_rew_scale] * len(hand.finger_q_ids)
        vel_q += list(hand.wrist_pos_q_ids) + list(hand.finger_q_ids)
        vel_dof += list(hand.wrist_pos_dof_ids) + list(hand.finger_dof_ids)
    for binding in fc.scene.objects:
        root = binding.root.free_q_ids
        scalar_q += list(root[:3])
        scalar_w += [config.pos_rew_scale] * 3
        quat_q.append(int(root[3]))
        quat_w.append(config.rot_rew_scale)
        for joint in binding.articulations:
            scalar_q.append(int(joint.q_id))
            scalar_w.append(config.rot_rew_scale)
            vel_q.append(int(joint.q_id))
            vel_dof.append(int(joint.dof_id))
    long = lambda v: torch.as_tensor(v, dtype=torch.long, device=device)  # noqa: E731
    real = lambda v: torch.as_tensor(v, dtype=torch.float32, device=device)  # noqa: E731
    return TrackIndex(
        scalar_q=long(scalar_q),
        scalar_w=real(scalar_w),
        quat_q=long(quat_q)[:, None] + torch.arange(4, device=device)[None, :],
        quat_w=real(quat_w),
        vel_q=long(vel_q),
        vel_dof=long(vel_dof),
    )


def kinematic_dofs(fc) -> tuple[int, ...]:
    """MuJoCo dofs of kinematic bodies (Newton compiles them with a huge armature)."""
    armature = wp.to_torch(fc.solver.mjw_model.dof_armature)
    threshold = 0.5 * float(fc.solver._KINEMATIC_ARMATURE)
    armature = armature.reshape(-1, armature.shape[-1])
    return tuple(int(i) for i in torch.nonzero(armature[0] >= threshold).flatten())


def set_kinematic_armature(fc, dofs: tuple[int, ...], armature: float) -> None:
    """Give kinematic-body dofs a finite armature and rescale the solver statistics.

    Newton's ``SolverMuJoCo`` realises a kinematic body (FlashCHORD's support
    table) as joint dofs with armature 1e10. Besides the conditioning, that
    inflates ``stat.meaninertia``, by which MuJoCo scales its solver tolerance,
    so the constraint solver stops far from convergence. Kinematic coordinates
    are re-imposed from the Newton state every substep, so any armature much
    larger than the object mass keeps the body in place.
    """
    import mujoco

    if not dofs:
        return
    model = fc.solver.mj_model
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    dense = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, dense, data.qM)
    keep = np.ones(model.nv, dtype=bool)
    keep[list(dofs)] = False
    wp.to_torch(fc.solver.mjw_model.dof_armature)[..., list(dofs)] = float(armature)
    # Mean mass-matrix diagonal over the dofs that can move.
    wp.to_torch(fc.solver.mjw_model.stat.meaninertia)[...] = float(
        np.diag(dense)[keep].mean()
    )


def set_hinge_hold(fc, margin: float) -> None:
    """Give object hinges enough dry friction to hold their own lid against gravity.

    A recorded lid stays where the demonstrator left it, but a simulated one
    with less friction than its own weight needs swings open or shut as soon
    as the object is tilted, whatever the hands do.
    """
    solver = fc.solver
    newton_dof = solver.mjc_dof_to_newton_dof.numpy()[0]
    friction = wp.to_torch(solver.mjw_model.dof_frictionloss)
    for binding in fc.scene.objects:
        for joint in binding.articulations:
            dof = int(np.nonzero(newton_dof == int(joint.dof_id))[0][0])
            hold = hold_hinge(solver.mj_model, dof, margin)
            if hold is not None:
                friction[..., dof] = hold


def check_articulation_range(fc, qpos_ref: torch.Tensor) -> None:
    """Warn about object joints whose reference leaves the simulated joint range.

    A reference outside the range starts or drives the simulation into a
    violated joint limit, which throws the object off its reference (an asset
    whose hinge limit is tighter than the recorded motion, for example).
    """
    lower = wp.to_torch(fc.model.joint_limit_lower)
    upper = wp.to_torch(fc.model.joint_limit_upper)
    for binding in fc.scene.objects:
        for joint in binding.articulations:
            low, high = float(lower[joint.dof_id]), float(upper[joint.dof_id])
            ref = qpos_ref[:, int(joint.q_id)]
            if float(ref.min()) < low - 1e-3 or float(ref.max()) > high + 1e-3:
                loguru.logger.warning(
                    "object joint coordinate {}: the reference spans [{:.3f}, {:.3f}] "
                    "but the joint range is [{:.3f}, {:.3f}]; widen the limit in the "
                    "object's asset",
                    int(joint.q_id),
                    float(ref.min()),
                    float(ref.max()),
                    low,
                    high,
                )


def write_mjcf(fc, path: str | Path) -> None:
    """Have Newton compile the scene once more and save the MJCF it generates."""
    import newton

    sim = fc.config.sim
    newton.solvers.SolverMuJoCo(
        fc.model,
        solver=sim.solver,
        integrator=sim.integrator,
        njmax=sim.njmax,
        nconmax=sim.nconmax,
        impratio=sim.impratio,
        cone=sim.cone,
        iterations=sim.iterations,
        ls_iterations=sim.ls_iterations,
        use_mujoco_contacts=sim.use_mujoco_contacts,
        save_to_mjcf=str(path),
    )


def setup_env(config: Config, ref_data: tuple[torch.Tensor, ...] | None = None):
    """Build ``config.num_samples`` FlashCHORD worlds at reference frame 0.

    ``config.data_path`` is the FlashCHORD reference parquet (a file, or a
    directory holding exactly one).
    """
    del ref_data  # the reference lives in the environment
    from flash_chord.configuration import instantiate_typed
    from flash_chord.embodiments.base import Embodiment
    from flash_chord.envs.base import BaseEnv
    from flash_chord.envs.rl import RLEnvConfig
    from flash_chord.lifecycle.curriculum import FixedCurriculum
    from flash_chord.scene.collision import CollisionPolicy
    from flash_chord.scene.setup import setup_scene

    cfg = compose_chord_config(config)
    env_config = instantiate_typed(cfg.env, RLEnvConfig)
    num_worlds = int(config.num_samples)
    device = config.device

    wp.set_device(device)
    setup = setup_scene(
        parquet=cfg.task.parquet,
        control_fps=env_config.sim.fps,
        motion_speed=float(cfg.task.motion_speed),
        embodiment=instantiate_typed(cfg.embodiment, Embodiment),
        collision=instantiate_typed(cfg.collision, CollisionPolicy),
        world_count=num_worlds,
        include_support=bool(cfg.scene.support),
        decompose_objects=bool(cfg.scene.decompose_objects),
        object_scale_min=float(cfg.scene.get("object_scale_min", 1.0)),
        object_scale_max=float(cfg.scene.get("object_scale_max", 1.0)),
        object_scale_seed=int(cfg.scene.get("object_scale_seed", 0)),
        contact_friction=float(cfg.scene.get("contact_friction", 1.0)),
        object_free_joint_damping=float(
            cfg.scene.get("object_free_joint_damping", 0.0)
        ),
        source_frame_playback=bool(cfg.task.get("source_frame_playback", False)),
        motion_start_frame=int(cfg.task.get("motion_start_frame", 0)),
        motion_end_frame=int(cfg.task.get("motion_end_frame", -1)),
    )
    fc = BaseEnv(setup.scene, setup.reference, config=env_config, device=device)

    # The reward's CHORD part: the positive contact terms of the last training stage.
    curriculum = instantiate_typed(cfg.curriculum, FixedCurriculum)
    weights = {
        name: 0.0 if name in _TRACKING_TERMS else max(float(value), 0.0)
        for name, value in curriculum.stages[-1].objective_weights.items()
    }
    fc.objective.set_weights(weights)

    fixed_dofs = kinematic_dofs(fc)
    set_kinematic_armature(fc, fixed_dofs, KINEMATIC_ARMATURE)
    set_hinge_hold(fc, HINGE_HOLD)

    fc.reset(frame_id=0)
    if str(device).startswith("cuda"):
        fc.capture_transition()  # CUDA graph; the CPU device steps eagerly
    wp.synchronize_device(device)

    model = setup.scene.model
    nu = int(fc.action.action_dim)
    action_torch = torch.zeros(num_worlds * nu, dtype=torch.float32, device=device)
    track = _track_index(config, fc)
    qpos_ref = wp.to_torch(fc.reset_table.ref_joint_q).clone()
    check_articulation_range(fc, qpos_ref)
    nxt = torch.cat([qpos_ref[1:], qpos_ref[-1:]], dim=0)
    qvel_ref = (nxt[:, track.vel_q] - qpos_ref[:, track.vel_q]) / fc.frame_dt
    buffers = fc.termination.buffers
    long = lambda v: torch.as_tensor(v, dtype=torch.long, device=device)  # noqa: E731
    object_body_ids = long(fc.command.layout.body_ids)
    wrist_body_ids = long([hand.wrist_body_id for hand in hands_in_action_order(fc)])
    # Fingertips, for the viewer's traces: found in the MuJoCo model as the CPU
    # simulator finds them, then named as Newton bodies.
    to_newton = fc.solver.mjc_body_to_newton.numpy()[0]
    to_mujoco = {int(body): i for i, body in enumerate(to_newton) if body >= 0}
    tip_mujoco, tip_local = fingertips(
        fc.solver.mj_model, [to_mujoco[int(body)] for body in wrist_body_ids]
    )
    env = ChordEnv(
        fc=fc,
        cfg=cfg,
        setup=setup,
        mj_model=fc.solver.mj_model,
        device=device,
        num_worlds=num_worlds,
        num_frames=int(setup.reference.num_frames),
        frame_dt=float(fc.frame_dt),
        nq=model.joint_coord_count // num_worlds,
        nv=model.joint_dof_count // num_worlds,
        nu=nu,
        sides=tuple(fc.action.sides),
        action_scale=wp.to_torch(fc.action.buffers.scale).clone(),
        action_torch=action_torch,
        action_wp=wp.from_torch(action_torch),
        state=_state_views(fc, num_worlds),
        terms=_view(fc.objective.terms, num_worlds),
        weights=wp.to_torch(fc.objective.weights),
        errors={
            "object_pos_err": _view(buffers.object_pos_err, num_worlds),
            "object_rot_err": _view(buffers.object_ori_err, num_worlds),
            "wrist_pos_err": _view(buffers.wrist_pos_err, num_worlds),
            "wrist_rot_err": _view(buffers.wrist_ori_err, num_worlds),
        },
        frame=torch.zeros(num_worlds, dtype=torch.long, device=device),
        qpos_ref=qpos_ref,
        qvel_ref=qvel_ref,
        track=track,
        object_body_ids=object_body_ids,
        wrist_body_ids=wrist_body_ids,
        trace_body_ids=torch.cat([object_body_ids[:1], wrist_body_ids]),
        tip_body_ids=long(to_newton[tip_mujoco]),
        tip_local=torch.as_tensor(tip_local, dtype=torch.float32, device=device),
        kinematic_dofs=fixed_dofs,
        mj_qpos=wp.to_torch(fc.solver.mjw_data.qpos),
    )
    _read_reference(config, env)
    # For the viewer and the video only: colours, and the MJCF file the
    # streaming viewers build their scene from.
    env.mj_model.geom_rgba[:] = scene_colours(env.mj_model)
    if config.show_viewer and any(v in config.viewer for v in ("viser", "rerun")):
        path = Path(config.output_dir) / "scene_chord.xml"
        write_mjcf(fc, path)
        path.write_text(colour_mjcf(path.read_text()))
        config.model_path = str(path)
    loguru.logger.info(
        "chord: {} worlds, {} frames at {:.0f} Hz x {} substeps, nq {} nv {} nu {}",
        num_worlds,
        env.num_frames,
        1.0 / env.frame_dt,
        env_config.sim.substeps,
        env.nq,
        env.nv,
        nu,
    )
    return env


def _read_reference(config: Config, env: ChordEnv) -> None:
    """The reference in MuJoCo coordinates, by putting the worlds on every frame."""
    solver = env.fc.solver
    qpos, trace = [], []
    for frame in range(env.num_frames):
        env.fc.reset(frame_id=frame)
        # Newton's own state -> MuJoCo coordinate conversion.
        solver._update_mjc_data(solver.mjw_data, env.fc.model, env.fc.state_0)
        wp.synchronize_device(env.device)
        qpos.append(env.mj_qpos[0].cpu().numpy().copy())
        trace.append(get_trace(config, env)[0].cpu().numpy())
    env.mj_qpos_ref, env.trace_ref = np.stack(qpos), np.stack(trace)
    reset_env(config, env, 0)
    # Simulated object-body frames at reference frame 0. A reference may place a
    # body's frame elsewhere on the body than the simulated link frame does.
    env.object_pose_reset = get_body_poses(env, env.object_body_ids)[0].cpu().numpy()


def _sync_torch(device: str) -> None:
    """Wait for queued torch work; torch and Warp use different CUDA streams."""
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def reset_env(config: Config, env: ChordEnv, frame_id: int = 0) -> ChordEnv:
    """Put every world on reference frame ``frame_id`` at rest."""
    del config
    _sync_torch(env.device)
    env.fc.reset(frame_id=frame_id)
    wp.synchronize_device(env.device)
    env.state["qacc_warmstart"].zero_()
    env.frame.fill_(frame_id)
    return env


# --
# Stepping and state
# --


def step_env(config: Config, env: ChordEnv, ctrl: torch.Tensor) -> None:
    """Advance one control frame with residual controls of shape (N, nu) or (nu,)."""
    del config
    if ctrl.dim() == 1:
        ctrl = ctrl.unsqueeze(0).expand(env.num_worlds, -1)
    timestep = env.state["timestep"][:, 0].long()
    env.frame.copy_(timestep.clamp(0, env.num_frames - 1))
    env.action_torch.view(env.num_worlds, env.nu).copy_(ctrl / env.action_scale)
    _sync_torch(env.device)
    env.fc.step(env.action_wp)
    wp.synchronize_device(env.device)


def save_state(env: ChordEnv) -> dict[str, torch.Tensor]:
    """Copy the state of every world."""
    return {name: view.clone() for name, view in env.state.items()}


def load_state(env: ChordEnv, state: dict[str, torch.Tensor]) -> ChordEnv:
    """Restore a state saved by :func:`save_state`."""
    for name, view in env.state.items():
        view.copy_(state[name])
    return env


def sync_env(config: Config, env: ChordEnv, mj_data=None) -> ChordEnv:
    """Broadcast the first world (the executed one) to every world, as MJWP does."""
    del config, mj_data
    for view in env.state.values():
        view.copy_(view[:1].clone().expand_as(view))
    return env


def copy_sample_state(
    config: Config, env: ChordEnv, src_indices: torch.Tensor, dst_indices: torch.Tensor
) -> None:
    """Copy the state of worlds ``src_indices`` onto worlds ``dst_indices``."""
    del config
    for view in env.state.values():
        view[dst_indices] = view[src_indices]


def save_env_params(config: Config, env: ChordEnv) -> dict:
    """No per-rollout parameters."""
    del config, env
    return {}


def load_env_params(config: Config, env: ChordEnv, env_param: dict) -> ChordEnv:
    """No per-rollout parameters."""
    del config, env_param
    return env


# --
# Rewards
# --


def _quat_error_xyzw(quat: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Rotation vector of ``quat * ref^-1`` for xyzw quaternions (..., 4)."""
    v1, w1 = quat[..., :3], quat[..., 3:]
    v2, w2 = ref[..., :3], ref[..., 3:]
    vec = w2 * v1 - w1 * v2 - torch.linalg.cross(v1, v2, dim=-1)
    scalar = w1 * w2 + (v1 * v2).sum(dim=-1, keepdim=True)
    vec = torch.where(scalar < 0.0, -vec, vec)
    norm = vec.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(norm, scalar.abs())
    return vec * (angle / norm.clamp_min(1e-8))


def spider_cost(config: Config, env: ChordEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """SPIDER's weighted pose distance and joint-velocity distance per world.

    The velocity distance covers the robot's scalar joints and object
    articulations; ball and free joints have no finite-difference reference.
    """
    del config
    track = env.track
    qpos = env.state["joint_q"]
    ref = env.qpos_ref[env.frame]
    scalar = (qpos[:, track.scalar_q] - ref[:, track.scalar_q]) * track.scalar_w
    rot = _quat_error_xyzw(qpos[:, track.quat_q], ref[:, track.quat_q])
    rot = rot * track.quat_w[None, :, None]
    qpos_dist = torch.cat([scalar, rot.flatten(1)], dim=1).norm(dim=1)
    qvel = env.state["joint_qd"][:, track.vel_dof]
    qvel_dist = (qvel - env.qvel_ref[env.frame]).norm(dim=1)
    return qpos_dist, qvel_dist


def get_reward(
    config: Config, env: ChordEnv, ref: tuple[torch.Tensor, ...] | None = None
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Per-world reward of the last transition."""
    del ref
    qpos_dist, qvel_dist = spider_cost(config, env)
    contact = (env.terms * env.weights).sum(dim=1)
    reward = -qpos_dist - config.vel_rew_scale * qvel_dist + CONTACT_SCALE * contact
    info = {
        "qpos_dist": qpos_dist,
        # A world whose simulation diverged reports a large error, not NaN, so
        # that the statistics the optimizer keeps over the samples stay usable.
        "object_pos_err": torch.nan_to_num(
            env.errors["object_pos_err"].max(dim=1).values, nan=_LOST, posinf=_LOST
        ),
        "object_rot_err": torch.nan_to_num(
            env.errors["object_rot_err"].max(dim=1).values, nan=np.pi, posinf=np.pi
        ),
    }
    return reward, info


def get_terminal_reward(
    config: Config, env: ChordEnv, ref: tuple[torch.Tensor, ...] | None = None
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Reward of the last rollout step."""
    reward, info = get_reward(config, env, ref)
    return config.terminal_rew_scale * reward, info


def get_terminate(
    config: Config, env: ChordEnv, ref: tuple[torch.Tensor, ...] | None = None
) -> torch.Tensor:
    """FlashCHORD's failure flags (its failure terminations are switched off)."""
    del config, ref
    return env.state["terminated"][:, 0] > 0


# --
# Readouts
# --


def get_qpos(config: Config, env: ChordEnv) -> torch.Tensor:
    """Newton ``joint_q`` of every world, (N, nq)."""
    del config
    return env.state["joint_q"]


def get_qvel(config: Config, env: ChordEnv) -> torch.Tensor:
    """Newton ``joint_qd`` of every world, (N, nv)."""
    del config
    return env.state["joint_qd"]


def get_body_poses(env: ChordEnv, body_ids: torch.Tensor) -> torch.Tensor:
    """World poses (N, len(body_ids), 7) as position + xyzw quaternion."""
    body_q = env.state["body_q"].view(env.num_worlds, -1, 7)
    return body_q[:, body_ids]


def get_trace(config: Config, env: ChordEnv) -> torch.Tensor:
    """Positions of the object root, the wrists and the fingertips, (N, K, 3)."""
    del config
    bodies = get_body_poses(env, env.trace_body_ids)[..., :3]
    links = get_body_poses(env, env.tip_body_ids)
    # Rotate each tip by its link's xyzw quaternion.
    axis, tip = links[..., 3:6], env.tip_local.expand_as(links[..., :3])
    twice = 2.0 * torch.linalg.cross(axis, tip)
    tips = (
        links[..., :3] + tip + links[..., 6:] * twice + torch.linalg.cross(axis, twice)
    )
    return torch.cat([bodies, tips], dim=1)


def reference_object_poses(env: ChordEnv) -> np.ndarray:
    """Reference pose of every object body, (T, B, 7) as position + xyzw quaternion."""
    reference = env.setup.reference
    pos = np.asarray(reference.object_body_pos_w(), dtype=np.float64)
    wxyz = np.asarray(reference.object_body_quat_w(), dtype=np.float64)
    return np.concatenate([pos, wxyz[..., 1:], wxyz[..., :1]], axis=-1)


def run_arrays(env: ChordEnv) -> dict[str, np.ndarray]:
    """What a saved trajectory needs besides its frames: the reference and the layout."""
    fc, hands = env.fc, hands_in_action_order(env.fc)
    address = lambda q_ids: mujoco_qpos_address(fc, q_ids)  # noqa: E731
    return {
        "qpos_ref": env.mj_qpos_ref,
        "object_pose_ref": reference_object_poses(env),
        "object_pose_reset": env.object_pose_reset,
        "articulation_q_ids": address(
            np.asarray(fc.command.layout.articulation_q_ids, dtype=np.int64)
        ),
        "finger_q_ids": np.concatenate([address(h.finger_q_ids) for h in hands]),
        "wrist_quat_q_ids": np.concatenate(
            [address(h.wrist_orient_q_id) for h in hands]
        ),
        "body_names": np.array(env.setup.reference.object_body_names()),
        "sides": np.array(env.sides),
        "frame_dt": np.asarray(env.frame_dt),
    }


def read_step(config: Config, env: ChordEnv, world: int = 0) -> dict[str, np.ndarray]:
    """Record of the last transition in one world."""
    reward, _ = get_reward(config, env)
    num = lambda t: t.detach().cpu().numpy()  # noqa: E731
    record = {
        "frame": num(env.frame[world]),
        "qpos": num(env.mj_qpos[world]),
        "object_pose": num(get_body_poses(env, env.object_body_ids)[world]),
        "reward": num(reward[world]),
    }
    for name, value in env.errors.items():
        record[name] = num(value[world])
    return record
