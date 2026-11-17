# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Export a FlashCHORD scene for the CPU workflow (:mod:`spider.simulators.cpu`).

FlashCHORD compiles its scene through Newton into a MuJoCo model and drives it
with a control law of its own. The export writes a directory that plain MuJoCo
can run without Newton, Warp or FlashCHORD:

- ``scene.xml``: the model, with the support table fixed to the world, object
  hinges passive and held by friction, and the control timestep's substep.
- ``scene.npz``: the reference and the tables of the control law, in that
  model's coordinates.

The gains, the action filter and every other FlashCHORD setting are those of
:mod:`spider.simulators.chord` at the time of the export.

Example:
    python -m spider.simulators.chord_export --parquet REFERENCE_DIR \
        --out example_datasets/processed/chord/sharpa/bimanual/TASK/DATA_ID
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import loguru
import mujoco
import numpy as np

# Coordinate groups of SPIDER's tracking cost (see spider.simulators.cpu).
GROUP_WRIST_POS, GROUP_FINGER, GROUP_OBJECT_POS, GROUP_ARTICULATION = 0, 1, 2, 3
GROUP_WRIST_ROT, GROUP_OBJECT_ROT = 0, 1


def hold_hinge(model: mujoco.MjModel, dof: int, margin: float) -> float | None:
    """Raise a hinge's dry friction to ``margin`` times its largest gravity torque.

    The torque is the weight of everything the hinge carries times the distance
    of its centre of mass from the hinge axis (at the model's default pose).
    Friction is only ever raised. Returns the hinge's friction afterwards, or
    None if the dof does not belong to a hinge.
    """
    joint = int(model.dof_jntid[dof])
    if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
        return None
    body = int(model.jnt_bodyid[joint])
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    arm = data.subtree_com[body] - data.xanchor[joint]
    axis = data.xaxis[joint]
    lever = float(np.linalg.norm(arm - axis * (arm @ axis)))
    weight = float(model.body_subtreemass[body]) * float(
        np.linalg.norm(model.opt.gravity)
    )
    hold = max(float(model.dof_frictionloss[dof]), margin * weight * lever)
    model.dof_frictionloss[dof] = hold
    return hold


def scene_colours(model: mujoco.MjModel) -> np.ndarray:
    """Geom colours by what each body is, (ngeom, 4): rendering only."""
    colours = np.tile([0.95, 0.6, 0.2, 1.0], (model.ngeom, 1))  # the object
    for geom in range(model.ngeom):
        body = model.geom_bodyid[geom]
        name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or "").lower()
        if "support" in name:
            colours[geom] = (0.45, 0.35, 0.28, 1.0)
        elif "left" in name:
            colours[geom] = (0.45, 0.58, 0.88, 1.0)
        elif "right" in name:
            colours[geom] = (0.78, 0.78, 0.80, 1.0)
        elif model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_PLANE:
            colours[geom] = (0.9, 0.9, 0.9, 1.0)
    return colours


def colour_mjcf(xml: str) -> str:
    """Write :func:`scene_colours` into an MJCF whose geoms are named."""
    model = mujoco.MjModel.from_xml_string(xml)
    colours = scene_colours(model)
    root = ElementTree.fromstring(xml)
    for element in root.iter("geom"):
        name = element.get("name", "")
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom >= 0:
            element.set("rgba", " ".join(f"{c:g}" for c in colours[geom]))
    return ElementTree.tostring(root, encoding="unicode")


def _address_maps(
    old: mujoco.MjModel, new: mujoco.MjModel
) -> tuple[np.ndarray, np.ndarray]:
    """Old -> new ``qpos`` and dof addresses by joint name (-1 for removed joints)."""
    qpos_width = {0: 7, 1: 4, 2: 1, 3: 1}
    dof_width = {0: 6, 1: 3, 2: 1, 3: 1}
    qpos_map = np.full(old.nq, -1, dtype=np.int64)
    dof_map = np.full(old.nv, -1, dtype=np.int64)
    for joint in range(old.njnt):
        name = mujoco.mj_id2name(old, mujoco.mjtObj.mjOBJ_JOINT, joint)
        kept = mujoco.mj_name2id(new, mujoco.mjtObj.mjOBJ_JOINT, name)
        if kept < 0:
            continue
        kind = int(old.jnt_type[joint])
        for k in range(qpos_width[kind]):
            qpos_map[old.jnt_qposadr[joint] + k] = new.jnt_qposadr[kept] + k
        for k in range(dof_width[kind]):
            dof_map[old.jnt_dofadr[joint] + k] = new.jnt_dofadr[kept] + k
    return qpos_map, dof_map


def finalize_scene(
    xml: str, tables: dict[str, np.ndarray], hinge_hold: float
) -> tuple[str, dict[str, np.ndarray]]:
    """Turn Newton's MJCF and FlashCHORD's tables into a scene plain MuJoCo runs.

    Needs MuJoCo only. The joints of kinematic bodies (the support table) are
    removed, which fixes those bodies to the world; the drives Newton compiles
    onto object hinges, which FlashCHORD scales to zero, are switched off; the
    hinges get ``hinge_hold`` times the friction that holds their part against
    gravity; and the timestep becomes one physics substep. The tables are
    re-addressed to the resulting model.

    Args:
        xml: MJCF written by Newton's ``SolverMuJoCo``.
        tables: The reference and control tables in that model's addresses
            (``kinematic_joints``, ``actuator_dof``, ``frame_dt``, ...).
        hinge_hold: Margin of the object hinges' friction, see :func:`hold_hinge`.

    Returns:
        The final MJCF and the tables ``scene.npz`` holds.
    """
    exported = mujoco.MjModel.from_xml_string(xml)
    root = ElementTree.fromstring(xml)
    kinematic = {str(name) for name in tables["kinematic_joints"]}
    for parent in root.iter():
        for child in list(parent):
            if child.tag in ("joint", "freejoint") and child.get("name") in kinematic:
                parent.remove(child)
    model = mujoco.MjModel.from_xml_string(
        ElementTree.tostring(root, encoding="unicode")
    )
    qpos_map, dof_map = _address_maps(exported, model)

    scene = dict(tables)
    for name in ("scalar_q", "quat_q", "vel_q", "articulation_q", "wrist_qpos"):
        scene[name] = qpos_map[scene[name]]
    for name in ("vel_dof", "wrist_pos_dof", "wrist_ori_dof", "finger_dof"):
        scene[name] = dof_map[scene[name]]
    # Position actuator -> the dof whose target it tracks; velocity actuators hold zero.
    actuator_dof = scene.pop("actuator_dof")
    scene["position_actuator"] = np.nonzero(actuator_dof >= 0)[0]
    scene["position_actuator_dof"] = dof_map[actuator_dof[actuator_dof >= 0]]
    kept_qpos = np.nonzero(qpos_map >= 0)[0]
    scene["qpos_ref"] = scene["qpos_ref"][:, kept_qpos][
        :, np.argsort(qpos_map[kept_qpos])
    ]
    kept_dof = np.nonzero(dof_map >= 0)[0]
    targets = np.zeros((scene["targets0"].shape[0], model.nv))
    targets[:, dof_map[kept_dof]] = scene["targets0"][:, kept_dof]
    scene["targets0"] = targets
    del scene["kinematic_joints"]

    # Object hinges: no drive, and friction that holds the lid.
    actuators = list(root.iter("general"))
    if len(actuators) != model.nu:
        raise NotImplementedError("expected one <general> element per actuator")
    for address in scene["articulation_q"]:
        joint = int(np.nonzero(model.jnt_qposadr == int(address))[0][0])
        for actuator in np.nonzero(model.actuator_trnid[:, 0] == joint)[0]:
            actuators[actuator].set("gainprm", "0")
            actuators[actuator].set("biasprm", "0 0 0")
        hold = hold_hinge(model, int(model.jnt_dofadr[joint]), hinge_hold)
        if hold is not None:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            element = next(e for e in root.iter("joint") if e.get("name") == name)
            element.set("frictionloss", repr(hold))
    timestep = float(scene["frame_dt"]) / int(scene["substeps"])
    root.find("option").set("timestep", repr(timestep))
    return colour_mjcf(ElementTree.tostring(root, encoding="unicode")), scene


def export_scene(parquet: str, out_dir: str | Path, device: str = "cuda:0") -> Path:
    """Build the FlashCHORD scene with one world and write ``scene.xml`` / ``scene.npz``.

    Needs FlashCHORD (see :mod:`spider.simulators.chord`).
    """
    import tempfile

    import warp as wp

    from spider.config import Config
    from spider.simulators import chord

    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    config = Config(simulator="chord", data_path=parquet, device=device, num_samples=1)
    config.show_viewer = False
    env = chord.setup_env(config)
    fc, solver = env.fc, env.fc.solver
    model = solver.mj_model
    dof_map = solver.mjc_dof_to_newton_dof.numpy()[0]
    if not np.array_equal(dof_map, np.arange(model.nv)):
        raise NotImplementedError("MuJoCo and Newton order the dofs differently")
    address = lambda q_ids: chord.mujoco_qpos_address(fc, q_ids)  # noqa: E731

    # Residual action layout, in the action's side order.
    hand_index = {hand.side: i for i, hand in enumerate(fc.scene.layout.hands)}
    hands = chord.hands_in_action_order(fc)
    order = [hand_index[side] for side in env.sides]

    # SPIDER tracking cost, in MuJoCo coordinates.
    scalar_q: list[int] = []
    scalar_group: list[int] = []
    quat_q: list[int] = []
    quat_group: list[int] = []
    vel_q: list[int] = []
    vel_dof: list[int] = []
    for hand in hands:
        scalar_q += list(address(hand.wrist_pos_q_ids))
        scalar_group += [GROUP_WRIST_POS] * len(hand.wrist_pos_q_ids)
        quat_q.append(int(address(hand.wrist_orient_q_id)[0]))
        quat_group.append(GROUP_WRIST_ROT)
        scalar_q += list(address(hand.finger_q_ids))
        scalar_group += [GROUP_FINGER] * len(hand.finger_q_ids)
        vel_q += list(address(hand.wrist_pos_q_ids)) + list(address(hand.finger_q_ids))
        vel_dof += list(hand.wrist_pos_dof_ids) + list(hand.finger_dof_ids)
    articulation_q: list[int] = []
    for binding in fc.scene.objects:
        root = binding.root.free_q_ids
        scalar_q += list(address(root[:3]))
        scalar_group += [GROUP_OBJECT_POS] * 3
        quat_q.append(int(address(root[3])[0]))
        quat_group.append(GROUP_OBJECT_ROT)
        for joint in binding.articulations:
            q = int(address(joint.q_id)[0])
            scalar_q.append(q)
            scalar_group.append(GROUP_ARTICULATION)
            articulation_q.append(q)
            vel_q.append(q)
            vel_dof.append(int(joint.dof_id))

    # Joint targets a zero action commands at every reference frame, (T, ndof).
    ndof = fc.num_joint_dof
    zero = wp.zeros(env.num_worlds * env.nu, dtype=wp.float32, device=env.device)
    fc.action.reset(fc.reset_table.all_reset_mask)
    targets = np.zeros((env.num_frames, ndof))
    for frame in range(env.num_frames):
        fc.timestep.assign(np.full(env.num_worlds, frame, dtype=np.int32))
        fc.action.process(zero, fc.timestep, fc.state_0)
        targets[frame] = fc.action.joint_target.numpy()[:ndof]
    chord.reset_env(config, env, 0)

    body_to_newton = solver.mjc_body_to_newton.numpy()[0]
    mujoco_body = {int(b): i for i, b in enumerate(body_to_newton) if b >= 0}
    action = fc.action.buffers
    tables = {
        "frame_dt": env.frame_dt,
        "substeps": int(fc.config.sim.substeps),
        "sides": np.array(env.sides),
        "body_names": np.array(env.setup.reference.object_body_names()),
        "qpos_ref": env.mj_qpos_ref,
        "targets0": targets,
        "actuator_dof": solver.mjc_actuator_to_newton_idx.numpy(),
        "wrist_pos_dof": np.array([hand.wrist_pos_dof_ids for hand in hands]),
        "wrist_ori_dof": np.array([hand.wrist_orient_dof_ids for hand in hands]),
        "finger_dof": np.array([hand.finger_dof_ids for hand in hands]),
        "wrist_qpos": np.array([address(h.wrist_orient_q_id)[0] for h in hands]),
        "wrist_kp": fc.action.wrist.kp.numpy()[order],
        "wrist_tau_max": fc.action.wrist.tau_max.numpy()[order],
        "action_clip": action.clip.numpy(),
        "action_ema": float(action.ema),
        "object_pose_ref": chord.reference_object_poses(env),
        "object_pose_reset": env.object_pose_reset,
        "object_body": np.array(
            [mujoco_body[int(b)] for b in fc.command.layout.body_ids]
        ),
        "wrist_body": np.array([mujoco_body[int(h.wrist_body_id)] for h in hands]),
        "scalar_q": np.array(scalar_q),
        "scalar_group": np.array(scalar_group),
        "quat_q": np.array(quat_q),
        "quat_group": np.array(quat_group),
        "vel_q": np.array(vel_q),
        "vel_dof": np.array(vel_dof),
        "articulation_q": np.array(articulation_q, dtype=np.int64),
        "kinematic_joints": np.array(
            sorted(
                {
                    mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_JOINT, model.dof_jntid[d]
                    )
                    for d in env.kinematic_dofs
                }
            )
        ),
    }
    with tempfile.TemporaryDirectory() as tmp:
        chord.write_mjcf(fc, Path(tmp) / "newton.xml")
        xml = (Path(tmp) / "newton.xml").read_text()
    xml, scene = finalize_scene(xml, tables, chord.HINGE_HOLD)
    (out_dir / "scene.xml").write_text(xml)
    np.savez(out_dir / "scene.npz", **scene)
    loguru.logger.info("Exported {} frames to {}", env.num_frames, out_dir)
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--parquet", required=True, help="FlashCHORD reference")
    parser.add_argument("--out", required=True, help="output scene directory")
    parser.add_argument("--device", default="cuda:0", help="Warp device to build on")
    args = parser.parse_args()
    export_scene(args.parquet, args.out, args.device)
