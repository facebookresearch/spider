# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests of the export of FlashCHORD scenes for the CPU workflow (no FlashCHORD needed).

:func:`spider.simulators.chord_export.finalize_scene` turns Newton's MJCF and
FlashCHORD's tables into a scene of the CPU workflow. A small synthetic scene
stands in for Newton's (a floating one-finger hand, a free box with a lid and a
kinematic support); what the export makes of it is the scene of ``cpu_test.py``::

    pytest spider/simulators/chord_export_test.py
"""

import mujoco
import numpy as np
import pytest
import torch

from spider.simulators import cpu
from spider.simulators.chord_export import finalize_scene
from spider.simulators.cpu_test import (
    NUM_FRAMES,
    SCENE_XML,
    _config,
    _plans,
    _rollout,
    scene_tables,
)

# The scene as Newton writes it: every actuator a <general>, a position and a
# velocity one per driven joint, the lid's hinge included.
MJCF = """
<mujoco>
  <option gravity="0 0 -9.81"/>
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
        <joint name="lid_hinge" type="hinge" axis="0 1 0" frictionloss="0.001"
               range="-2 0" armature="0.001"/>
        <geom type="box" pos="0.03 0 0.003" size="0.03 0.03 0.003" mass="0.05"/>
      </body>
    </body>
    <body name="support" pos="0 0 0.02">
      <joint name="support_free_joint" type="free"/>
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
    <general joint="lid_hinge" biastype="affine" gainprm="5" biasprm="0 -5"/>
    <general joint="lid_hinge" biastype="affine" gainprm="1" biasprm="0 0 -1"/>
  </actuator>
</mujoco>
"""


def _support_first(xml: str) -> str:
    """The same scene with the kinematic support declared before everything else."""
    support = xml[xml.index('    <body name="support"') : xml.index("  </worldbody>")]
    return xml.replace(support, "").replace(
        "  <worldbody>\n", "  <worldbody>\n" + support
    )


def _raw_scene(support_first: bool = False) -> tuple[str, dict]:
    """Newton's MJCF and FlashCHORD's tables, in the unwelded model's addresses.

    With ``support_first`` the kinematic joint owns the first seven coordinates
    and six dofs, so every other address is shifted in the unwelded model.
    """
    xml = _support_first(MJCF) if support_first else MJCF
    q, d = (7, 6) if support_first else (0, 0)  # address shifts
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    qpos_ref = np.tile(data.qpos, (NUM_FRAMES, 1))
    qpos_ref[:, q + 2] = np.linspace(0.0, 0.02, NUM_FRAMES)  # the hand rises 2 cm
    targets = np.zeros((NUM_FRAMES, model.nv))
    targets[:, d + 2] = qpos_ref[:, q + 2]
    bodies = [model.body("box").id, model.body("lid").id]
    pose = np.concatenate([data.xpos[bodies], data.xquat[bodies][:, [1, 2, 3, 0]]], 1)
    tables = {
        "frame_dt": 0.05,
        "substeps": 5,
        "sides": np.array(["right"]),
        "body_names": np.array(["box", "lid"]),
        "targets0": targets,
        # A position actuator names its dof; velocity actuators are negative.
        "actuator_dof": np.array([d, -2, d + 1, -3, d + 2, -4, d + 6, -8, d + 13, -15]),
        "wrist_pos_dof": np.array([[0, 1, 2]]) + d,
        "wrist_ori_dof": np.array([[3, 4, 5]]) + d,
        "finger_dof": np.array([[6]]) + d,
        "wrist_qpos": np.array([3]) + q,
        "wrist_kp": np.array([25.0]),
        "wrist_tau_max": np.array([45.0]),
        "action_clip": np.array([0.2] * 3 + [1.0] * 4),
        "action_ema": 0.3,
        "qpos_ref": qpos_ref,
        "object_pose_ref": np.tile(pose, (NUM_FRAMES, 1, 1)),
        "object_pose_reset": pose,
        "object_body": np.array(bodies),
        "wrist_body": np.array([model.body("hand").id]),
        "scalar_q": np.array([0, 1, 2, 7, 8, 9, 10, 15]) + q,
        "scalar_group": np.array([0, 0, 0, 1, 2, 2, 2, 3]),
        "quat_q": np.array([3, 11]) + q,
        "quat_group": np.array([0, 1]),
        "vel_q": np.array([0, 1, 2, 7, 15]) + q,
        "vel_dof": np.array([0, 1, 2, 6, 13]) + d,
        "articulation_q": np.array([15]) + q,
        "kinematic_joints": np.array(["support_free_joint"]),
    }
    return xml, tables


def _env(directory, support_first: bool = False):
    """CPU worlds of the exported raw scene."""
    pytest.importorskip("mujoco.batch")
    directory.mkdir(exist_ok=True)
    xml, scene = finalize_scene(*_raw_scene(support_first), hinge_hold=1.5)
    (directory / "scene.xml").write_text(xml)
    np.savez(directory / "scene.npz", **scene)
    config = _config(directory)
    ref_data = cpu.load_data(config, config.data_path)
    return config, cpu.setup_env(config, ref_data, num_threads=1)


def test_export_fixes_the_support_and_frees_the_lid():
    """The exported model has no support joint, a passive lid and one substep."""
    raw_xml, tables = _raw_scene()
    raw = mujoco.MjModel.from_xml_string(raw_xml)
    xml, scene = finalize_scene(raw_xml, tables, hinge_hold=1.5)
    model = mujoco.MjModel.from_xml_string(xml)
    assert (model.nq, model.nv, model.nu) == (raw.nq - 7, raw.nv - 6, raw.nu)
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "support_free_joint") < 0
    assert model.opt.timestep == pytest.approx(0.01)
    # The lid's drive is off; every other actuator is as Newton wrote it.
    assert np.all(model.actuator_gainprm[8:, 0] == 0.0)
    assert np.all(model.actuator_biasprm[8:, :3] == 0.0)
    assert np.array_equal(model.actuator_gainprm[:8], raw.actuator_gainprm[:8])
    # The lid weighs 0.05 kg, its centre of mass 3 cm out and 3 mm up from the hinge.
    lid = model.joint("lid_hinge").dofadr[0]
    lever = np.hypot(0.03, 0.003)
    assert model.dof_frictionloss[lid] == pytest.approx(1.5 * 0.05 * 9.81 * lever)
    assert "kinematic_joints" not in scene and "actuator_dof" not in scene
    assert list(scene["position_actuator"]) == [0, 2, 4, 6, 8]
    assert list(scene["position_actuator_dof"]) == [0, 1, 2, 6, 13]
    assert scene["qpos_ref"].shape == (NUM_FRAMES, model.nq)


def test_kinematic_joint_position_does_not_change_the_scene(tmp_path):
    """Addresses are remapped by joint name, wherever the fixed joint sits."""
    config, last = _env(tmp_path / "last")
    _, first = _env(tmp_path / "first", support_first=True)
    assert (first.nq, first.nv, first.nu) == (last.nq, last.nv, last.nu)
    for name in ("scalar_q", "quat_q", "vel_q", "vel_dof", "wrist_qpos", "finger_dof"):
        assert np.array_equal(first.scene[name], last.scene[name]), name
    assert np.array_equal(first.mj_qpos_ref, last.mj_qpos_ref)
    plans = _plans(last)
    np.testing.assert_allclose(
        _rollout(config, first, plans).numpy(),
        _rollout(config, last, plans).numpy(),
        atol=1e-9,
    )


def test_the_export_writes_the_scene_of_the_cpu_workflow():
    """The exported raw scene is the scene ``cpu_test.py`` writes by hand."""
    xml, scene = finalize_scene(*_raw_scene(), hinge_hold=1.5)
    exported = mujoco.MjModel.from_xml_string(xml)
    written = mujoco.MjModel.from_xml_string(SCENE_XML)
    for field in ("nq", "nv", "nu", "nbody", "ngeom"):
        assert getattr(exported, field) == getattr(written, field), field
    for field in (
        "actuator_gainprm",
        "actuator_biasprm",
        "dof_frictionloss",
        "dof_armature",
        "dof_damping",
        "body_pos",
        "body_mass",
        "jnt_range",
    ):
        np.testing.assert_allclose(
            getattr(exported, field), getattr(written, field), err_msg=field
        )
    assert exported.opt.timestep == written.opt.timestep
    tables = scene_tables()
    assert sorted(scene) == sorted(tables)
    for name, value in tables.items():
        np.testing.assert_array_equal(scene[name], value, err_msg=name)


def test_rollouts_of_the_exported_scene_match_the_written_one(tmp_path):
    """Same model, same tables: the same physics, the colours aside."""
    pytest.importorskip("mujoco.batch")
    config, exported = _env(tmp_path / "exported")
    (tmp_path / "written").mkdir()
    (tmp_path / "written" / "scene.xml").write_text(SCENE_XML)
    np.savez(tmp_path / "written" / "scene.npz", **scene_tables())
    config_written = _config(tmp_path / "written")
    written = cpu.setup_env(
        config_written,
        cpu.load_data(config_written, config_written.data_path),
        num_threads=1,
    )
    plans = _plans(exported)
    assert torch.equal(
        _rollout(config, exported, plans), _rollout(config_written, written, plans)
    )
