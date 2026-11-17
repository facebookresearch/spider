# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests of the Viser viewer's scene (a stand-in for the server, no browser).

Run with: pytest spider/viewers/viser_viewer_test.py
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import mujoco
import numpy as np
import pytest
import trimesh

from spider.viewers import viser_viewer

# An irregular tetrahedron far from its own origin: its compiled copy is moved to
# its centre of mass and turned into its principal axes.
VERTICES = np.array(
    [[0.30, 0.10, 0.20], [0.50, 0.12, 0.22], [0.33, 0.40, 0.18], [0.31, 0.15, 0.45]]
)
FACES = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]])
SCALE = np.array([2.0, 1.0, 0.5])
GEOM_POS = np.array([0.05, -0.02, 0.10])
GEOM_QUAT = np.array([0.9, 0.1, 0.3, 0.2])
FROMTO = np.array([0.1, 0.2, 0.3, 0.4, 0.1, 0.5])

MJCF = """
<mujoco>
  <asset>
    <mesh name="wedge" {source} scale="{scale}"/>
  </asset>
  <worldbody>
    <body name="table" pos="0 0 0.5">
      <geom name="top" type="box" size="0.4 0.4 0.01"/>
    </body>
    <body name="object" pos="0.2 0.1 0.6" quat="0.8 0.2 0.1 0.5">
      <freejoint/>
      <geom name="wedge" type="mesh" mesh="wedge" pos="{pos}" quat="{quat}"/>
      <body name="lid" pos="0 0 0.1">
        <joint name="hinge" axis="0 1 0"/>
        <geom name="bar" type="capsule" size="0.02" fromto="{fromto}"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _text(values: np.ndarray) -> str:
    return " ".join(repr(float(v)) for v in np.asarray(values).ravel())


def _placed(vertices: np.ndarray, position, wxyz) -> np.ndarray:
    """Vertices of a drawn mesh in the frame of its body."""
    mat = np.zeros(9)
    quat = np.asarray(wxyz, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, quat / np.linalg.norm(quat))
    return np.asarray(vertices) @ mat.reshape(3, 3).T + np.asarray(position)


class _Slider:
    """A slider like Viser's: setting a new value calls the update callbacks."""

    def __init__(self, value: int):
        self._value, self.max, self.callbacks = value, 1, []

    @property
    def value(self) -> int:
        return self._value

    @value.setter
    def value(self, value: int) -> None:
        if value != self._value:
            self._value = value
            for callback in self.callbacks:
                callback(None)

    def on_update(self, callback):
        self.callbacks.append(callback)
        return callback


def _scene(monkeypatch, tmp_path, mesh_from_file: bool):
    """Build the scene on a stand-in server; the model, the handles, what was drawn."""
    if mesh_from_file:
        trimesh.Trimesh(VERTICES, FACES, process=False).export(tmp_path / "wedge.stl")
        source = 'file="wedge.stl"'
    else:
        faces = " ".join(str(i) for i in FACES.ravel())
        source = f'vertex="{_text(VERTICES)}" face="{faces}"'
    xml = tmp_path / "scene.xml"
    xml.write_text(
        MJCF.format(
            source=source,
            scale=_text(SCALE),
            pos=_text(GEOM_POS),
            quat=_text(GEOM_QUAT),
            fromto=_text(FROMTO),
        )
    )
    drawn: dict[str, np.ndarray] = {}

    def add_mesh(name, mesh, position=(0, 0, 0), wxyz=(1, 0, 0, 0), **_):
        drawn[name] = _placed(mesh.vertices, position, wxyz)
        return MagicMock()

    server = MagicMock()
    server.scene.add_mesh_trimesh.side_effect = add_mesh
    server.scene.add_frame.side_effect = lambda name, **_: SimpleNamespace(
        name=name, position=None, wxyz=None
    )
    server.gui.add_slider.side_effect = lambda label, initial_value=0, **_: _Slider(
        initial_value
    )
    state = viser_viewer._ViserState(server=server)
    monkeypatch.setattr(viser_viewer, "_STATE", state)
    monkeypatch.setattr(viser_viewer, "threading", MagicMock())  # no playback thread
    _, model, handles = viser_viewer.build_and_log_scene(xml)
    return model, handles, drawn


@pytest.mark.parametrize("mesh_from_file", [False, True])
def test_meshes_are_drawn_where_the_model_has_them(
    monkeypatch, tmp_path, mesh_from_file
):
    """A mesh given inline or as a file lands on its vertices, scaled once."""
    _, _, drawn = _scene(monkeypatch, tmp_path, mesh_from_file)
    expected = _placed(VERTICES * SCALE, GEOM_POS, GEOM_QUAT)
    for root in ("mujoco", "mujoco_ref"):
        vertices = drawn[f"{root}/object/geom_wedge"]
        gaps = np.linalg.norm(vertices[:, None] - expected[None], axis=-1)
        assert len(vertices) == len(expected)
        assert gaps.min(axis=0).max() < 1e-5 and gaps.min(axis=1).max() < 1e-5


def test_a_capsule_lies_on_its_segment(monkeypatch, tmp_path):
    """A capsule given by its end points is drawn between them."""
    _, _, drawn = _scene(monkeypatch, tmp_path, mesh_from_file=False)
    vertices = drawn["mujoco/lid/geom_bar"]
    start, end = FROMTO[:3], FROMTO[3:]
    along = (end - start) / np.linalg.norm(end - start)
    reach = (vertices - start) @ along
    off_axis = np.linalg.norm(vertices - start - reach[:, None] * along, axis=1)
    assert off_axis.max() < 0.02 + 1e-5
    assert np.isclose(reach.min(), -0.02, atol=1e-3)
    assert np.isclose(reach.max(), np.linalg.norm(end - start) + 0.02, atol=1e-3)


def test_bodies_fixed_to_the_world_have_no_ghost(monkeypatch, tmp_path):
    """The reference ghost covers what can move, not the static scene."""
    _, _, drawn = _scene(monkeypatch, tmp_path, mesh_from_file=False)
    assert "mujoco/table/geom_top" in drawn
    assert "mujoco_ref/table/geom_top" not in drawn
    assert "mujoco_ref/lid/geom_bar" in drawn


def test_the_scene_follows_the_run_until_the_slider_is_moved_back(
    monkeypatch, tmp_path
):
    """Logged frames are shown as they come; going back in time stops that."""
    model, handles, _ = _scene(monkeypatch, tmp_path, mesh_from_file=False)
    data, data_ref = mujoco.MjData(model), mujoco.MjData(model)
    mujoco.mj_kinematics(model, data_ref)
    object_handle, object_id = next(
        (handle, body) for handle, body in handles if handle.name == "mujoco/object"
    )
    ghost = next(
        handle
        for handle, _ in viser_viewer._STATE.ref_body_handles
        if handle.name == "mujoco_ref/object"
    )
    shown = []
    for frame in range(4):
        if frame == 3:  # the user goes back to the first frame
            viser_viewer._STATE.playback_slider.value = 0
        data.qpos[0] = 0.2 + 0.1 * frame
        mujoco.mj_kinematics(model, data)
        viser_viewer.log_frame(data, 0.0, handles, data_ref=data_ref)
        shown.append(float(object_handle.position[0]))
        assert np.allclose(ghost.position, data_ref.xpos[object_id])
    assert np.allclose(shown, [0.2, 0.3, 0.4, 0.2])
    assert viser_viewer._STATE.playback_slider.max == 3


def test_rewind_keeps_the_frames_before_a_return(monkeypatch, tmp_path):
    """A run that goes back drops the frames and traces after the return."""
    model, handles, _ = _scene(monkeypatch, tmp_path, mesh_from_file=False)
    data = mujoco.MjData(model)
    object_handle = next(h for h, _ in handles if h.name == "mujoco/object")
    for frame in range(6):
        data.qpos[0] = 0.1 * frame
        mujoco.mj_kinematics(model, data)
        viser_viewer.log_frame(data, 0.0, handles, data_ref=data)
        traces = np.zeros((1, 1, 2, 2, 3))  # (iterations, samples, horizon, points, 3)
        viser_viewer._STATE.trace_history[frame] = (frame, traces, None, None, None)
    state = viser_viewer._STATE
    viser_viewer.rewind(3)
    assert len(state.frame_history) == 3 and sorted(state.trace_history) == [0, 1, 2]
    assert state.playback_slider.value == 2 and state.playback_slider.max == 2
    assert np.isclose(object_handle.position[0], 0.2)  # the last frame kept
    # The run goes on from there and the scene follows it again.
    data.qpos[0] = 0.9
    mujoco.mj_kinematics(model, data)
    viser_viewer.log_frame(data, 0.0, handles, data_ref=data)
    assert state.playback_slider.value == 3 and np.isclose(
        object_handle.position[0], 0.9
    )


def test_reset_scene_forgets_the_run(monkeypatch, tmp_path):
    """Another run starts from an empty timeline."""
    model, handles, _ = _scene(monkeypatch, tmp_path, mesh_from_file=False)
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    for _ in range(3):
        viser_viewer.log_frame(data, 0.0, handles, data_ref=data)
    viser_viewer.reset_scene()
    state = viser_viewer._STATE
    assert not state.frame_history and not state.body_handles
    assert state.playback_slider.value == 0
