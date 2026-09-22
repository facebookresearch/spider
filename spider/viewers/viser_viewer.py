# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""MuJoCo XML Viser visualizer.

This mirrors the Rerun viewer APIs so it can be used as a drop-in replacement.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import loguru
import mujoco
import numpy as np
import trimesh

# -----------------------------
# Trace visualization defaults
# -----------------------------

DEFAULT_TRACE_COLOR = [204, 26, 204]  # ~ (0.8, 0.1, 0.8)
DEFAULT_TRACE_RADIUS = 0.002
DEFAULT_FLOOR_COLOR = [200, 200, 200]  # light grey


def _lazy_import_viser():
    try:
        import viser  # type: ignore

        return viser
    except ImportError as exc:
        raise ImportError(
            "viser is required for the Viser viewer. Install with `pip install viser`."
        ) from exc


@dataclass
class _ViserState:
    server: Any | None = None
    entity_root: str = "mujoco"
    body_handles: list[tuple[Any, int]] = field(default_factory=list)
    ref_body_handles: list[tuple[Any, int]] = field(default_factory=list)
    ref_geom_handles: list[Any] = field(default_factory=list)
    visual_geom_handles: list[Any] = field(default_factory=list)
    collision_geom_handles: list[Any] = field(default_factory=list)
    scene_checkboxes: dict[str, Any] = field(default_factory=dict)
    trace_handle: Any | None = None
    trace_handles: dict[str, Any] = field(default_factory=dict)
    trace_colors: np.ndarray | None = None
    trace_slider: Any | None = None
    trace_checkboxes: dict[str, Any] = field(default_factory=dict)
    last_traces: np.ndarray | None = None
    last_trace_ref: np.ndarray | None = None

    # Frame-change callbacks
    frame_change_callbacks: list[Callable[[int], None]] = field(default_factory=list)

    # Timeline
    frame_history: list[dict[int, tuple[np.ndarray, np.ndarray]]] = field(
        default_factory=list
    )
    trace_history: dict[
        int,
        tuple[int, np.ndarray, np.ndarray | None, np.ndarray | None, int | None],
    ] = field(default_factory=dict)
    trace_id_counter: int = 0
    playback_slider: Any | None = None
    playback_fps_slider: Any | None = None
    playback_speed: int = 0
    playback_thread: Any | None = None

    # Optimization-process recording (timeline of all transform updates).
    recording_serializer: Any | None = None
    recording_dt: float = 0.02


_STATE = _ViserState()


def start_recording(dt: float = 0.02) -> None:
    """Begin recording every frame update into a serializable timeline.

    After this call, every viser scene message (transform updates from
    log_frame, etc.) is also routed to a StateSerializer. Call
    `end_recording_and_save(path)` after the optimization loop to write
    the full timeline to a .viser file.

    Args:
        dt: Wall-clock spacing (seconds) inserted between frames in the
            recording. Use the simulator's render_dt or sim_dt.
    """
    if _STATE.server is None:
        return
    if _STATE.recording_serializer is not None:
        # Already recording; nothing to do.
        return
    _STATE.recording_serializer = _STATE.server.get_scene_serializer()
    _STATE.recording_dt = float(dt)


def tick_recording() -> None:
    """Advance the recording timeline by one dt.

    Call this once per logged frame in the optimization loop. Any frame
    updates emitted between calls are stored at the current time.
    """
    if _STATE.recording_serializer is None:
        return
    _STATE.recording_serializer.insert_sleep(_STATE.recording_dt)


def end_recording_and_save(path: str) -> bool:
    """Serialize the recorded timeline to disk. Returns True on success."""
    if _STATE.recording_serializer is None:
        return False
    try:
        from pathlib import Path

        data = _STATE.recording_serializer.serialize()
        Path(path).write_bytes(data)
        return True
    except Exception:
        return False
    finally:
        _STATE.recording_serializer = None


# -----------------------------
# Geometry helpers
# -----------------------------


def _rgba_to_uint8(rgba: np.ndarray) -> np.ndarray:
    rgba_arr = np.asarray(rgba)
    if np.issubdtype(rgba_arr.dtype, np.floating):
        rgba_arr = np.clip(rgba_arr, 0.0, 1.0)
        rgba_arr = (rgba_arr * 255.0).astype(np.uint8)
    else:
        rgba_arr = rgba_arr.astype(np.uint8)
    if rgba_arr.size == 3:
        rgba_arr = np.concatenate([rgba_arr, np.array([255], dtype=np.uint8)])
    return rgba_arr


def _set_mesh_color(mesh: trimesh.Trimesh, rgba: np.ndarray) -> None:
    from trimesh.visual import TextureVisuals
    from trimesh.visual.material import PBRMaterial

    rgba_int = _rgba_to_uint8(rgba)
    mesh.visual = TextureVisuals(
        material=PBRMaterial(
            baseColorFactor=rgba_int,
            main_color=rgba_int,
            metallicFactor=0.5,
            roughnessFactor=1.0,
            alphaMode="BLEND" if rgba_int[-1] < 255 else "OPAQUE",
        )
    )


def _trimesh_from_primitive(
    geom_type: int, size: np.ndarray, rgba: np.ndarray | None = None
) -> trimesh.Trimesh | None:
    t = mujoco.mjtGeom
    if geom_type == t.mjGEOM_SPHERE:
        mesh = trimesh.creation.icosphere(radius=float(size[0]), subdivisions=2)
    elif geom_type == t.mjGEOM_CAPSULE:
        radius = float(size[0])
        length = float(2.0 * size[1])
        mesh = trimesh.creation.capsule(radius=radius, height=length)
    elif geom_type == t.mjGEOM_CYLINDER:
        radius = float(size[0])
        height = float(2.0 * size[1])
        mesh = trimesh.creation.cylinder(radius=radius, height=height)
    elif geom_type == t.mjGEOM_BOX:
        extents = 2.0 * np.asarray(size[:3], dtype=np.float32)
        mesh = trimesh.creation.box(extents=extents)
    elif geom_type == t.mjGEOM_PLANE:
        mesh = trimesh.creation.box(extents=[20.0, 20.0, 0.01])
    else:
        return None

    if rgba is not None:
        _set_mesh_color(mesh, rgba)
    return mesh


def _mujoco_mesh_to_trimesh(
    model: mujoco.MjModel, geom_id: int
) -> trimesh.Trimesh | None:
    mesh_id = model.geom_dataid[geom_id]
    if mesh_id < 0:
        return None

    vert_start = int(model.mesh_vertadr[mesh_id])
    vert_count = int(model.mesh_vertnum[mesh_id])
    face_start = int(model.mesh_faceadr[mesh_id])
    face_count = int(model.mesh_facenum[mesh_id])

    vertices = model.mesh_vert[vert_start : vert_start + vert_count]
    faces = model.mesh_face[face_start : face_start + face_count]

    if len(vertices) == 0 or len(faces) == 0:
        return None

    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _get_mesh_file(spec: mujoco.MjSpec, geom: mujoco.MjsGeom) -> Path | None:
    try:
        meshname = geom.meshname
        if not meshname:
            return None
        mesh = spec.mesh(meshname)
        mesh_dir = spec.meshdir if spec.meshdir is not None else ""
        model_dir = spec.modelfiledir if spec.modelfiledir is not None else ""
        return (Path(model_dir) / mesh_dir / mesh.file).resolve()
    except Exception:
        return None


def _get_mesh_scale(spec: mujoco.MjSpec, geom: mujoco.MjsGeom) -> np.ndarray | None:
    try:
        mesh = spec.mesh(geom.meshname)
        scale = mesh.scale
        if scale is None:
            return None
        return np.asarray(scale, dtype=np.float32)
    except Exception:
        return None


def _ensure_names(spec: mujoco.MjSpec) -> None:
    geom_placeholder_idx = 0
    body_placeholder_idx = 0
    for body in spec.bodies[1:]:
        if not body.name:
            body.name = f"VISER_BODY_{body_placeholder_idx}"
            body_placeholder_idx += 1
        for geom in body.geoms:
            if not geom.name:
                geom.name = f"VISER_GEOM_{geom_placeholder_idx}"
                geom_placeholder_idx += 1


# -----------------------------
# Scene construction
# -----------------------------


def init_viser(
    app_name: str = "spider",
    spawn: bool | None = None,
    host: str = "0.0.0.0",
    port: int = 8080,
) -> Any:
    """Initialize Viser server (spawn unused, kept for drop-in compatibility)."""
    if _STATE.server is not None:
        return _STATE.server
    viser = _lazy_import_viser()
    _STATE.server = viser.ViserServer(label=app_name, host=host, port=port)
    return _STATE.server


def reset_scene() -> None:
    """Clear geometry before loading another saved trial in the inspector."""
    _get_server().scene.reset()
    _STATE.body_handles.clear()
    _STATE.ref_body_handles.clear()
    _STATE.ref_geom_handles.clear()
    _STATE.visual_geom_handles.clear()
    _STATE.collision_geom_handles.clear()
    _STATE.frame_history.clear()
    _STATE.trace_history.clear()


def update_frame(
    data: mujoco.MjData,
    body_handles: list[tuple[Any, int]],
    data_ref: mujoco.MjData | None = None,
) -> None:
    """Display one state without recording an optimization playback history."""
    with _get_server().atomic():
        for handle, body_id in body_handles:
            handle.position = tuple(data.xpos[body_id])
            handle.wxyz = tuple(data.xquat[body_id])
        if data_ref is not None:
            for handle, body_id in _STATE.ref_body_handles:
                handle.position = tuple(data_ref.xpos[body_id])
                handle.wxyz = tuple(data_ref.xquat[body_id])


def _get_server() -> Any:
    if _STATE.server is None:
        init_viser()
    return _STATE.server


def register_frame_callback(callback: Callable[[int], None]) -> None:
    """Register a callback that fires whenever the timeline frame changes."""
    _STATE.frame_change_callbacks.append(callback)


def build_and_log_scene_from_spec(
    spec: mujoco.MjSpec,
    model: mujoco.MjModel,
    xml_path: Path | None = None,
    entity_root: str = "mujoco",
    build_ref: bool = True,
) -> list[tuple[Any, int]]:
    """Build and log a Viser scene directly from a spec and compiled model."""
    _ensure_names(spec)
    server = _get_server()
    _STATE.entity_root = entity_root

    if "visual_meshes" not in _STATE.scene_checkboxes:
        folder = server.gui.add_folder("Meshes")
        with folder:
            _STATE.scene_checkboxes["visual_meshes"] = server.gui.add_checkbox(
                "Visual", initial_value=True
            )
            _STATE.scene_checkboxes["collision_meshes"] = server.gui.add_checkbox(
                "Collision", initial_value=True
            )

            @_STATE.scene_checkboxes["visual_meshes"].on_update
            def _(_) -> None:
                val = _STATE.scene_checkboxes["visual_meshes"].value
                for h in _STATE.visual_geom_handles:
                    h.visible = val

            @_STATE.scene_checkboxes["collision_meshes"].on_update
            def _(_) -> None:
                val = _STATE.scene_checkboxes["collision_meshes"].value
                for h in _STATE.collision_geom_handles:
                    h.visible = val

            _STATE.scene_checkboxes["ref"] = server.gui.add_checkbox(
                "Reference", initial_value=True
            )

            @_STATE.scene_checkboxes["ref"].on_update
            def _(_) -> None:
                val = _STATE.scene_checkboxes["ref"].value
                for h in _STATE.ref_geom_handles:
                    h.visible = val

    # Add a floor grid.
    try:
        server.scene.add_grid(
            f"{entity_root}/ground_plane",
            section_color=tuple(np.array(DEFAULT_FLOOR_COLOR) / 255.0),
            cell_color=tuple(np.array(DEFAULT_FLOOR_COLOR) / 255.0),
        )
    except Exception:
        pass

    body_entity_and_ids: list[tuple[Any, int]] = []

    for body in spec.bodies[1:]:
        body_name = body.name
        body_path = f"{entity_root}/{body_name}"
        body_handle = server.scene.add_frame(body_path, show_axes=False)

        try:
            body_id = model.body(body_name).id
        except Exception:
            body_id = body.id

        body_entity_and_ids.append((body_handle, body_id))

        for geom in body.geoms:
            geom_name = geom.name

            # Skip placeholder mass geoms (dummy spheres with no visual purpose).
            if "_object_mass" in geom_name:
                continue

            # Skip geoms in hidden groups (group >= 5 are never rendered).
            try:
                gv = (
                    int(np.asarray(geom.group).ravel()[0])
                    if hasattr(geom, "group")
                    else 0
                )
                if gv >= 5:
                    continue
            except Exception:
                pass

            geom_path = f"{body_path}/geom_{geom_name}"

            model_geom = None
            try:
                model_geom = model.geom(geom.name)
            except Exception:
                model_geom = None

            rgba = None
            if model_geom is not None:
                try:
                    rgba = np.asarray(model_geom.rgba, dtype=np.float32)
                except Exception:
                    rgba = None
            if rgba is None:
                try:
                    rgba = np.asarray(geom.rgba, dtype=np.float32)
                except Exception:
                    rgba = None

            if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
                tm = None
                mesh_file = _get_mesh_file(spec, geom)
                mesh_scale = _get_mesh_scale(spec, geom)
                if mesh_file is not None and mesh_file.exists():
                    try:
                        tm = trimesh.load(str(mesh_file), force="mesh")
                        if isinstance(tm, trimesh.Scene):
                            tm = tm.to_mesh()
                    except Exception:
                        tm = None
                if tm is None:
                    try:
                        geom_id = model_geom.id if model_geom is not None else -1
                        tm = _mujoco_mesh_to_trimesh(model, geom_id)
                    except Exception:
                        tm = None
                if tm is None:
                    loguru.logger.warning(
                        f"Viser: failed to load mesh for geom '{geom_name}'"
                    )
                    continue
                if mesh_scale is not None:
                    try:
                        tm.apply_scale(mesh_scale)
                    except Exception:
                        pass
                if rgba is not None:
                    _set_mesh_color(tm, rgba)
            else:
                size = geom.size
                if model_geom is not None:
                    try:
                        model_size = model.geom_size[model_geom.id]
                        if np.any(np.asarray(size) == 0) or np.any(np.isnan(size)):
                            size = model_size
                    except Exception:
                        pass
                tm = _trimesh_from_primitive(geom.type, size, rgba=rgba)

            if tm is None:
                continue

            # For primitive (non-mesh) geoms, use compiled model transforms
            # which correctly resolve fromto-specified capsules. For mesh geoms,
            # keep the spec transforms since the STL vertices are in the original
            # frame (the compiler re-centers mesh vertices at their COM which
            # doesn't match the file we loaded).
            if geom.type != mujoco.mjtGeom.mjGEOM_MESH and model_geom is not None:
                geom_pos = np.asarray(model.geom_pos[model_geom.id], dtype=np.float32)
                geom_quat = np.asarray(model.geom_quat[model_geom.id], dtype=np.float32)
            else:
                geom_pos = np.asarray(geom.pos, dtype=np.float32)
                geom_quat = np.asarray(geom.quat, dtype=np.float32)

            try:
                handle = server.scene.add_mesh_trimesh(
                    geom_path,
                    tm,
                    position=geom_pos,
                    wxyz=geom_quat,
                )

                is_collision = False
                try:
                    group_val = (
                        int(np.asarray(geom.group).ravel()[0])
                        if hasattr(geom, "group")
                        else 0
                    )
                    if group_val >= 3:
                        is_collision = True
                except Exception:
                    pass
                if "collision" in geom_name.lower():
                    is_collision = True

                if is_collision:
                    handle.visible = (
                        _STATE.scene_checkboxes["collision_meshes"].value
                        if "collision_meshes" in _STATE.scene_checkboxes
                        else False
                    )
                    _STATE.collision_geom_handles.append(handle)
                else:
                    handle.visible = (
                        _STATE.scene_checkboxes["visual_meshes"].value
                        if "visual_meshes" in _STATE.scene_checkboxes
                        else True
                    )
                    _STATE.visual_geom_handles.append(handle)

            except Exception as exc:
                loguru.logger.warning(f"Viser: failed to add geom '{geom_name}': {exc}")

    _STATE.body_handles = body_entity_and_ids

    # Build reference meshes (transparent blue)
    ref_body_entity_and_ids: list[tuple[Any, int]] = []

    if not build_ref:
        _STATE.ref_body_handles = ref_body_entity_and_ids
        return body_entity_and_ids

    ref_color = np.array([0.0, 0.0, 1.0, 0.25], dtype=np.float32)

    for body in spec.bodies[1:]:
        body_name = body.name
        body_path = f"{entity_root}_ref/{body_name}"
        body_handle = server.scene.add_frame(body_path, show_axes=False)

        try:
            body_id = model.body(body_name).id
        except Exception:
            body_id = body.id

        ref_body_entity_and_ids.append((body_handle, body_id))

        for geom in body.geoms:
            geom_name = geom.name

            try:
                gv = (
                    int(np.asarray(geom.group).ravel()[0])
                    if hasattr(geom, "group")
                    else 0
                )
                if gv >= 3:
                    continue
            except Exception:
                pass

            geom_path = f"{body_path}/geom_{geom_name}"

            model_geom = None
            try:
                model_geom = model.geom(geom.name)
            except Exception:
                model_geom = None

            if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
                tm = None
                mesh_file = _get_mesh_file(spec, geom)
                mesh_scale = _get_mesh_scale(spec, geom)
                if mesh_file is not None and mesh_file.exists():
                    try:
                        tm = trimesh.load(str(mesh_file), force="mesh")
                        if isinstance(tm, trimesh.Scene):
                            tm = tm.to_mesh()
                    except Exception:
                        tm = None
                if tm is None:
                    try:
                        geom_id = model_geom.id if model_geom is not None else -1
                        tm = _mujoco_mesh_to_trimesh(model, geom_id)
                    except Exception:
                        tm = None
                if tm is None:
                    continue
                if mesh_scale is not None:
                    try:
                        tm.apply_scale(mesh_scale)
                    except Exception:
                        pass

                # Override to transparent blue
                _set_mesh_color(tm, ref_color)
            else:
                size = geom.size
                if model_geom is not None:
                    try:
                        model_size = model.geom_size[model_geom.id]
                        if np.any(np.asarray(size) == 0) or np.any(np.isnan(size)):
                            size = model_size
                    except Exception:
                        pass
                tm = _trimesh_from_primitive(geom.type, size, rgba=ref_color)

            if tm is None:
                continue

            if geom.type != mujoco.mjtGeom.mjGEOM_MESH and model_geom is not None:
                geom_pos = np.asarray(model.geom_pos[model_geom.id], dtype=np.float32)
                geom_quat = np.asarray(model.geom_quat[model_geom.id], dtype=np.float32)
            else:
                geom_pos = np.asarray(geom.pos, dtype=np.float32)
                geom_quat = np.asarray(geom.quat, dtype=np.float32)

            try:
                handle = server.scene.add_mesh_trimesh(
                    geom_path,
                    tm,
                    position=geom_pos,
                    wxyz=geom_quat,
                )
                _STATE.ref_geom_handles.append(handle)
                handle.visible = _STATE.scene_checkboxes["ref"].value
            except Exception as exc:
                loguru.logger.warning(
                    f"Viser: failed to add ref geom '{geom_name}': {exc}"
                )

    _STATE.ref_body_handles = ref_body_entity_and_ids

    return body_entity_and_ids


def build_and_log_scene(
    xml_path: Path, entity_root: str = "mujoco"
) -> tuple[mujoco.MjSpec, mujoco.MjModel, list[tuple[Any, int]]]:
    """Load MJCF, create static geometry, and log it to Viser."""
    spec = mujoco.MjSpec.from_file(str(xml_path))
    _ensure_names(spec)
    model = spec.compile()
    body_entity_and_ids = build_and_log_scene_from_spec(
        spec=spec,
        model=model,
        xml_path=xml_path,
        entity_root=entity_root,
    )
    return spec, model, body_entity_and_ids


def log_scene_from_npz(npz_path: Path) -> list[tuple[Any, int]]:
    """Viser does not support baked .npz scenes; noop for compatibility."""
    loguru.logger.warning(
        f"Viser: log_scene_from_npz is not supported (requested {npz_path})."
    )
    return []


# -----------------------------
# Logging helpers
# -----------------------------


def log_frame(
    data: mujoco.MjData,
    sim_time: float,
    viewer_body_entity_and_ids: list[tuple[Any, int]] = [],
    data_ref: mujoco.MjData | None = None,
    show_ui: bool = True,
    playback_fps: float = 50.0,
) -> None:
    del sim_time
    if _STATE.server is None or not viewer_body_entity_and_ids:
        return

    server = _STATE.server

    # Record history for timeline
    frame_state = {}
    for handle, bid in viewer_body_entity_and_ids:
        pos = np.asarray(data.xpos[bid], dtype=np.float32)
        quat = np.asarray(data.xquat[bid], dtype=np.float32)
        frame_state[bid] = (pos, quat)

    if data_ref is not None:
        for handle, bid in _STATE.ref_body_handles:
            # We index by the handle instance to avoid overlapping with identical
            # body_ids from the main scene, since both scenes share the exact same
            # MJCF ID map.
            pos = np.asarray(data_ref.xpos[bid], dtype=np.float32)
            quat = np.asarray(data_ref.xquat[bid], dtype=np.float32)
            frame_state[handle.name] = (pos, quat)

    _STATE.frame_history.append(frame_state)
    current_frame = len(_STATE.frame_history) - 1

    # Instantiate timeline UI on first frame if not present
    if show_ui and _STATE.playback_slider is None:
        folder = server.gui.add_folder("Timeline")
        with folder:
            _STATE.playback_slider = server.gui.add_slider(
                "Frame",
                min=0,
                max=max(1, current_frame),
                step=1,
                initial_value=0,
            )
            _STATE.playback_fps_slider = server.gui.add_slider(
                "FPS", min=1, max=120, step=1, initial_value=int(playback_fps)
            )

            btn_rev = server.gui.add_button("Play Backward")
            btn_pause = server.gui.add_button("Pause")
            btn_fwd = server.gui.add_button("Play Forward")

            @btn_rev.on_click
            def _(_) -> None:
                _STATE.playback_speed = -1

            @btn_pause.on_click
            def _(_) -> None:
                _STATE.playback_speed = 0

            @btn_fwd.on_click
            def _(_) -> None:
                _STATE.playback_speed = 1

            @_STATE.playback_slider.on_update
            def _on_playback_update(_) -> None:
                _render_frame()

            def playback_loop():
                while True:
                    if _STATE.playback_fps_slider is not None:
                        fps = max(1.0, float(_STATE.playback_fps_slider.value))
                        sleep_time = 1.0 / fps
                    else:
                        sleep_time = 1.0 / 50.0

                    if (
                        _STATE.playback_speed != 0
                        and _STATE.playback_slider is not None
                    ):
                        new_val = (
                            int(_STATE.playback_slider.value) + _STATE.playback_speed
                        )

                        if new_val >= _STATE.playback_slider.max:
                            new_val = int(_STATE.playback_slider.max)
                        elif new_val <= 0:
                            new_val = 0

                        _STATE.playback_slider.value = new_val
                    time.sleep(sleep_time)

            _STATE.playback_thread = threading.Thread(target=playback_loop, daemon=True)
            _STATE.playback_thread.start()

    # Update slider max bound
    if _STATE.playback_slider is not None:
        _STATE.playback_slider.max = max(1, current_frame)


def _render_frame(
    viewer_body_entity_and_ids: list[tuple[Any, int]] | None = None,
) -> None:
    """Updates geometry natively or triggered by timeline scrub."""
    if _STATE.server is None or _STATE.playback_slider is None:
        return

    frame_idx = int(_STATE.playback_slider.value)
    if frame_idx >= len(_STATE.frame_history) or frame_idx < 0:
        return

    frame_state = _STATE.frame_history[frame_idx]

    if viewer_body_entity_and_ids is None:
        viewer_body_entity_and_ids = _STATE.body_handles

    server = _STATE.server
    with server.atomic():
        for handle, bid in viewer_body_entity_and_ids:
            if bid not in frame_state:
                continue
            pos, quat = frame_state[bid]
            try:
                handle.position = tuple(pos)
                handle.wxyz = tuple(quat)
            except Exception:
                try:
                    handle.position = pos
                    handle.wxyz = quat
                except Exception:
                    pass

        for handle, bid in _STATE.ref_body_handles:
            if handle.name not in frame_state:
                continue
            pos, quat = frame_state[handle.name]
            try:
                handle.position = tuple(pos)
                handle.wxyz = tuple(quat)
            except Exception:
                try:
                    handle.position = pos
                    handle.wxyz = quat
                except Exception:
                    pass

    # Fire frame-change callbacks
    for cb in _STATE.frame_change_callbacks:
        try:
            cb(frame_idx)
        except Exception as exc:
            loguru.logger.warning(f"Frame callback error: {exc}")

    # Check if we need to update trace geometries based on closest history
    # Find newest trace log <= current restricted frame_idx
    valid_trace_idxs = [i for i in _STATE.trace_history.keys() if i >= frame_idx]
    if valid_trace_idxs:
        newest_trace_idx = min(valid_trace_idxs)
        trace_id, traces, trace_ref, trace_cost, num_iters = _STATE.trace_history[
            newest_trace_idx
        ]

        # Check if they are already active
        if getattr(_STATE, "_active_trace_id", -1) != trace_id:
            _STATE._active_trace_id = trace_id
            _STATE.last_traces = traces
            _STATE.last_trace_ref = trace_ref
            _update_traces_geometry(trace_id, traces, trace_ref, trace_cost, num_iters)
            _render_traces()

    try:
        server.flush()
    except Exception:
        pass


def _compute_trace_colors(
    I: int, N: int, K: int, alt_colors: bool = False
) -> np.ndarray:
    colors = np.zeros([I, N, K, 3])
    white = np.array([255, 255, 255])

    if alt_colors:
        # object: green, robot: yellow
        object_color = np.array([0, 255, 0])
        robot_color = np.array([255, 255, 0])
    else:
        # object: red, robot: blue
        object_color = np.array([255, 0, 0])
        robot_color = np.array([0, 0, 255])

    for i in range(I):
        for k in range(K):
            if I == 1:
                colors[i, :, k, :] = object_color if k < 1 else robot_color
            else:
                if k < 1:
                    colors[i, :, k, :] = (1 - i / (I - 1)) * white + (
                        i / (I - 1)
                    ) * object_color
                else:
                    colors[i, :, k, :] = (1 - i / (I - 1)) * white + (
                        i / (I - 1)
                    ) * robot_color
    return colors.reshape(I * N * K, 3).astype(np.uint8)


def _compute_cost_colors(I: int, N: int, K: int, trace_cost: np.ndarray) -> np.ndarray:
    import matplotlib.cm as cm

    colors = np.zeros([I, N, K, 3])
    for i in range(I):
        c_min, c_max = trace_cost[i].min(), trace_cost[i].max()
        if c_max == c_min:
            norm_cost = np.zeros_like(trace_cost[i])
        else:
            denom = c_max - c_min
            norm_cost = (trace_cost[i] - c_min) / denom
        mapped = cm.viridis(norm_cost)[:, :3] * 255
        for k in range(K):
            colors[i, :, k, :] = mapped
    return colors.astype(np.uint8)


def _update_traces_geometry(
    trace_id: int,
    a: np.ndarray,
    trace_ref: np.ndarray | None,
    trace_cost: np.ndarray | None = None,
    num_iters: int | None = None,
) -> None:
    I, N, P, K, _ = a.shape
    # Rearrange to (I, N, K, P, 3)
    a = a.transpose(0, 1, 3, 2, 4)
    # Use actual iteration count for color gradient (optimizer may pad with zeros)
    color_I = num_iters if num_iters is not None and num_iters <= I else I
    colors_all = _compute_trace_colors(color_I, N, K, alt_colors=False).reshape(
        color_I, N, K, 3
    )
    # Pad colors to full I by repeating the last (fully saturated) color
    if color_I < I:
        pad = np.tile(colors_all[-1:], (I - color_I, 1, 1, 1))
        colors_all = np.concatenate([colors_all, pad], axis=0)
    colors_cost = None
    if trace_cost is not None:
        colors_cost = _compute_cost_colors(I, N, K, trace_cost).reshape(I, N, K, 3)
    server = _STATE.server

    with server.atomic():
        # Re-draw geometries by iteration and site
        for i in range(I):
            for k in range(K):
                group_name = "object" if k == 0 else f"robot/site_{k}"
                handle_key = (group_name, i)

                k_strips = a[i, :, k, :, :].reshape(-1, P, 3)
                k_segments = np.stack(
                    [k_strips[:, :-1, :], k_strips[:, 1:, :]], axis=2
                ).reshape(-1, 2, 3)

                k_colors = colors_all[i, :, k, :].reshape(-1, 3)
                k_colors = np.repeat(k_colors, repeats=P - 1, axis=0)
                k_colors = np.repeat(k_colors[:, None, :], repeats=2, axis=1)

                _STATE.trace_handles[handle_key] = server.scene.add_line_segments(
                    f"{_STATE.entity_root}/traces/{group_name}/iter_{i}",
                    k_segments,
                    k_colors,
                    line_width=2.0,
                    visible=False,
                )

                if colors_cost is not None:
                    cost_handle_key = (f"{group_name}_cost", i)
                    k_colors_cost = colors_cost[i, :, k, :].reshape(-1, 3)
                    k_colors_cost = np.repeat(k_colors_cost, repeats=P - 1, axis=0)
                    k_colors_cost = np.repeat(
                        k_colors_cost[:, None, :], repeats=2, axis=1
                    )
                    _STATE.trace_handles[cost_handle_key] = (
                        server.scene.add_line_segments(
                            f"{_STATE.entity_root}/traces/{group_name}_cost/iter_{i}",
                            k_segments,
                            k_colors_cost,
                            line_width=2.0,
                            visible=False,
                        )
                    )

        # Re-draw reference traces
        if trace_ref is not None:
            ref_a = np.asarray(trace_ref, dtype=np.float32)
            if ref_a.ndim == 5 and ref_a.shape[-1] == 3:
                # (1, 1, H, K, 3) -> (1, 1, K, H, 3)
                ref_a = ref_a.transpose(0, 1, 3, 2, 4)
                ref_K = ref_a.shape[2]
                ref_P = ref_a.shape[3]

                ref_colors_all = _compute_trace_colors(
                    1, 1, ref_K, alt_colors=True
                ).reshape(1, 1, ref_K, 3)

                for k in range(ref_K):
                    group_name = "object_ref" if k == 0 else f"robot_ref/site_{k}"
                    handle_key = (group_name, 0)

                    k_strips = ref_a[:, :, k, :, :].reshape(-1, ref_P, 3)
                    k_segments = np.stack(
                        [k_strips[:, :-1, :], k_strips[:, 1:, :]], axis=2
                    ).reshape(-1, 2, 3)

                    k_colors = ref_colors_all[:, :, k, :].reshape(-1, 3)
                    k_colors = np.repeat(k_colors, repeats=ref_P - 1, axis=0)
                    k_colors = np.repeat(k_colors[:, None, :], repeats=2, axis=1)

                    _STATE.trace_handles[handle_key] = server.scene.add_line_segments(
                        f"{_STATE.entity_root}/traces/ref/{group_name}",
                        k_segments,
                        k_colors,
                        line_width=2.0,
                        visible=False,
                    )


def _render_traces() -> None:
    if _STATE.server is None or _STATE.last_traces is None:
        return

    I = _STATE.last_traces.shape[0]

    if _STATE.trace_slider is not None:
        val = _STATE.trace_slider.value
        min_i, max_i = int(val[0]), int(val[1])
    else:
        min_i, max_i = 0, I

    min_i = max(0, min(min_i, I))
    max_i = max(min_i, min(max_i, I))

    # Fast boolean visibility toggling
    show_obj = (
        _STATE.trace_checkboxes.get("object").value
        if _STATE.trace_checkboxes and "object" in _STATE.trace_checkboxes
        else True
    )
    show_rob = (
        _STATE.trace_checkboxes.get("robot").value
        if _STATE.trace_checkboxes and "robot" in _STATE.trace_checkboxes
        else True
    )
    show_obj_ref = (
        _STATE.trace_checkboxes.get("object_ref").value
        if _STATE.trace_checkboxes and "object_ref" in _STATE.trace_checkboxes
        else True
    )
    show_rob_ref = (
        _STATE.trace_checkboxes.get("robot_ref").value
        if _STATE.trace_checkboxes and "robot_ref" in _STATE.trace_checkboxes
        else True
    )
    see_cost = (
        _STATE.trace_checkboxes.get("see_cost").value
        if _STATE.trace_checkboxes and "see_cost" in _STATE.trace_checkboxes
        else False
    )

    for handle_key, handle in _STATE.trace_handles.items():
        if isinstance(handle_key, tuple) and len(handle_key) == 2:
            group_name, i = handle_key

            is_ref = "ref" in group_name

            # Base visibility mask from slider
            visible = True if is_ref else (min_i <= i < max_i)

            # Additional mask from checkboxes
            if "object" in group_name:
                if is_ref and not show_obj_ref:
                    visible = False
                elif not is_ref and not show_obj:
                    visible = False
            elif "robot" in group_name:
                if is_ref and not show_rob_ref:
                    visible = False
                elif not is_ref and not show_rob:
                    visible = False

            if not is_ref:
                is_cost_handle = group_name.endswith("_cost")
                has_cost_handles = any(
                    k[0].endswith("_cost")
                    for k in _STATE.trace_handles
                    if isinstance(k, tuple)
                )
                if has_cost_handles and see_cost and not is_cost_handle:
                    visible = False
                elif not see_cost and is_cost_handle:
                    visible = False

            handle.visible = visible

    try:
        _STATE.server.flush()
    except Exception:
        pass


def log_traces_from_info(
    traces: np.ndarray,
    trace_ref: np.ndarray | None = None,
    trace_cost: np.ndarray | None = None,
    sim_time: float = 0.0,
    show_ui: bool = True,
    num_iters: int | None = None,
) -> None:
    del sim_time
    if _STATE.server is None:
        return

    a = np.asarray(traces, dtype=np.float32)
    if a.ndim != 5 or a.shape[-1] != 3:
        loguru.logger.warning(
            f"Viser: skip trace logging with incompatible shape {a.shape}"
        )
        return

    I, N, P, K, _ = a.shape
    if P < 2:
        return

    # Cache into global timeline
    current_frame = max(0, len(_STATE.frame_history) - 1)
    trace_id = _STATE.trace_id_counter
    _STATE.trace_id_counter += 1
    _STATE.trace_history[current_frame] = (
        trace_id,
        a,
        trace_ref,
        trace_cost,
        num_iters,
    )

    if show_ui and _STATE.trace_slider is None:
        folder = _STATE.server.gui.add_folder("Traces")
        with folder:
            max_i = max(1, I)
            _STATE.trace_slider = _STATE.server.gui.add_multi_slider(
                "Trace Iters",
                min=0,
                max=max_i,
                step=1,
                initial_value=(0, max_i),
            )

            # Add checkboxes
            _STATE.trace_checkboxes["object"] = _STATE.server.gui.add_checkbox(
                "Object", initial_value=True
            )
            _STATE.trace_checkboxes["robot"] = _STATE.server.gui.add_checkbox(
                "Robot", initial_value=True
            )
            _STATE.trace_checkboxes["object_ref"] = _STATE.server.gui.add_checkbox(
                "Object Ref", initial_value=True
            )
            _STATE.trace_checkboxes["robot_ref"] = _STATE.server.gui.add_checkbox(
                "Robot Ref", initial_value=True
            )
            _STATE.trace_checkboxes["see_cost"] = _STATE.server.gui.add_checkbox(
                "Cost", initial_value=False
            )

            @_STATE.trace_slider.on_update
            def _on_slider_update(_) -> None:
                _render_traces()

            @_STATE.trace_checkboxes["object"].on_update
            def _on_obj_toggled(_) -> None:
                _render_traces()

            @_STATE.trace_checkboxes["robot"].on_update
            def _on_rob_toggled(_) -> None:
                _render_traces()

            @_STATE.trace_checkboxes["object_ref"].on_update
            def _on_obj_ref_toggled(_) -> None:
                _render_traces()

            @_STATE.trace_checkboxes["robot_ref"].on_update
            def _on_rob_ref_toggled(_) -> None:
                _render_traces()

            @_STATE.trace_checkboxes["see_cost"].on_update
            def _on_cost_toggled(_) -> None:
                _render_traces()
    else:
        max_i = max(1, I)
        if _STATE.trace_slider.max != max_i:
            _STATE.trace_slider.max = max_i

    # Only update geometry and render immediately if the timeline slider is
    # already exactly on the newest frame
    if (
        _STATE.playback_slider is not None
        and int(_STATE.playback_slider.value) == current_frame
    ):
        _STATE._active_trace_id = trace_id
        _STATE.last_traces = a
        _STATE.last_trace_ref = trace_ref
        _update_traces_geometry(trace_id, a, trace_ref, trace_cost, num_iters)
        _render_traces()
